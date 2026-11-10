#!/usr/bin/env python3
"""Faithfulness F of each branch's forced first tool call.

For every branch whose action is a tool, F is judged on ``tool_calls[0]`` (the forced call) against the answer
regions of ``rq1_annotations.py``, on the 0–1000 scale both sides share:
- crop_zoomin: F=1 when at least half of some region's area lies inside the applied crop box;
- draw_bbox: F=1 when one drawn box and some region have overlap coefficient ≥ 0.5 (overlap / the smaller area, so
  a box inside a large region also counts);
- mark_dots: F=1 when one point lies inside some region;
- crop_ocr_image: F=1 when the normalised answer is in the normalised output (a numeric answer must not touch
  other digits, so "2" does not match "2020"), or, for a numeric answer, when a number in the output has the same
  value (so "3.66e+07" matches "36,600,000"). Needs no annotation. Not for plotqa: its dot charts rarely print the
  value, so plotqa OCR goes to the LLM judge.
gqa (``--gqa-regions``):
- ``all`` (default): F=1 when every referenced object is visible. Object o is visible in area V when
  overlap / min(area(o), area(V)) ≥ 0.5, so a crop inside a large object (the room) shows it. V is the crop box, or
  any one drawn box; for mark_dots every object needs a point inside;
- ``union``: one region, the union box of the referenced objects. crop_zoomin/draw_bbox must hold half of its area
  (draw_bbox keeps this area rule here, since the overlap coefficient with one large box is uninformative), mark_dots
  a point inside it. Fails exact framings when a large object is also referenced;
- ``objects``: F=1 when any one referenced object is covered (too lenient for relation questions).
A region of zero area (a line-chart point) counts as covered when its centre is inside. F is None, with ``reason``,
when it cannot be judged: no forced call, call on a tool output image (``image_index`` ≠ 0), problem not linked or
without a region, or a tool the rules do not cover (the LLM-judge tools). A call that returned an error has F=0.

Writes OUT/branch_f.jsonl (one row per tool branch, with the region or text that decided it), OUT/pair_f.jsonl
(per decision point and tool: answer-region candidates, judged branches, mean F, and accuracy of F=1 and F=0 branches among valid
continuations, ``analyze_branches`` exclusions) and OUT/summary.json.

    python value_head/rq1_faithfulness.py --annotations ANN/annotations.jsonl \
        --branches R1/branches/branches.jsonl --branches R2/branches/branches.jsonl --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branches import EXCLUDED, FUNCTION_RE  # noqa: E402
from rq1_annotations import norm_text, parse_number  # noqa: E402

REGION_TOOLS = {"crop_zoomin": "applied_crop_box", "draw_bbox": "applied_bboxes", "mark_dots": "applied_points"}


def covered(region: list[float], box: list[float]) -> float:
    """Share of ``region`` inside ``box``; for a zero-area region, 1.0 when its centre is inside."""
    rx0, ry0, rx1, ry1 = region
    bx0, by0, bx1, by1 = box
    area = (rx1 - rx0) * (ry1 - ry0)
    if area <= 0:
        cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        return float(bx0 <= cx <= bx1 and by0 <= cy <= by1)
    inter = max(0.0, min(rx1, bx1) - max(rx0, bx0)) * max(0.0, min(ry1, by1) - max(ry0, by0))
    return inter / area


def judge_region(tool: str, meta: dict, regions: list[dict], overlap: bool = True) -> tuple[int, dict]:
    """``overlap``: draw_bbox by overlap coefficient; False keeps the half-the-region's-area rule (gqa union)."""
    applied = meta[REGION_TOOLS[tool]]
    if tool == "crop_zoomin":
        boxes = [applied["normalized_1000"]]
    elif tool == "draw_bbox":
        boxes = [b["normalized_1000"] for b in applied]
    else:
        points = [p["normalized_1000"] for p in applied]
        for i, r in enumerate(regions):
            for x, y in points:
                if r["box"][0] <= x <= r["box"][2] and r["box"][1] <= y <= r["box"][3]:
                    return 1, {"region": i, "point": [x, y]}
        return 0, {"points": points}
    def score(region: list[float], box: list[float]) -> float:
        zero_area = (region[2] - region[0]) * (region[3] - region[1]) <= 0
        return overlap_coefficient(region, box) if tool == "draw_bbox" and overlap and not zero_area else covered(region, box)

    best = max(((score(r["box"], b), i, j) for i, r in enumerate(regions) for j, b in enumerate(boxes)), default=(0.0, None, None))
    return int(best[0] >= 0.5), {"region": best[1], "box": boxes[best[2]] if best[2] is not None else None, "coverage": round(best[0], 3)}


def overlap_coefficient(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def judge_all_visible(tool: str, meta: dict, objects: list[dict]) -> tuple[int, dict]:
    """GQA default: every referenced object visible in the crop, in some drawn box, or holding some point."""
    applied = meta[REGION_TOOLS[tool]]
    if tool == "mark_dots":
        points = [p["normalized_1000"] for p in applied]
        hit = [any(o["box"][0] <= x <= o["box"][2] and o["box"][1] <= y <= o["box"][3] for x, y in points) for o in objects]
        return int(all(hit)), {"objects_hit": hit, "points": points}
    boxes = [applied["normalized_1000"]] if tool == "crop_zoomin" else [b["normalized_1000"] for b in applied]
    best = [max(overlap_coefficient(o["box"], b) for b in boxes) for o in objects]
    return int(all(v >= 0.5 for v in best)), {"overlap": [round(v, 3) for v in best], "boxes": boxes}


def answer_in_text(answer: str, text: str) -> bool:
    a, t = norm_text(answer), norm_text(text)
    if not a:
        return False
    if re.fullmatch(r"-?[\d.]+%?", a):
        if re.search(rf"(?<![\d.]){re.escape(a)}(?![\d])", t):
            return True
    elif a in t:
        return True
    number = parse_number(answer)
    if number is None:
        return False
    values = re.findall(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?", text.lower())
    return any(abs(float(v.replace(",", "")) - number[0]) <= 1e-6 * max(1.0, abs(number[0])) for v in values)


def judge(row: dict, problem: dict | None, gqa_regions: str = "all") -> tuple[int | None, str, dict]:
    calls = row["tool_calls"]
    if not calls or calls[0]["name"] != row["action"]:
        return None, "no forced call", {}
    call, tool = calls[0], row["action"]
    if call["arguments"].get("image_index", 0) != 0:
        return None, "call on a tool output image", {}
    if tool not in REGION_TOOLS and tool != "crop_ocr_image":
        return None, "LLM judge tool", {}
    if tool == "crop_ocr_image" and row["dataset"] == "plotqa":
        return None, "LLM judge (plotqa OCR)", {}
    if tool in REGION_TOOLS and (problem is None or not problem["linked"]):
        return None, "problem not linked", {}
    if tool in REGION_TOOLS and not problem["regions"]:
        return None, f"no region ({problem['region_note']})", {}
    if call["error"]:
        return 0, "tool error", {"error": str(call["error"])[:200]}
    if tool == "crop_ocr_image":
        text = re.sub(r"</?tool_response>", "", call["observation"] or "").strip()
        return int(answer_in_text(problem["answer"], text)), "answer string", {"output": text[:300]}
    objects = problem["source"]["objects"] if problem["dataset"].startswith("gqa") else None
    if objects and gqa_regions == "all":
        f, basis = judge_all_visible(tool, call["meta"], objects)
    elif objects and gqa_regions == "objects":
        f, basis = judge_region(tool, call["meta"], objects)
    else:  # the gqa union keeps the area rule: one large box makes the overlap coefficient meaningless
        f, basis = judge_region(tool, call["meta"], problem["regions"], overlap=not problem["dataset"].startswith("gqa"))
    return f, "answer region", basis


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness of the forced first tool call per branch.")
    parser.add_argument("--annotations", required=True, type=Path, help="rq1_annotations.py output annotations.jsonl")
    parser.add_argument("--branches", action="append", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gqa-regions", choices=("all", "union", "objects"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    problems = {p["id"]: p for p in map(json.loads, open(args.annotations))}
    answers = {}
    for path in args.branches:  # crop_ocr_image needs only the answer, also for problems without annotations
        for line in open(path.parent.parent / "selected_points.jsonl"):
            d = json.loads(line)
            answers.setdefault(d["id"], {"answer": d["answer"], "linked": False, "regions": [], "region_note": None})

    pairs = defaultdict(lambda: {"judged": 0, "f1": 0, "valid": {0: 0, 1: 0}, "correct": {0: 0, 1: 0}, "reasons": Counter()})
    with open(args.out / "branch_f.jsonl", "w") as handle:
        for path in args.branches:
            for line in open(path):
                row = json.loads(line)
                if row["action"] == "answer_now":
                    continue
                f, reason, basis = judge(row, problems.get(row["id"]) or answers.get(row["id"]), args.gqa_regions)
                handle.write(json.dumps({k: row[k] for k in ("dp_id", "id", "dataset", "action", "k")}
                                        | {"f": f, "reason": reason, "basis": basis}, ensure_ascii=False) + "\n")
                if row["first_action"] == "tool_call_unparsed" and (m := FUNCTION_RE.search(row["text"])) and m.group(1).strip() == row["action"]:
                    row["finish"], row["correct"] = "length" if row["tokens"] >= 4096 else "no_answer", False
                pair = pairs[(row["dp_id"], row["action"], row["dataset"])]
                pair["reasons"][reason] += 1
                if f is not None:
                    pair["judged"] += 1
                    pair["f1"] += f
                    if row["finish"] not in EXCLUDED:
                        pair["valid"][f] += 1
                        pair["correct"][f] += bool(row["correct"])

    summary = defaultdict(lambda: {"pairs": 0, "pairs_judged": 0, "branches": Counter(), "f1": 0, "judged": 0})
    with open(args.out / "pair_f.jsonl", "w") as handle:
        for (dp_id, action, dataset), p in sorted(pairs.items()):
            acc = {f"acc_f{f}": round(p["correct"][f] / p["valid"][f], 4) if p["valid"][f] else None for f in (0, 1)}
            problem = problems.get(dp_id.split(":", 1)[0], {})
            # Answer-region candidates (any covered gives F=1); report single-candidate problems separately.
            candidates = len(problem.get("regions", [])) if action in REGION_TOOLS else None
            handle.write(json.dumps({"dp_id": dp_id, "action": action, "dataset": dataset, "candidates": candidates, "judged": p["judged"],
                                     "f_mean": round(p["f1"] / p["judged"], 4) if p["judged"] else None,
                                     "valid_f0": p["valid"][0], "valid_f1": p["valid"][1], **acc,
                                     "reasons": dict(p["reasons"])}) + "\n")
            s = summary[f"{dataset}/{action}"]
            s["pairs"] += 1
            s["pairs_judged"] += p["judged"] > 0
            s["branches"].update(p["reasons"])
            s["f1"] += p["f1"]
            s["judged"] += p["judged"]
    out = {key: {"pairs": s["pairs"], "pairs_judged": s["pairs_judged"], "branches": dict(s["branches"]),
                 "f_rate": round(s["f1"] / s["judged"], 4) if s["judged"] else None} for key, s in sorted(summary.items())}
    (args.out / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
