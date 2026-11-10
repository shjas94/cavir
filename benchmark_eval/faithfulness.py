#!/usr/bin/env python3
"""Faithfulness of each natural solution's first tool call.

    python benchmark_eval/faithfulness.py --problems PROBLEMS.jsonl --out OUT

Reads OUT/natural.jsonl (``rollout.py natural``) and writes OUT/faithfulness.jsonl, one row per problem whose solution
called a tool: {index, dataset, faithful (1, 0 or None), reason, detail}. The first call is the first one of the first
assistant message with calls; the execute continuations all read its output, so there is one value per problem.
- VStarBench (crop_zoomin): 1 when the crop the tool applied (pixels of the original image) holds every answer object
  box of the benchmark's annotation completely. None when the problem has no matched box (``prepare.py``).
- DocVQA_VAL, ChartQA_TEST (crop_ocr_image): 1 when the text read holds one accepted answer, by the string rule of
  the crop_ocr_image faithfulness rule (``value_head/rq1_faithfulness.answer_in_text``).
A call that returned an error has faithful None (reason tool_error): no output to judge; it is counted, not judged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "value_head"))

from rq1_faithfulness import answer_in_text, covered  # noqa: E402


def judge(problem: dict, first: dict) -> tuple[int | None, str, dict]:
    if first["error"]:
        return None, "tool_error", {"error": first["error"], "text": (first.get("text") or "")[:200]}
    if problem["dataset"] == "VStarBench":
        if not problem.get("bboxes"):
            return None, "no_bbox", {}
        crop = first["meta"]["applied_crop_box"]["pixels"]
        shares = [covered([x, y, x + w, y + h], crop) for x, y, w, h in problem["bboxes"]]
        return int(all(s >= 1.0 for s in shares)), "crop", {"crop": crop, "covered": [round(s, 3) for s in shares]}
    hit = next((a for a in problem["answers"] if answer_in_text(a, first["text"] or "")), None)
    return int(hit is not None), "ocr_text", {"matched": hit, "text": (first["text"] or "")[:300]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness of the first tool call of each natural solution.")
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out / "faithfulness.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists")
    problems = {p["index"]: p for p in map(json.loads, open(args.problems))}
    rows = []
    for r in map(json.loads, open(args.out / "natural.jsonl")):
        if not r["results"]:
            continue
        faithful, reason, detail = judge(problems[r["index"]], r["results"][0])
        rows.append({"index": r["index"], "dataset": r["dataset"], "faithful": faithful, "reason": reason, "detail": detail})
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({d: dict(Counter(f"{r['faithful']} ({r['reason']})" for r in rows if r["dataset"] == d))
                      for d in sorted({r["dataset"] for r in rows})}, indent=1))


if __name__ == "__main__":
    main()
