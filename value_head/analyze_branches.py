#!/usr/bin/env python3
"""Labels and statistics from a branch run (``branch_rollout.py finalize`` output).

    /root/miniconda3/envs/research/bin/python value_head/analyze_branches.py --run RUN_DIR --out OUT.json

RUN_DIR holds ``points/first_turns.jsonl`` and ``branches/branches.jsonl``; ``--branches FILE`` (repeatable) pools
more continuations of the same decision points, e.g. a think_more run. A continuation counts toward its
(decision point, action) label unless it did not take the forced action (``not_followed``) or was left
unfinished (``parked_unfinished``); truncation, tool errors and the tool-call limit are outcomes of the
action and count as failures. Writes ``labels.jsonl`` next to OUT (s successes out of n per point and
action) and OUT with:
- first turns: how often the policy opens a decision point, per dataset;
- per dataset and action: success rate, followed rate, finish reasons, tokens per continuation;
- Δ = q(tool) - q(answer_now) per (point, tool): share positive / zero / negative, mean |Δ|;
- split-half reliability of Δ for K = 4, 8, 16 (first K continuations, halves by even/odd k, Spearman,
  Spearman-Brown corrected); its square root bounds the rank correlation any predictor can reach against
  labels estimated from K continuations;
- with think_more continuations, the same Δ statistics under ``by_reference`` for the other references
  (``REFERENCES``), and q(think_more) − q(answer_now) per point;
- answer_now statement length (characters between the forced block and <answer>);
- crop_ocr_image observations reporting no text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman's rho (Pearson correlation of average ranks, so ties are handled)."""
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy)

EXCLUDED = {"not_followed", "parked_unfinished"}
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")
NO_TEXT_RE = re.compile(r"没有|未包含|无法|no (visible |readable )?text|does not contain any text", re.IGNORECASE)


def rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


# What a tool's Δ is measured against. "best" is the main definition: the better of answer_now and
# think_more, cross-fitted (one half of the k's picks the better one, the other half measures both sides, and
# the two directions are averaged) so that taking the max does not bias Δ downwards; "best_raw" is the plain max.
BASES = ("answer_now", "think_more")
REFERENCES = ("answer_now", "think_more", "best", "best_raw")


def success_rates(groups: dict, ks: set[int] | None = None) -> dict[tuple[str, str], float]:
    """q per (dp_id, action) from continuations whose k is in ``ks`` (all if None)."""
    q = {}
    for (dp, action), rows in groups.items():
        valid = [r for r in rows if r["finish"] not in EXCLUDED and (ks is None or r["k"] in ks)]
        if valid:
            q[(dp, action)] = sum(bool(r["correct"]) for r in valid) / len(valid)
    return q


def deltas(groups: dict, ks: set[int] | None = None, reference: str = "answer_now") -> dict[tuple[str, str], float]:
    """Δ per (dp_id, tool) from continuations whose k is in ``ks`` (all if None), against ``reference``."""
    out = {}
    if reference == "best":
        ordered = sorted(ks if ks is not None else {r["k"] for rows in groups.values() for r in rows})
        halves = [success_rates(groups, set(ordered[0::2])), success_rates(groups, set(ordered[1::2]))]
        both = defaultdict(list)
        for pick, measure in (halves, halves[::-1]):
            for (dp, action), value in measure.items():
                if action in BASES or not all((dp, b) in pick and (dp, b) in measure for b in BASES):
                    continue
                base = max(BASES, key=lambda b: pick[(dp, b)])  # a tie goes to answer_now
                both[(dp, action)].append(value - measure[(dp, base)])
        return {key: sum(v) / 2 for key, v in both.items() if len(v) == 2}
    q = success_rates(groups, ks)
    bases = BASES if reference == "best_raw" else (reference,)
    for (dp, action), value in q.items():
        if action not in BASES and all((dp, b) in q for b in bases):
            out[(dp, action)] = value - max(q[(dp, b)] for b in bases)
    return out


def reliability(groups: dict, k_total: int, dataset_of: dict, reference: str = "answer_now") -> dict:
    result = {}
    for k in (4, 8, 16):
        if k > k_total:
            continue
        a = deltas(groups, {i for i in range(k) if i % 2 == 0}, reference)
        b = deltas(groups, {i for i in range(k) if i % 2 == 1}, reference)
        keys = sorted(set(a) & set(b))
        entry = {}
        for name, subset in [("all", keys)] + [(d, [x for x in keys if dataset_of[x[0]] == d])
                                              for d in sorted(set(dataset_of.values()))]:
            xs, ys = [a[x] for x in subset], [b[x] for x in subset]
            if len(subset) < 5 or len(set(xs)) < 2 or len(set(ys)) < 2:
                entry[name] = None
                continue
            r = spearman(xs, ys)
            full = 2 * r / (1 + r) if r > -1 else None
            entry[name] = {"pairs": len(subset), "half_r": round(r, 3), "reliability": round(full, 3) if full is not None else None,
                           "ceiling": round(math.sqrt(full), 3) if full and full > 0 else None}
        result[f"K={k}"] = entry
    return result


def distribution(values_by_key: dict, dataset_of: dict) -> dict:
    """Share positive / zero / negative, mean |x|, max, min, overall and per dataset (key[0] or key is a dp_id)."""
    dp = lambda key: key[0] if isinstance(key, tuple) else key
    dist = {}
    for name in ["all"] + sorted(set(dataset_of.values())):
        values = [v for key, v in values_by_key.items() if name == "all" or dataset_of[dp(key)] == name]
        if values:
            dist[name] = {"pairs": len(values), "positive": rate(sum(v > 0 for v in values), len(values)),
                          "zero": rate(sum(v == 0 for v in values), len(values)),
                          "negative": rate(sum(v < 0 for v in values), len(values)),
                          "mean_abs": round(sum(abs(v) for v in values) / len(values), 3),
                          "max": round(max(values), 3), "min": round(min(values), 3)}
    return dist


def delta_report(groups: dict, k_total: int, dataset_of: dict, reference: str) -> dict:
    d_all = deltas(groups, reference=reference)
    by_tool = {}
    for tool in sorted({a for _, a in d_all}):
        values = [v for (dp, a), v in d_all.items() if a == tool]
        by_tool[tool] = {"pairs": len(values), "mean": round(sum(values) / len(values), 3),
                         "positive": rate(sum(v > 0 for v in values), len(values))}
    return {"delta_distribution": distribution(d_all, dataset_of), "delta_by_tool": by_tool,
            "reliability": reliability(groups, k_total, dataset_of, reference)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Labels and statistics from a branch run.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--branches", action="append", type=Path, default=[],
                        help="more branches.jsonl of the same decision points (e.g. think_more)")
    parser.add_argument("--run-branches", type=Path,
                        help="use this file instead of RUN/branches/branches.jsonl (e.g. regrade_branches.py output)")
    args = parser.parse_args()
    labels_path = args.out.with_name(args.out.stem + "_labels.jsonl")
    for path in (args.out, labels_path):
        if path.exists():
            raise SystemExit(f"{path} exists")

    first_path = args.run / "points/first_turns.jsonl"  # absent for runs built from later decision points
    first = [json.loads(line) for line in open(first_path)] if first_path.exists() else []
    rows = [json.loads(line) for path in [args.run_branches or args.run / "branches/branches.jsonl", *args.branches]
            for line in open(path)]
    reclassified = 0
    for row in rows:  # a cut-off or malformed call to the forced tool did follow it: reclassify such rows
        if row["first_action"] == "tool_call_unparsed" and (m := FUNCTION_RE.search(row["text"])) and m.group(1).strip() == row["action"]:
            row["first_action"], row["followed"], row["correct"] = row["action"], True, False
            row["finish"] = "length" if row["tokens"] >= 4096 else "no_answer"
            reclassified += 1
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dp_id"], row["action"])].append(row)
    if len({(dp, a, r["k"]) for (dp, a), members in groups.items() for r in members}) != len(rows):
        raise SystemExit("the same (dp_id, action, k) appears twice across the branch files")
    dataset_of = {row["dp_id"]: row["dataset"] for row in rows}
    k_total = max(row["k"] for row in rows) + 1

    with open(labels_path, "w") as handle:
        for (dp, action), members in sorted(groups.items()):
            valid = [r for r in members if r["finish"] not in EXCLUDED]
            handle.write(json.dumps({"dp_id": dp, "dataset": dataset_of[dp], "action": action, "n": len(valid),
                                     "s": sum(bool(r["correct"]) for r in valid), "k": len(members)}) + "\n")

    report = {"run": str(args.run), "continuations": len(rows), "decision_points": len(dataset_of), "k": k_total,
              "reclassified_cut_off_calls": reclassified}
    by_ds = defaultdict(Counter)
    for turn in first:
        by_ds[turn["dataset"]][turn["kind"]] += 1
    report["first_turns"] = {d: {"questions": sum(c.values()), "decision_point_rate": rate(c["decision_point"], sum(c.values())),
                                 **dict(c)} for d, c in sorted(by_ds.items())}

    per = defaultdict(list)
    for row in rows:
        per[(row["dataset"], row["action"])].append(row)
    actions = {}
    for (dataset, action), members in sorted(per.items()):
        valid = [r for r in members if r["finish"] not in EXCLUDED]
        actions.setdefault(dataset, {})[action] = {
            "continuations": len(members), "followed": rate(sum(r["followed"] for r in members), len(members)),
            "success": rate(sum(bool(r["correct"]) for r in valid), len(valid)),
            "finish": dict(Counter(r["finish"] for r in members)),
            "tool_errors": sum(1 for r in members for c in r["tool_calls"] if c["error"]),
            "mean_tokens": round(sum(r["tokens"] for r in members) / len(members)),
        }
    report["actions"] = actions

    report |= delta_report(groups, k_total, dataset_of, "answer_now")
    if any(a == "think_more" for _, a in groups):
        report["by_reference"] = {ref: delta_report(groups, k_total, dataset_of, ref) for ref in REFERENCES if ref != "answer_now"}
        q = success_rates(groups)
        gain = {dp: q[(dp, "think_more")] - q[(dp, "answer_now")] for dp, a in q if a == "think_more" and (dp, "answer_now") in q}
        report["think_more_minus_answer_now"] = distribution(gain, dataset_of)

    statements = defaultdict(list)
    for row in rows:
        if row["action"] == "answer_now" and row["statement_chars"] is not None:
            statements[row["dataset"]].append(row["statement_chars"])
    report["answer_now_statement_chars"] = {
        d: {"n": len(v), "median": sorted(v)[len(v) // 2], "p90": sorted(v)[int(len(v) * 0.9)], "over_500": sum(x > 500 for x in v)}
        for d, v in sorted(statements.items())}
    ocr = [c["observation"] or "" for row in rows for c in row["tool_calls"] if c["name"] == "crop_ocr_image"]
    report["ocr"] = {"calls": len(ocr), "no_text": sum(bool(NO_TEXT_RE.search(o)) for o in ocr),
                     "no_text_rate": rate(sum(bool(NO_TEXT_RE.search(o)) for o in ocr), len(ocr))}
    report["generated_tokens"] = sum(r["tokens"] for r in rows)

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("continuations", "decision_points", "k", "delta_distribution", "reliability", "ocr",
                                             "generated_tokens", "by_reference") if k in report}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
