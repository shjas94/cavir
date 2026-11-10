#!/usr/bin/env python3
"""Accuracy with and without the first tool call, for one model and benchmark.

    python benchmark_eval/analyze.py --out OUT

Reads OUT/natural.jsonl, OUT/branches.jsonl, OUT/scores/ (``score.py``) and OUT/faithfulness.jsonl, writes
OUT/analysis.json. Scores are VLMEvalKit's per-problem official scores in [0, 1]. For problem p and condition c
(execute or block), s_c(p) is the mean of its K=4 continuation scores; a problem whose solution called no tool has the
natural score in both conditions (the prediction files carry it), so its difference is 0.
- accuracy: per condition, the benchmark's official overall averaged over the 4 prediction files, and the same from the
  per-problem scores (mean over all problems of s_c(p)), with the official natural score.
- difference_all: mean over all problems of s_execute(p) - s_block(p), the change in benchmark accuracy.
- difference_called: the same mean over the problems whose solution called a tool.
- by_need: the called problems split by block continuations 1 and 2 (both full score: "solved_without_tool", else
  "may_fail_without_tool"); the difference uses all 4 execute continuations and block continuations 3 and 4, so the
  continuations that form a group are not the ones that measure it (regression to the mean).
- by_faithfulness: the called problems split by the first call's faithfulness (1, 0; None left out of the split only).
Each group of the two splits reports the difference and the accuracy of each condition over the same continuations
(by_need: execute 1-4, block 3-4; by_faithfulness: execute 1-4, block 1-4).
Every mean comes with a 95% interval from BOOTSTRAP resamples of problems. ``counts`` holds the
problems, called problems and their share, solutions that wrote a call but were not branched (``call_note``
of ``rollout.py``), block continuations without an answer, and continuations or solutions that ended by the length
limit, the tool-call limit or a rejected request.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

K = 4
BOOTSTRAP = 2000
SEED = 0


def interval(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = x[rng.integers(0, len(x), size=(BOOTSTRAP, len(x)))].mean(axis=1)
    return {"n": len(x), "mean": float(x.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy difference between executing and blocking the first tool call.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    target = args.out / "analysis.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    scores = {name: {r["index"]: r["score"] for r in map(json.loads, open(args.out / "scores" / f"{name}.jsonl"))}
              for name in ["natural"] + [f"{c}_r{k}" for c in ("execute", "block") for k in range(K)]}
    official = json.loads((args.out / "scores" / "official.json").read_text())
    natural = {r["index"]: r for r in map(json.loads, open(args.out / "natural.jsonl"))}
    branches = {r["index"]: r for r in map(json.loads, open(args.out / "branches.jsonl"))}
    faithful = {r["index"]: r["faithful"] for r in map(json.loads, open(args.out / "faithfulness.jsonl"))} \
        if (args.out / "faithfulness.jsonl").exists() else {}

    def mean_score(index: str, condition: str, ks=range(K)) -> float:
        return float(np.mean([scores[f"{condition}_r{k}"][index] for k in ks]))

    def group(members: list[str], block_ks=range(K)) -> dict:
        return {"difference": interval([mean_score(p, "execute") - mean_score(p, "block", block_ks) for p in members]),
                "execute": interval([mean_score(p, "execute") for p in members]),
                "block": interval([mean_score(p, "block", block_ks) for p in members])}

    problems = sorted(natural)
    called = [p for p in problems if p in branches]
    diff = {p: mean_score(p, "execute") - mean_score(p, "block") for p in problems}
    solved = [p for p in called if all(scores[f"block_r{k}"][p] >= 1.0 for k in (0, 1))]

    finishes = Counter()
    for r in natural.values():
        finishes[f"natural:{r['finish'].split(':')[0]}"] += 1
    for b in branches.values():
        for c in ("execute", "block"):
            for x in b[c]:
                finishes[f"{c}:{(x['finish'] or 'none').split(':')[0]}"] += 1
    analysis = {
        "accuracy": {
            "official_natural": official["natural"]["overall"],
            "official_mean": {c: float(np.mean([official[f"{c}_r{k}"]["overall"] for k in range(K)])) for c in ("execute", "block")},
            "per_problem_mean": {c: float(np.mean([mean_score(p, c) for p in problems])) for c in ("execute", "block")},
        },
        "difference_all": interval([diff[p] for p in problems]),
        "difference_called": interval([diff[p] for p in called]),
        "by_need": {"solved_without_tool": group(solved, (2, 3)),
                    "may_fail_without_tool": group([p for p in called if p not in solved], (2, 3))},
        "by_faithfulness": {str(f): group([p for p in called if faithful.get(p) == f]) for f in (1, 0)}
                           | {"none": len([p for p in called if faithful.get(p) is None])},
        "counts": {"problems": len(problems), "called": len(called), "called_share": len(called) / len(problems),
                   "call_not_branched": dict(Counter(r["call_note"] for r in natural.values() if r["call_note"])),
                   "block_without_answer": sum(not x["prediction"] for b in branches.values() for x in b["block"]),
                   "block_with_call_text": sum("<function=" in x["prediction"] for b in branches.values() for x in b["block"]),
                   "finish": dict(sorted(finishes.items()))},
    }
    target.write_text(json.dumps(analysis, indent=1) + "\n")
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
