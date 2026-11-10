#!/usr/bin/env python3
"""Pool the test predictions of a cross-validation's folds into one evaluation (e.g. 5-fold).

    python value_head/evaluate_folds.py --evals CKPT_F0/eval_test ... CKPT_F4/eval_test \
        --baseline-evals TABLE_F0 ... TABLE_F4 --branches B1 --branches B2 ... --reference best --bootstrap 2000 \
        --subset-split value_head/outputs/scorer_<name>/split.json --out OUT

Each ``--evals`` directory is an ``evaluate_scorer.py`` output on one fold's test split (``predictions.jsonl``), and
``--baseline-evals`` the same folds predicted by the mean table (``evaluate_scorer.py --table-baseline``). Every
problem is a test problem in exactly one fold, so the pooled predictions cover all problems once; metrics and the
paired scorer − table bootstrap CI are computed over all of them, exactly as ``evaluate_scorer.report``.
``--subset-split`` adds the same report on that split's test problems alone (e.g. an earlier fixed test
split, reported for reference). Writes OUT/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branches import BASES, EXCLUDED, REFERENCES, deltas, reliability  # noqa: E402
from evaluate_scorer import load_branches, main_block, report  # noqa: E402


def pooled(dirs: list[Path]) -> list[dict]:
    rows = [json.loads(line) for d in dirs for line in open(d / "predictions.jsonl")]
    ids = [p["dp_id"] for p in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"a decision point appears in two folds: {dirs}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled evaluation over cross-validation folds.")
    parser.add_argument("--evals", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-evals", nargs="+", type=Path)
    parser.add_argument("--branches", action="append", type=Path, required=True)
    parser.add_argument("--reference", default="best", choices=REFERENCES)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--subset-split", type=Path, help="split.json whose test problems get a separate report")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    predictions = pooled(args.evals)
    baseline = pooled(args.baseline_evals) if args.baseline_evals else None
    if baseline and {p["dp_id"]: p["labels"] for p in baseline} != {p["dp_id"]: p["labels"] for p in predictions}:
        raise SystemExit("the baseline folds cover other decision points or labels")
    groups = load_branches(args.branches, {p["dp_id"] for p in predictions})
    for p in predictions:  # labels must be what the branches give (evaluate_scorer checked each fold already)
        for action, (s, n) in p["labels"].items():
            valid = [r for r in groups[(p["dp_id"], action)] if r["finish"] not in EXCLUDED]
            if (sum(bool(r["correct"]) for r in valid), len(valid)) != (s, n):
                raise SystemExit(f"labels of {p['dp_id']} {action} differ from --branches")
    empirical = defaultdict(dict)
    for (dp, tool), delta in deltas(groups, reference=args.reference).items():
        empirical[dp][tool] = delta
    needed = {args.reference} if args.reference in BASES else set(BASES)
    for p in predictions + (baseline or []):
        if not needed <= set(p["q_mean"]):
            raise SystemExit(f"--reference {args.reference} needs q̂ of {sorted(needed)}; {p['dp_id']} has {sorted(p['q_mean'])}")
        p["reference"], p["delta"] = args.reference, empirical[p["dp_id"]]

    dataset_of = {p["dp_id"]: p["dataset"] for p in predictions}
    k_total = max(r["k"] for rows in groups.values() for r in rows) + 1
    ceiling = reliability(groups, k_total, dataset_of, args.reference).get(f"K={k_total}", {})
    metrics = {"evals": [str(d) for d in args.evals], "baseline_evals": [str(d) for d in args.baseline_evals or []],
               "points": len(predictions), "problems": len({p["id"] for p in predictions}), "k": k_total,
               "reference": args.reference, "kind": predictions[0].get("kind", "scorer")}
    metrics |= report(predictions, ceiling, args.bootstrap, baseline)
    if baseline:
        metrics["baseline"] = report(baseline, ceiling, args.bootstrap)
    metrics = {"main": main_block(metrics, metrics.get("baseline"), baseline[0].get("kind") if baseline else None)} | metrics
    if args.subset_split:
        split = json.loads(args.subset_split.read_text())
        keep = lambda rows: [p for p in rows if split.get(p["id"]) == "test"]
        sub, sub_base = keep(predictions), keep(baseline) if baseline else None
        metrics["subset"] = {"split": str(args.subset_split), "points": len(sub), "problems": len({p["id"] for p in sub})}
        metrics["subset"] |= report(sub, ceiling, args.bootstrap, sub_base)
        if sub_base:
            metrics["subset"]["baseline"] = report(sub_base, ceiling, args.bootstrap)
        metrics["subset"]["main"] = main_block(metrics["subset"], metrics["subset"].get("baseline"),
                                               sub_base[0].get("kind") if sub_base else None)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("points", "problems", "main")}, indent=1))


if __name__ == "__main__":
    main()
