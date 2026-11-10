#!/usr/bin/env python3
"""Regrade saved branch continuations with the answer-type rules (``grading.py``) and drop excluded problems.

    python value_head/regrade_branches.py --points R1/selected_points.jsonl --points R2/selected_points.jsonl \
        --branches R1/branches/branches.jsonl OUT/full_r1/branches.jsonl [--branches IN OUT ...] \
        [--exclude problems.txt] --summary OUT/regrade_summary.json

Nothing is regenerated: each continuation keeps its stored ``prediction``. In the output rows ``correct`` is the new
verdict, ``correct_strict`` the old one and ``answer_type`` the problem's type. Rows of problems listed in ``--exclude``
(one problem id per line or a .json object keyed by id, e.g. truncated problems of definition B and ground-truth defects) are left out, so
``analyze_branches.py`` gives them no labels and the scorer scripts skip them. Existing outputs are never overwritten.
The summary counts newly correct continuations by (dataset, type) and lists every newly accepted (answer, prediction)
pair, counted over continuations that analyze_branches keeps (finish not in its EXCLUDED).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branches import EXCLUDED  # noqa: E402
from grading import answer_type, grade  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regrade branch continuations by answer type.")
    parser.add_argument("--points", action="append", type=Path, required=True, help="selected_points.jsonl with the answers")
    parser.add_argument("--branches", nargs=2, action="append", type=Path, required=True, metavar=("IN", "OUT"))
    parser.add_argument("--exclude", type=Path, help="problem ids to leave out: one per line, or a .json object keyed by id")
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    for path in [out for _, out in args.branches] + [args.summary]:
        if path.exists():
            raise SystemExit(f"{path} exists")

    answer, dataset, problem = {}, {}, {}
    for path in args.points:
        for line in open(path):
            p = json.loads(line)
            answer[p["dp_id"]], dataset[p["dp_id"]], problem[p["dp_id"]] = p["answer"], p["dataset"], p["id"]
    excluded = set()
    if args.exclude:  # a JSON object keyed by problem id ({id: [reasons]}) or one id per line
        text = args.exclude.read_text()
        excluded = set(json.loads(text)) if args.exclude.suffix == ".json" else {l.strip() for l in text.splitlines() if l.strip()}
    unknown = excluded - set(problem.values())
    if unknown:
        raise SystemExit(f"--exclude has {len(unknown)} ids that are not in --points, e.g. {sorted(unknown)[:3]}")

    files, newly, pairs = {}, Counter(), Counter()
    for src, dst in args.branches:
        kept = dropped = 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as handle:
            for line in open(src):
                row = json.loads(line)
                dp = row["dp_id"]
                if problem[dp] in excluded:
                    dropped += 1
                    continue
                kind = answer_type(dataset[dp], answer[dp])
                strict = bool(row["correct"])
                row["correct_strict"], row["answer_type"] = strict, kind
                row["correct"] = grade(row.get("prediction"), answer[dp], dataset[dp], strict)
                if row["correct"] and not strict and row["finish"] not in EXCLUDED:
                    newly[f"{dataset[dp]} {kind}"] += 1
                    pairs[(answer[dp], row["prediction"].strip())] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        files[str(src)] = {"out": str(dst), "kept": kept, "dropped_excluded": dropped}
        print(f"{src} -> {dst}: kept {kept}, dropped {dropped}", flush=True)

    types = Counter(answer_type(dataset[dp], answer[dp]) for dp in {problem[d]: d for d in problem}.values())
    summary = {"files": files, "excluded_problems": len(excluded), "problems_by_type": dict(types),
               "newly_correct": sum(newly.values()), "newly_correct_by_dataset_type": dict(sorted(newly.items())),
               "newly_accepted_pairs": len(pairs),
               "pairs": [{"answer": a, "prediction": p, "continuations": n} for (a, p), n in sorted(pairs.items())]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("excluded_problems", "problems_by_type", "newly_correct",
                                              "newly_correct_by_dataset_type", "newly_accepted_pairs")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
