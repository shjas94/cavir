#!/usr/bin/env python3
"""Freeze the selection-following prompts as the cold-start v2 pilot test set.

Run with the research env from /workspace/cavir/dataset_construction:
    /root/miniconda3/envs/research/bin/python ../sft/build_pilot_set.py --prompts PROMPTS.jsonl --out OUT.jsonl

Keeps every prompt and field of ``make_followcheck_prompts.py``'s output (so the file still works with
``selection_following.py --prompts``) and adds the normalized ground-truth ``answer`` with
``filter_out.extract_ground_truth``; grade predictions against it with ``filter_out.answers_match``.
Writes nothing if an id repeats, an image is missing, a prompt appears in the v2 training data, or the
output already exists. Images stay as Hugging Face cache paths, like the training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_construction"))

from filter_out import extract_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pilot test set from held-out prompts.")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-data", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dataset_construction/legacy/llamafactory_data/cavir_coldstart_v2/cavir_coldstart_v2.jsonl")
    args = parser.parse_args()
    report_path = args.out.with_name(args.out.stem + "_report.json")
    for path in (args.out, report_path):
        if path.exists():
            raise SystemExit(f"{path} exists")

    rows = [json.loads(line) for line in open(args.prompts)]
    ids = Counter(row["id"] for row in rows)
    trained = {json.loads(line)["id"] for line in open(args.train_data)}
    problems = {
        "repeated_ids": [i for i, n in ids.items() if n > 1],
        "missing_images": [p for row in rows for p in row["images"] if not os.path.exists(p)],
        "in_training_data": [row["id"] for row in rows if row["id"] in trained],
    }
    if any(problems.values()):
        raise SystemExit(json.dumps({k: v[:5] for k, v in problems.items() if v}))

    methods = Counter()
    for row in rows:
        row["answer"], row["answer_method"] = extract_ground_truth(row, row["dataset"])
        methods[row["answer_method"]] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "created": date.today().isoformat(), "output": str(args.out.resolve()),
        "source": str(args.prompts.resolve()), "source_md5": hashlib.md5(args.prompts.read_bytes()).hexdigest(),
        "records": len(rows), "by_dataset": dict(Counter(row["dataset"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "answer_methods": dict(methods.most_common()),
        "unparsed_answers": [row["id"] for row in rows if row["answer"] is None],
        "note": "Held-out prompts for the selection-following check of the cold-start v2 model; "
                "none of them is in the v2 training data.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("records", "by_dataset", "by_split", "answer_methods", "unparsed_answers")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
