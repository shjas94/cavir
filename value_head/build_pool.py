#!/usr/bin/env python3
"""Stage-2 problem pool: questions the noarith policy never trained on, with parsed ground truth.

    /root/miniconda3/envs/research/bin/python value_head/build_pool.py --per-dataset 15 --seed 0 --out OUT_DIR

Per dataset, samples questions that were never rolled out (``split: unrolled``). plotqa and geometry3k were
rolled out in full, so they fall back to rolled-out questions absent from the noarith training data
(``split: rolled_not_trained``; the teacher failed them or the quality filter dropped them, so report them
apart). The ``cavir_pilot_v2`` questions are always excluded (held-out test set), as are questions whose
ground truth does not parse. Writes ``<ds>.jsonl`` with the question rendered as in training, the
system prompt and tools of the noarith training records, the image path and its sha1 (for later overlap
checks against evaluation benchmarks), and the normalized answer. Refuses to write into an existing dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset_construction"
sys.path.insert(0, str(DATA_DIR))

from dataset import HFDataset  # noqa: E402
from filter_out import extract_ground_truth  # noqa: E402
from host import DATASET_PATHS  # noqa: E402
from reformat_dataset import escape_image_placeholder  # noqa: E402

TRAIN_FILE = REPO / "sft/LlamaFactory/data/cavir_coldstart_v2_noarith/cavir_coldstart_v2_noarith.jsonl"
PILOT_FILE = REPO / "sft/LlamaFactory/data/cavir_pilot_v2/cavir_pilot_v2.jsonl"
ROLLOUTS = [DATA_DIR / "legacy/outputs", DATA_DIR / "outputs_noarith"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the stage-2 problem pool.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-pool", nargs="*", type=Path, default=[], help="earlier pool dirs whose questions are skipped")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    headers, trained = {}, set()
    for line in open(TRAIN_FILE):
        record = json.loads(line)
        trained.add(record["id"])
        header = (record["system"], record["tools"])
        if headers.setdefault(record["meta"]["dataset"], header) != header:
            raise SystemExit(f"{record['meta']['dataset']}: training records disagree on system prompt or tools")
    pilot = {json.loads(line)["id"] for line in open(PILOT_FILE)}
    for pool_dir in args.exclude_pool:  # e.g. the stage-2 pilot: its questions stay out of the full run
        pilot |= {json.loads(line)["id"] for path in pool_dir.glob("*.jsonl") for line in open(path)}

    args.out.mkdir(parents=True)
    summary = {}
    for dataset in args.datasets:
        rolled = set()
        for root in ROLLOUTS:
            path = root / dataset / "results.jsonl"
            if path.exists():
                rolled |= {json.loads(line)["id"] for line in open(path)}
        source = HFDataset(dataset, DATASET_PATHS[dataset])
        ids = [str(i) for i in source.item_ids()]
        unrolled = [i for i, x in enumerate(ids) if x not in rolled and x not in pilot]
        fallback = [i for i, x in enumerate(ids) if x in rolled and x not in trained and x not in pilot]
        rng = random.Random(f"{args.seed}:{dataset}")
        rng.shuffle(unrolled)
        rng.shuffle(fallback)
        candidates = unrolled if unrolled else fallback
        split = "unrolled" if unrolled else "rolled_not_trained"

        rows, skipped = [], 0
        for index in candidates:
            if len(rows) == args.per_dataset:
                break
            item = source[index]
            answer, method = extract_ground_truth(item, dataset)
            if answer is None:
                skipped += 1
                continue
            system, tools = headers[dataset]
            rows.append({
                "id": str(item["id"]), "dataset": dataset, "split": split, "system": system, "tools": tools,
                "question": f"<image>\n{escape_image_placeholder(item['question'].strip())}",
                "image_path": item["image_path"],
                "image_sha1": hashlib.sha1(Path(item["image_path"]).read_bytes()).hexdigest(),
                "gt": item["gt"], "answer": answer, "answer_method": method,
            })
        with open(args.out / f"{dataset}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[dataset] = {"source": len(ids), "unrolled": len(unrolled), "rolled_not_trained": len(fallback),
                            "split": split, "picked": len(rows), "skipped_no_answer": skipped}
        print(dataset, summary[dataset])
    (args.out / "pool_summary.json").write_text(json.dumps(
        {"seed": args.seed, "per_dataset": args.per_dataset, "train_file": str(TRAIN_FILE), "datasets": summary},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
