#!/usr/bin/env python3
"""Draw held-out questions for the selection-following check.

Run with the research env from /workspace/cavir/dataset_construction:
    /root/miniconda3/envs/research/bin/python ../sft/make_followcheck_prompts.py --out PATH.jsonl

For every dataset, samples questions whose id never appears in ``outputs/<ds>/results.jsonl``
(``split: unrolled``: no rollout, hence no cold-start record, used them). plotqa and geometry3k were
rolled out in full, so they fall back to rolled-out questions absent from the v2 training data
(``split: rolled_not_trained``); the v2 policy was trained from the base model and never saw them.
Writes one-turn ShareGPT prompts: the question exactly as the rollout would have phrased it (``dataset.py``),
and the system prompt and tool list copied from that dataset's cold-start v2 training records.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset_construction"
sys.path.insert(0, str(DATA_DIR))

from dataset import HFDataset  # noqa: E402
from host import DATASET_PATHS  # noqa: E402
from reformat_dataset import escape_image_placeholder  # noqa: E402

def item_ids(source: HFDataset) -> list[str]:
    """Ids in HFDataset index order, by the rule in ``HFDataset.__getitem__``, without decoding images."""
    with open(source.jsonl_path, "rb") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    return [item.get("id", f"{source.dataset_name}_{index}") for index, item in enumerate(items)]


DATASETS = ["docvqa", "infographics_vqa", "chartqa_1", "chartqa_2", "plotqa", "gqa_1", "gqa_2", "geometry3k", "geomverse_cot"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample held-out prompts for the selection-following check.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-data", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dataset_construction/legacy/llamafactory_data/cavir_coldstart_v2/cavir_coldstart_v2.jsonl")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    # system prompt and tools per dataset, as the policy saw them in training
    headers: dict[str, tuple[str, str]] = {}
    trained: set[str] = set()
    for line in open(args.train_data):
        record = json.loads(line)
        header = (record["system"], record["tools"])
        dataset = record["meta"]["dataset"]
        trained.add(record["id"])
        if headers.setdefault(dataset, header) != header:
            raise SystemExit(f"{dataset}: training records disagree on system prompt or tools")

    rng = random.Random(args.seed)
    rows = []
    for dataset in DATASETS:
        used = {json.loads(line)["id"] for line in open(DATA_DIR / "outputs" / dataset / "results.jsonl")}
        source = HFDataset(dataset, DATASET_PATHS[dataset])
        indices = list(range(len(source)))
        rng.shuffle(indices)
        picked = 0
        ids = item_ids(source)
        unrolled = [i for i in indices if ids[i] not in used]
        fallback = [i for i in indices if ids[i] in used and ids[i] not in trained]
        for index in unrolled + fallback:
            item = source[index]
            system, tools = headers[dataset]
            rows.append({
                "id": item["id"], "dataset": dataset, "split": "unrolled" if item["id"] not in used else "rolled_not_trained",
                "system": system, "tools": tools,
                "conversations": [{"from": "human", "value": f"<image>\n{escape_image_placeholder(item['question'].strip())}"}],
                "images": [item["image_path"]], "gt": item["gt"],
            })
            picked += 1
            if picked == args.per_dataset:
                break
        print(f"{dataset}: {len(source)} samples, {len(unrolled)} never rolled out, {len(fallback)} rolled out but not "
              f"trained; picked {picked}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
