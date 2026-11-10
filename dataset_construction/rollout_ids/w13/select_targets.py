#!/usr/bin/env python3
"""The problem ids to generate, 4 rollouts each.

Existing datasets: every source record not in exclude_<ds>.txt (make_exclusions.py) and not a repeat of an earlier
record's (image, question) pair; then 5,000 drawn at random (seed 0), 10,000 for gqa_1 and chartqa_1, all for plotqa
and geometry3k. New datasets (Nemotron-Image-Training-v3.5): the records shuffled with seed 0 after the same repeat
rule; the first 135 are held out as evaluation problems, the next 5,000 are generated. Images are not opened.

    /root/miniconda3/envs/research/bin/python dataset_construction/rollout_ids/w13/select_targets.py

Writes <ds>.txt (ids to generate, one per line), eval_heldout_<ds>.txt for the new datasets and targets_summary.json,
next to this script.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import dataset  # noqa: E402
from host import DATASET_PATHS  # noqa: E402
from make_exclusions import image_key, question_key  # noqa: E402

V35 = Path("/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3.5/snapshots/"
           "70b27e021e64bd97641597dc1f127a208b58be0e")
EXISTING = {"docvqa": 5000, "infographics_vqa": 5000, "geomverse_cot": 5000, "gqa_1": 10000, "chartqa_1": 10000,
            "plotqa": None, "geometry3k": None}
NEW = ("hopchain_fintabnet", "chartnet_qa_01_difficulty", "coco_count_spatial_qa")
HELD_OUT, NEW_SIZE = 135, 5000


def unique(rows):
    """(id, pair) rows in order, keeping the first record of each (image, question) pair."""
    seen, out = set(), []
    for i, pair in rows:
        if pair not in seen:
            seen.add(pair)
            out.append(i)
    return out


def write(name, ids):
    (HERE / name).write_text("".join(f"{i}\n" for i in ids))


def main() -> None:
    dataset._preprocess_image = lambda path: ""   # parse only
    summary = {}
    for name, n in EXISTING.items():
        data = dataset.HFDataset(name, DATASET_PATHS[name])
        excluded = set((HERE / f"exclude_{name}.txt").read_text().split())
        rows = []
        for k in range(len(data)):
            item = data[k]
            if item["id"] not in excluded:
                rows.append((item["id"], (image_key(name, item["image_path"]), question_key(item["question"]))))
        pool = unique(rows)
        ids = sorted(random.Random(0).sample(pool, n) if n and n < len(pool) else pool)
        write(f"{name}.txt", ids)
        summary[name] = {"records": len(data), "excluded": len(excluded), "repeat_pairs": len(rows) - len(pool),
                         "pool": len(pool), "generate": len(ids)}
        print(name, summary[name], flush=True)
    for name in NEW:
        rows = []
        for r in map(json.loads, open(V35 / name / f"{name}.jsonl")):
            content = r["messages"][0]["content"]
            image = next(c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image")
            text = " ".join(c for c in content if isinstance(c, str))
            rows.append((r["id"], (image, question_key(text))))
        pool = unique(rows)
        random.Random(0).shuffle(pool)
        held, ids = pool[:HELD_OUT], pool[HELD_OUT:HELD_OUT + NEW_SIZE]
        write(f"eval_heldout_{name}.txt", sorted(held))
        write(f"{name}.txt", sorted(ids))
        images = {dict(rows)[i][0] for i in held}
        summary[name] = {"records": len(rows), "repeat_pairs": len(rows) - len(pool), "pool": len(pool),
                         "held_out": len(held), "generate": len(ids),
                         "generate_same_image_as_held_out": sum(dict(rows)[i][0] in images for i in ids)}
        print(name, summary[name], flush=True)
    (HERE / "targets_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
