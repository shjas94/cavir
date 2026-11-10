#!/usr/bin/env python3
"""Ids each source dataset must not train on, matched by (image, question) pair.

Excluded problems: the evaluation problems (the stage-2 pools of the 2B and 4B runs) and the 450
selection-following problems (cavir_pilot_v2). A source record is excluded when its image (the path inside the
dataset's image folder, since geometry3k names every image img_diagram.png) and question text equal those of an
excluded problem of the same family: gqa_1 and gqa_2 share their records, and so do chartqa_1 and
chartqa_2, so a problem held out from
gqa_2 is excluded from gqa_1 too. Records that share only the image with an excluded problem are counted, not
excluded.

    /root/miniconda3/envs/research/bin/python dataset_construction/rollout_ids/w13/make_exclusions.py

Writes exclude_<dataset>.txt (one source id per line) for the seven generated datasets and summary.json, next to
this script. Images are not opened: dataset.py's parsing branch gives the question and image path.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "dataset_construction"))

import dataset  # noqa: E402
from host import DATASET_PATHS  # noqa: E402

POOLS = [REPO / "value_head/outputs/full_r1_20261015/pool", REPO / "value_head/outputs/full4b_20261024/pool"]
SELECTION_FOLLOWING = REPO / "sft/LlamaFactory/data/cavir_pilot_v2/cavir_pilot_v2.jsonl"
GENERATED = ("docvqa", "infographics_vqa", "chartqa_1", "plotqa", "gqa_1", "geometry3k", "geomverse_cot")
FAMILY = {"gqa_1": "gqa", "gqa_2": "gqa", "chartqa_1": "chartqa", "chartqa_2": "chartqa"}


def image_key(dataset_name: str, path: str) -> str:
    return str(Path(path).relative_to(dataset.IMAGES_DIR[dataset_name]))


def question_key(text: str) -> str:
    return " ".join(text.removeprefix("<image>").split())


def held_out() -> tuple[set, dict]:
    pairs, sources = set(), Counter()
    for pool in POOLS:
        for path in sorted(pool.glob("*.jsonl")):
            for row in map(json.loads, open(path)):
                pairs.add((FAMILY.get(row["dataset"], row["dataset"]), image_key(row["dataset"], row["image_path"]),
                           question_key(row["question"])))
                sources[f"pool:{pool.parent.name}"] += 1
    for row in map(json.loads, open(SELECTION_FOLLOWING)):
        pairs.add((FAMILY.get(row["dataset"], row["dataset"]), image_key(row["dataset"], row["images"][0]),
                   question_key(row["conversations"][0]["value"])))
        sources["selection_following"] += 1
    return pairs, dict(sources)


def main() -> None:
    dataset._preprocess_image = lambda path: ""   # parse only; the image is not needed here
    pairs, sources = held_out()
    images = {(family, image) for family, image, _ in pairs}
    summary = {"held_out_rows": sources, "held_out_pairs": len(pairs), "datasets": {}}
    for name in GENERATED:
        data = dataset.HFDataset(name, DATASET_PATHS[name])
        family = FAMILY.get(name, name)
        excluded, image_only = [], 0
        for i in range(len(data)):
            item = data[i]
            key = (family, image_key(name, item["image_path"]), question_key(item["question"]))
            if key in pairs:
                excluded.append(item["id"])
            elif key[:2] in images:
                image_only += 1
        (HERE / f"exclude_{name}.txt").write_text("".join(f"{i}\n" for i in excluded))
        summary["datasets"][name] = {"records": len(data), "excluded": len(excluded), "same_image_other_question": image_only}
        print(name, summary["datasets"][name], flush=True)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
