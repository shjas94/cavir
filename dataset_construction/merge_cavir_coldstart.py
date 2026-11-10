#!/usr/bin/env python3
"""Merge per-dataset filtered cold-start v2 files into one LlamaFactory SFT file.

Reads ``<dataset_dir>/<input-name>.jsonl`` for each dataset dir (never modified),
checks that every image path exists, shuffles with a fixed seed and writes one
ShareGPT jsonl plus ``<output stem>_report.json`` with counts per dataset, variant,
Case and selection. No reweighting: every record is kept once. Refuses to overwrite.
The ``meta`` field is kept; LlamaFactory ignores columns it is not told about.
With ``--dataset-info`` the file is also registered in a LlamaFactory ``dataset_info.json``
under the output stem (an existing entry for the same file is left as is).
With ``--image-dir`` every image stored under this directory (tool outputs in ``outputs/`` and
``outputs_cavir/``) is copied there, keeping its relative path, and the records point to the copy,
so training no longer depends on this repository. Other images (the Hugging Face cache) stay as is.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from filter_cavir_coldstart import selections

HERE = Path(__file__).resolve().parent

# Same columns and tags as the v1 cold-start entries in LlamaFactory's data/dataset_info.json.
DATASET_INFO_ENTRY = {
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "tools": "tools", "images": "images", "system": "system"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
}


def registered_file_name(dataset_info: Path, output: Path) -> str:
    output, root = output.resolve(), dataset_info.resolve().parent
    return str(output.relative_to(root)) if output.is_relative_to(root) else str(output)


def register(dataset_info: Path, name: str, file_name: str) -> None:
    """Append ``name`` to dataset_info.json as text, so the rest of the file stays byte-identical."""
    text = dataset_info.read_text() if dataset_info.exists() else "{}"
    if name in json.loads(text):
        return
    entry = json.dumps({name: {"file_name": file_name, **DATASET_INFO_ENTRY}}, ensure_ascii=False, indent=2)[1:-1].strip("\n")
    body = text.rstrip()[:-1].rstrip()
    dataset_info.write_text(body + (",\n" if body != "{" else "\n") + entry + "\n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge filtered cold-start v2 files into one SFT file.")
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--input-name", default="cavir_filtered_v2", help="Per-dataset file stem to merge.")
    parser.add_argument("--output", required=True, type=Path, help="Merged .jsonl path; must not exist yet.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-info", type=Path, help="LlamaFactory dataset_info.json to register the output in.")
    parser.add_argument("--image-dir", type=Path, help="Copy images stored under dataset_construction/ here.")
    args = parser.parse_args()

    report_path = args.output.with_name(args.output.stem + "_report.json")
    for path in (args.output, report_path):
        if path.exists():
            raise SystemExit(f"{path} exists; choose another --output")
    if args.dataset_info:
        existing = json.loads(args.dataset_info.read_text()) if args.dataset_info.exists() else {}
        wanted = registered_file_name(args.dataset_info, args.output)
        if args.output.stem in existing and existing[args.output.stem].get("file_name") != wanted:
            raise SystemExit(f"{args.dataset_info} already maps '{args.output.stem}' to another file")

    records, per_dataset = [], Counter()
    for dataset_dir in args.dataset_dirs:
        source = dataset_dir.resolve() / f"{args.input_name}.jsonl"
        for line in open(source):
            record = json.loads(line)
            record["meta"]["dataset"] = dataset_dir.resolve().name
            records.append(record)
            per_dataset[dataset_dir.resolve().name] += 1

    missing = [p for r in records for p in r["images"] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} image paths do not exist, e.g. {missing[0]}")

    copied = 0
    if args.image_dir:
        image_dir = args.image_dir.resolve()
        for record in records:
            for i, path in enumerate(record["images"]):
                if Path(path).is_relative_to(HERE):
                    target = image_dir / Path(path).relative_to(HERE)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                        copied += 1
                    record["images"][i] = str(target)

    random.Random(args.seed).shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    picked = Counter(s for r in records for s in selections(r))
    report = {
        "inputs": [str(d.resolve() / f"{args.input_name}.jsonl") for d in args.dataset_dirs],
        "output": str(args.output.resolve()), "seed": args.seed, "records": len(records),
        "images": sum(len(r["images"]) for r in records), "images_copied": copied,
        "image_dir": str(args.image_dir.resolve()) if args.image_dir else None,
        "by_dataset": dict(per_dataset),
        "by_variant": dict(Counter(r["meta"]["variant"] for r in records).most_common()),
        "by_case": {str(k): v for k, v in sorted(Counter(r["meta"]["case"] for r in records).items())},
        "selections": {"tool": sum(n for s, n in picked.items() if s not in ("answer_now", "think_more")),
                       "answer_now": picked["answer_now"], "think_more": picked["think_more"]},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.dataset_info:
        register(args.dataset_info, args.output.stem, registered_file_name(args.dataset_info, args.output))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
