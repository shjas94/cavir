#!/usr/bin/env python3
"""Write a bf16 copy of a full fine-tuned checkpoint, keeping the dtypes the base model ships with.

LLaMA Factory full fine-tuning with bf16 mixed precision keeps fp32 master weights, so the saved
``model.safetensors`` is all fp32 (twice the size of the base model). This copies the checkpoint to
``--dst`` with every floating tensor in bf16, except the tensors the base model stores in fp32
(for Qwen3.5: ``linear_attn.A_log`` and ``linear_attn.norm.weight``), and sets ``dtype`` in config.json.
Training is unchanged; only the saved copy is converted. Refuses an existing ``--dst``.

    /root/miniconda3/envs/train_cluster/bin/python sft/convert_to_bf16.py --src SAVE_DIR --dst OUT_DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SERVING_FILES = ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
                 "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "README.md", "train_results.json", "all_results.json")


def fp32_names(base: str) -> set[str]:
    names = set()
    for path in glob.glob(f"/root/.cache/huggingface/hub/models--{base.replace('/', '--')}/snapshots/*/*.safetensors"):
        with safe_open(path, "pt") as handle:
            names |= {k for k in handle.keys() if str(handle.get_slice(k).get_dtype()) == "F32"}
    if not names:
        raise SystemExit(f"no fp32 tensors found for base model {base}; is it in the HF cache?")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="bf16 copy of a fine-tuned checkpoint.")
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--base", default=None, help="base model repo id (default: base_model in README / config)")
    args = parser.parse_args()
    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists")
    base = args.base
    if base is None:
        readme = (args.src / "README.md").read_text()
        base = next(line.split(":", 1)[1].strip() for line in readme.splitlines() if line.startswith("base_model:"))
    keep_fp32 = fp32_names(base)

    tensors, kinds = {}, Counter()
    with safe_open(args.src / "model.safetensors", "pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(torch.float32 if key in keep_fp32 else torch.bfloat16)
            kinds[str(tensor.dtype)] += 1
            tensors[key] = tensor.contiguous()
    args.dst.mkdir(parents=True)
    save_file(tensors, args.dst / "model.safetensors", metadata=metadata or {"format": "pt"})
    for name in SERVING_FILES:
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)
    config = json.loads((args.dst / "config.json").read_text())
    config["dtype"] = "bfloat16"
    (args.dst / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"base": base, "src": str(args.src), "dst": str(args.dst), "tensors": dict(kinds)}))


if __name__ == "__main__":
    main()
