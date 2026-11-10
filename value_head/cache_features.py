#!/usr/bin/env python3
"""Cache the frozen policy's last-token hidden state of every labeled decision point (the probe).

Run with the train_cluster env from sft/LlamaFactory, one GPU:

    cd /workspace/cavir/sft/LlamaFactory
    /root/miniconda3/envs/train_cluster/bin/python ../../value_head/cache_features.py \
        --data R1/selected_points.jsonl LABELS_R1 --data R2/selected_points.jsonl LABELS_R2 --out FEATURES.pt

The policy is never trained in the probe, so these states are the same for every fold: computed once (one decision
point per forward pass, no gradients) and read by ``train_scorer.py --frozen`` and ``evaluate_scorer.py``. The file
holds {"model": path, "width": hidden size, "features": {dp_id: float32 tensor [width]}}; 2,048 floats per point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer import POLICY, Encoder, dist_setup, gather_to_all, last_hidden, load_backbone, load_examples, memory_guard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache policy hidden states for the probe.")
    parser.add_argument("--data", nargs=2, action="append", type=Path, required=True, metavar=("POINTS", "LABELS"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=str(POLICY))
    parser.add_argument("--limit", type=int, default=0, help="smoke: only this many points (0 = all)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    rank, world = dist_setup()  # two nodes (run_ddp.sh): each rank caches every world-th point, rank 0 saves

    examples = load_examples([tuple(pair) for pair in args.data])
    if args.limit:
        examples = examples[:args.limit]
    encoder = Encoder(args.model)
    backbone = load_backbone(args.model).eval()
    features, start = {}, time.time()
    mine = examples[rank::world]
    with torch.inference_mode():
        for i, e in enumerate(mine):
            features[e["dp_id"]] = last_hidden(backbone, {k: v.cuda() for k, v in encoder(e).items()}).cpu()
            if (i + 1) % 50 == 0:
                memory_guard()
                print(f"[rank {rank}] cached {i + 1}/{len(mine)} ({time.time() - start:.0f}s)", flush=True)
    features = dict(gather_to_all(list(features.items()), world))
    if rank != 0:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "width": backbone.config.text_config.hidden_size, "features": features}, args.out)
    print(f"cached {len(features)} points in {time.time() - start:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
