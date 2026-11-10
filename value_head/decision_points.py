#!/usr/bin/env python3
"""Stage-2 step 2: free rollouts of the policy's first turn, kept where it opens ``<evaluating>``.

    /root/miniconda3/envs/research/bin/python value_head/decision_points.py --pool POOL_DIR --out RUN_DIR \\
        --base-url http://HOST:PORT/v1 --model cavir-policy [--datasets ...] [--resume]

For each pool question the policy writes its first turn until it opens ``<evaluating>`` (a decision point),
answers, or calls a tool without one. Writes ``RUN_DIR/points.jsonl`` (one state per decision point, prefix
up to just before ``<evaluating>``) and ``RUN_DIR/first_turns.jsonl`` (every question with its
outcome, so the rate of answering without a decision point is reported per dataset). Appends; refuses an
existing RUN_DIR unless ``--resume``, which skips questions already in ``first_turns.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy import Policy, grade  # noqa: E402


async def first_turn(policy: Policy, row: dict, seed: int) -> tuple[dict, dict | None]:
    state = {"system": row["system"], "tools": row["tools"], "turns": [row["question"]],
             "images": [row["image_path"]], "partial": ""}
    out = (await policy.generate(state, stop=["<evaluating>", "</answer>", "</tool_call>"], seed=seed))[0]
    text = out["text"]
    if text.endswith("<evaluating>") and "</think>" in text:
        kind = "decision_point"
    elif out["finish_reason"] == "length":
        kind = "length"
    elif "</answer>" in text:
        kind = "answer"
    elif "</tool_call>" in text:
        kind = "tool_without_evaluating"
    else:
        kind = "other"
    prediction, correct = grade(text, row["answer"]) if kind == "answer" else (None, None)
    turn = {"id": row["id"], "dataset": row["dataset"], "split": row["split"], "kind": kind, "seed": seed,
            "tokens": out["tokens"], "prediction": prediction, "correct": correct, "text": text}
    if kind != "decision_point":
        return turn, None
    point = {"dp_id": f"{row['id']}:0", "id": row["id"], "dataset": row["dataset"], "split": row["split"], "depth": 0,
             "system": row["system"], "tools": row["tools"], "turns": state["turns"], "images": state["images"],
             "partial": text[: -len("<evaluating>")], "answer": row["answer"], "source": "free_rollout"}
    return turn, point


async def run(args: argparse.Namespace) -> None:
    done = set()
    if args.resume and (args.out / "first_turns.jsonl").exists():
        done = {json.loads(line)["id"] for line in open(args.out / "first_turns.jsonl")}
    rows = []
    for dataset in args.datasets:
        path = args.pool / f"{dataset}.jsonl"
        rows += [r for r in map(json.loads, open(path)) if r["id"] not in done]
    policy = Policy(args.base_url, args.model, concurrency=args.concurrency)
    lock = asyncio.Lock()
    turns_file = open(args.out / "first_turns.jsonl", "a")
    points_file = open(args.out / "points.jsonl", "a")

    async def one(index: int, row: dict):
        turn, point = await first_turn(policy, row, seed=args.seed * 1_000_003 + index)
        async with lock:
            if point:
                points_file.write(json.dumps(point, ensure_ascii=False) + "\n")
                points_file.flush()
            turns_file.write(json.dumps(turn, ensure_ascii=False) + "\n")
            turns_file.flush()
        return turn["kind"]

    kinds = await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    await policy.close()
    print({kind: kinds.count(kind) for kind in set(kinds)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract decision points from free rollouts.")
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise SystemExit(f"{args.out} exists (use --resume to continue it)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.datasets is None:
        args.datasets = sorted(p.stem for p in args.pool.glob("*.jsonl"))
    meta = args.out / "decision_points_args.json"
    if not meta.exists():
        meta.write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
