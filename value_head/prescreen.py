#!/usr/bin/env python3
"""Cheap pre-screen of decision points: answer_now forced N times, to stratify which points get branched.

    /root/miniconda3/envs/research/bin/python value_head/prescreen.py --points POINTS.jsonl --out PRESCREEN.jsonl \\
        --n 4 --base-url http://HOST:PORT/v1 --model MODEL

Many decision points have answer_now right in all or none of their continuations, and most Δ there are
exactly 0, so ``select_points.py`` keeps every point whose pre-screen is
mixed and fills up with all-right and all-wrong points. The block is rendered as in ``branch_rollout.py``
(same seed rule, a different seed so the pre-screen samples are not the branch's). Writes one line per point
(dp_id, dataset, s correct out of n); appends and skips points already there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branch_rollout import STOP, TAU, dataset_tools, seeded  # noqa: E402
from build_cavir_coldstart import render_evaluating  # noqa: E402
from policy import Policy, grade  # noqa: E402


async def run(args) -> None:
    done = set()
    if args.out.exists():
        done = {json.loads(line)["dp_id"] for line in open(args.out)}
    points = [p for p in map(json.loads, open(args.points)) if p["dp_id"] not in done]
    policy = Policy(args.base_url, args.model, concurrency=args.concurrency)
    handle = open(args.out, "a")
    lock = asyncio.Lock()
    print(f"{len(points)} points to pre-screen ({len(done)} done)", flush=True)

    async def one(point):
        block = render_evaluating("answer_now", dataset_tools(point["tools"]), TAU,
                                  random.Random(seeded("prescreen", point["dp_id"])))
        state = {key: point[key] for key in ("system", "tools", "turns", "images")}
        state["partial"] = f"{point['partial'].rstrip()}\n\n{block}\n\n"
        outs = await policy.generate(state, n=args.n, stop=STOP, seed=seeded("prescreen", point["dp_id"], "n"))
        s = sum(grade(o["text"], point["answer"])[1] for o in outs)
        async with lock:
            handle.write(json.dumps({"dp_id": point["dp_id"], "dataset": point["dataset"], "depth": point["depth"],
                                     "s": s, "n": len(outs)}) + "\n")
            handle.flush()

    await asyncio.gather(*(one(p) for p in points))
    await policy.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="answer_now pre-screen of decision points.")
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=256, help="sequences running on the server")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
