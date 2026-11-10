#!/usr/bin/env python3
"""Policy's own choice at stored decision points: let it write the ``<evaluating>`` block K times.

Each point's prefix is exactly the policy's text up to where it opened ``<evaluating>`` (decision_points.py and
branch_rollout sub-points cut it there), so the request continues ``partial + "<evaluating>"`` and stops at
``</evaluating>``: no action is forced and no tool runs. The K samples come from one request (n=K, seeded by dp_id).
Each sample's block is parsed into answer_now p, every tool's Δ and the selection.

    /root/miniconda3/envs/research/bin/python value_head/natural_selection.py --points P1.jsonl [--points P2.jsonl] \\
        --out OUT.jsonl --k 16 --base-url http://HOST:PORT/v1 --model MODEL

OUT has one line per point: dp_id, id, dataset, depth, and ``samples`` [{selection, p_answer_now, deltas, finish,
tokens, text}]; ``selection`` is None when the block did not close or had no selection line. Points already in OUT
are skipped, so a rerun resumes; a server that is gone or failing stops the run (ServerUnavailable) with OUT intact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_construction"))

from branch_rollout import seeded  # noqa: E402
from policy import Policy, ServerUnavailable  # noqa: E402

BLOCK_MAX_TOKENS = 512   # a block is about 10 short lines
P_RE = re.compile(r"^answer_now:\s*p=([0-9.]+)\s*$", re.MULTILINE)
DELTA_RE = re.compile(r"^([A-Za-z_]+):\s*Δ=([+-]?[0-9.]+)\s*$", re.MULTILINE)
SELECTION_RE = re.compile(r"^selection:\s*([A-Za-z_]+)\s*$", re.MULTILINE)


def parse_block(text: str) -> dict:
    """answer_now p, tool Δs and selection from one generated block (None where a field is missing)."""
    block = text.split("</evaluating>")[0]
    p = P_RE.search(block)
    selection = SELECTION_RE.findall(block)
    return {"selection": selection[-1] if selection and "</evaluating>" in text else None,
            "p_answer_now": float(p.group(1)) if p else None,
            "deltas": {tool: float(value) for tool, value in DELTA_RE.findall(block)}}


async def run(args) -> None:
    done = set()
    if args.out.exists():
        done = {json.loads(line)["dp_id"] for line in open(args.out)}
    points = [p for path in args.points for p in map(json.loads, open(path)) if p["dp_id"] not in done]
    if args.limit:
        points = points[: args.limit]
    policy = Policy(args.base_url, args.model, concurrency=args.concurrency)
    handle, lock = open(args.out, "a"), asyncio.Lock()
    print(f"{len(points)} points to sample ({len(done)} done), k={args.k}", flush=True)

    async def one(point):
        state = {key: point[key] for key in ("system", "tools", "turns", "images")}
        state["partial"] = point["partial"] + "<evaluating>"
        outs = await policy.generate(state, n=args.k, stop=["</evaluating>"], max_tokens=BLOCK_MAX_TOKENS,
                                     seed=seeded(str(args.seed), "natural", point["dp_id"]))
        samples = [parse_block(o["text"]) | {"finish": o["finish_reason"], "tokens": o["tokens"], "text": o["text"]} for o in outs]
        async with lock:
            handle.write(json.dumps({k: point[k] for k in ("dp_id", "id", "dataset", "depth")} | {"samples": samples},
                                    ensure_ascii=False) + "\n")
            handle.flush()
        return Counter(s["selection"] for s in samples)

    try:
        counts = await asyncio.gather(*(one(p) for p in points))
    except ServerUnavailable as exc:
        raise SystemExit(f"server unreachable or failing ({exc}); {args.out} keeps the finished points, rerun to resume")
    finally:
        await policy.close()
    total = sum(counts, Counter())
    print(json.dumps({"points": len(points), "samples": sum(total.values()), "selections": dict(total.most_common())}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the policy's own <evaluating> block at stored decision points.")
    parser.add_argument("--points", action="append", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=256, help="sequences running on the server")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N new points only (smoke runs)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
