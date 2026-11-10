#!/usr/bin/env python3
"""Choose which decision points get branched, and turn later decision points into points.

select     keep every point whose answer_now pre-screen is mixed (0 < s < n), then fill each dataset up to
           ``--per-dataset`` with all-right and all-wrong points in equal parts (both still carry nonzero Δ:
           a tool can fix a wrong answer or break a right one). Fewer points than the target means all are kept.
subpoints  from a finished branch run, take the later decision points the policy opened after a tool result
           (``sub_points`` of tool branches) and keep up to ``--per-problem`` per question, each from a different
           (action, continuation), as point records for a second round. Points after the last allowed tool call
           (depth >= MAX_TOOL_CALLS) are left out before picking: no tool can run there, so a forced tool would
           only record the limit as a failure.

    python value_head/select_points.py select --points P.jsonl --prescreen S.jsonl --per-dataset 45 --out OUT.jsonl
    python value_head/select_points.py subpoints --points P.jsonl --branches B.jsonl --per-problem 2 --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MAX_TOOL_CALLS = 3  # policy.MAX_TOOL_CALLS (= host.MAX_TOOL_CALLS); not imported to keep this script light


def select(args) -> None:
    screen = {row["dp_id"]: row for row in map(json.loads, open(args.prescreen))}
    by_ds = defaultdict(lambda: {"mixed": [], "right": [], "wrong": []})
    for point in map(json.loads, open(args.points)):
        row = screen.get(point["dp_id"])
        if row is None:
            continue
        kind = "right" if row["s"] == row["n"] else "wrong" if row["s"] == 0 else "mixed"
        point["prescreen"] = {"s": row["s"], "n": row["n"]}
        by_ds[point["dataset"]][kind].append(point)
    rng = random.Random(args.seed)
    caps = {k: int(v) for k, v in (item.split("=") for item in args.cap)}
    chosen, summary = [], {}
    for dataset, pools in sorted(by_ds.items()):
        target = caps.get(dataset, args.per_dataset)
        for pool in pools.values():
            rng.shuffle(pool)
        picked = pools["mixed"][:target]
        room = target - len(picked)
        right, wrong = pools["right"], pools["wrong"]
        half = room // 2
        take_wrong = min(len(wrong), max(half, room - len(right)))
        take_right = min(len(right), room - take_wrong)
        picked += wrong[:take_wrong] + right[:take_right]
        chosen += picked
        summary[dataset] = {"available": {k: len(v) for k, v in pools.items()}, "picked": len(picked),
                            "mixed": min(len(pools["mixed"]), target), "wrong": take_wrong, "right": take_right}
    write(args.out, chosen)
    print(json.dumps(summary, indent=1))


def subpoints(args) -> None:
    parents = {p["dp_id"]: p for p in map(json.loads, open(args.points))}
    by_problem, dropped = defaultdict(list), Counter()
    for row in map(json.loads, open(args.branches)):
        if row["action"] == "answer_now" or not row["followed"]:
            continue
        for j, sub in enumerate(row["sub_points"]):
            if sub["depth"] >= MAX_TOOL_CALLS:
                dropped[sub["depth"]] += 1
                continue
            by_problem[row["id"]].append((row, j, sub))
    rng = random.Random(args.seed)
    chosen = []
    for problem, candidates in sorted(by_problem.items()):
        rng.shuffle(candidates)
        used, picked = set(), 0
        for row, j, sub in candidates:
            key = (row["action"], row["k"])
            if key in used or picked >= args.per_problem:
                continue
            used.add(key)
            picked += 1
            parent = parents[row["dp_id"]]
            chosen.append({"dp_id": f"{row['dp_id']}|{row['action']}|{row['k']}|s{j}", "id": row["id"],
                           "dataset": row["dataset"], "split": row["split"], "depth": sub["depth"],
                           "system": parent["system"], "tools": parent["tools"], "turns": sub["turns"],
                           "images": sub["images"], "partial": sub["partial"], "answer": parent["answer"],
                           "source": f"after_tool:{row['action']}"})
    write(args.out, chosen)
    print(json.dumps({"problems": len(by_problem), "points": len(chosen), "dropped_at_tool_limit": dict(dropped),
                      "by_depth": dict(Counter(p["depth"] for p in chosen)),
                      "by_dataset": dict(Counter(p["dataset"] for p in chosen)),
                      "by_source": dict(Counter(p["source"] for p in chosen))}, indent=1))


def write(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--points", required=True, type=Path)
    s.add_argument("--prescreen", required=True, type=Path)
    s.add_argument("--per-dataset", type=int, required=True)
    s.add_argument("--cap", nargs="*", default=[], help="per-dataset targets overriding --per-dataset, e.g. chartqa_1=20")
    t = sub.add_parser("subpoints")
    t.add_argument("--points", required=True, type=Path, help="the round's point file (for system/tools/answer)")
    t.add_argument("--branches", required=True, type=Path)
    t.add_argument("--per-problem", type=int, default=2)
    for p in (s, t):
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    select(args) if args.command == "select" else subpoints(args)


if __name__ == "__main__":
    main()
