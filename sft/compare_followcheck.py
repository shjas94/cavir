#!/usr/bin/env python3
"""Compare two selection-following runs (``selection_following.py`` output dirs) on the same prompts.

    python compare_followcheck.py --runs v2=results/followcheck/full_450 noarith=results/followcheck_noarith/full_450 \
        --out results/followcheck_noarith/compare_v2.json

Rates come with Wilson 95% intervals. Besides each run's own groups, it reports the injections both runs share
(same prompt id and target), which removes targets that exist in only one run, such as the arithmetic tools
of the v2 tool lists; ``exclude_targets`` lists the targets dropped from that comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL = ("answer_now", "think_more")


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def rate(rows: list[dict]) -> dict:
    k = sum(r["followed"] for r in rows)
    return {"n": len(rows), "followed": k, "rate": round(k / len(rows), 4) if rows else None, "ci95": wilson(k, len(rows))}


def groups(rows: list[dict]) -> dict:
    kind = lambda r: r["target"] if r["target"] in SPECIAL else "tool"
    out = defaultdict(list)
    for r in rows:
        out["all"].append(r)
        out[f"type / {kind(r)}"].append(r)
        if kind(r) == "tool":
            out[f"tool / {r['target']}"].append(r)
        out[f"dataset / {r['dataset']}"].append(r)
        out[f"point / {r['point']}"].append(r)
    return {k: rate(v) for k, v in sorted(out.items())}


def misses(rows: list[dict]) -> dict:
    kind = lambda a: a if a in SPECIAL + ("none",) else "tool"
    return dict(Counter(f"{kind(r['target'])} -> {kind(r['action'])}" for r in rows if not r["followed"]).most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare selection-following runs.")
    parser.add_argument("--runs", nargs=2, required=True, metavar="LABEL=DIR")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    runs = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        injections = [json.loads(line) for line in open(Path(path) / "injections.jsonl")]
        first_turns = [json.loads(line) for line in open(Path(path) / "first_turns.jsonl")]
        runs[label] = {"dir": path, "injections": injections, "first_turns": first_turns}

    report = {"runs": {}}
    for label, run in runs.items():
        report["runs"][label] = {
            "dir": run["dir"], "prompts": len(run["first_turns"]),
            "decision_points": dict(Counter(t["point"] for t in run["first_turns"])),
            "natural_selection": dict(Counter(t["natural_selection"] for t in run["first_turns"] if t["point"] == "natural")),
            "follow": groups(run["injections"]), "misses": misses(run["injections"]),
        }

    (a, ra), (b, rb) = runs.items()
    key = lambda r: (r["id"], r["target"])
    keys_a, keys_b = {key(r) for r in ra["injections"]}, {key(r) for r in rb["injections"]}
    shared = keys_a & keys_b
    only = lambda keys: dict(Counter(t for _, t in keys).most_common())
    shared_rows = {label: [r for r in run["injections"] if key(r) in shared] for label, run in runs.items()}
    report["shared"] = {
        "injections": len(shared),
        "exclude_targets": {a: only(keys_a - shared), b: only(keys_b - shared)},
        "follow": {label: groups(rows) for label, rows in shared_rows.items()},
        "misses": {label: misses(rows) for label, rows in shared_rows.items()},
    }
    by_key = {label: {key(r): r["followed"] for r in rows} for label, rows in shared_rows.items()}
    report["shared"]["paired"] = dict(Counter(
        f"{a}={'Y' if by_key[a][k] else 'N'} {b}={'Y' if by_key[b][k] else 'N'}" for k in shared))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for label, run in report["runs"].items():
        print(label, run["follow"]["all"], run["decision_points"], run["misses"])
    for label, follow in report["shared"]["follow"].items():
        print("shared", label, follow["all"])
    print("paired", report["shared"]["paired"])


if __name__ == "__main__":
    main()
