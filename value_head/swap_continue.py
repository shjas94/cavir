#!/usr/bin/env python3
"""Continue stored branches from right after the forced tool call with a swapped tool output.

For each target call (dp_id, forced tool) every stored continuation k keeps its forced call's arguments but gets an
output from a region of the same size moved where it does not touch the answer regions (``rq1_annotations``; the
original applied box when the problem has none): crop_ocr_image re-reads that region (teacher), crop_zoomin crops it.
The swapped output is judged with the faithfulness rule (``rq1_faithfulness.judge``); an output whose box touches an avoided
region or whose F is 1 is redrawn elsewhere (``retries``). The state right after the call is rebuilt exactly as
``branch_rollout`` built it (point partial + forced block + first segment, then the tool turn), so the swapped continuation differs from the stored one only in the output.

    prepare  (teacher served for crop_ocr_image) build the swapped calls and write OUT/state.json, a branch_rollout
             store whose continuations are all at "generate"; prints the policy command that resumes them
    exclude  after a policy phase: a continuation that calls crop_ocr_image or imaginate_editing again ends as
             ``ocr_again`` or ``imaginate_again``, so
             no teacher phase is needed

    /root/miniconda3/envs/research/bin/python value_head/swap_continue.py prepare --targets TARGETS.jsonl --model-tag 2B \\
        --points P1.jsonl --points P2.jsonl --branches B1.jsonl --branches B2.jsonl --annotations ANN.jsonl --out OUT
    branch_rollout.py policy --points OUT/no_points.jsonl --out OUT --k 16 --seed 0 --base-url ... --model ...
    /root/miniconda3/envs/research/bin/python value_head/swap_continue.py exclude --out OUT
    branch_rollout.py finalize --out OUT

Targets: one line per call, {model, dp_id, action}; lines of other models are skipped. The seed is the stored runs'
(0), so each swapped continuation's next segment is sampled with the seed the stored one had. finalize's
branches.jsonl keeps the usual fields plus ``model`` and ``swap`` (box, avoided regions, retries, swapped F and output);
calls that found no position are in OUT/skipped.jsonl. Grade the result with ``regrade_branches.py`` like the stored
branches. Refuses an OUT that already has a store unless ``--resume``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branch_rollout import Store, finish  # noqa: E402
from rq1_faithfulness import judge  # noqa: E402
from tool_env import ToolEnv  # noqa: E402

TOOLS = ("crop_ocr_image", "crop_zoomin")
SEED = 0              # seed of the stored 2B and 4B branch runs
K = 16
POSITIONS = 500       # random positions tried for one box
CALLS = 5             # tool calls tried per continuation before it is skipped


def touches(box: list[float], region: list[float]) -> bool:
    if (region[2] - region[0]) * (region[3] - region[1]) <= 0:  # zero-area region: its centre
        cx, cy = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
    return min(box[2], region[2]) > max(box[0], region[0]) and min(box[3], region[3]) > max(box[1], region[1])


def unusable(call: dict) -> str | None:
    """Why a stored forced call cannot be swapped: no rule F for it, or no box of its own to copy."""
    if call["arguments"].get("image_index", 0) != 0:   # rq1_faithfulness does not judge calls on a tool output image
        return "call_on_tool_output_image"
    box = call["arguments"].get("crop_box")
    if call["error"] or "applied_crop_box" not in call["meta"] or not box or any(not 0 <= v <= 1000 for v in box):
        return "original_call_error"
    return None


def free_box(size: tuple[int, int], avoid: list[list[float]], rng: random.Random) -> list[int] | None:
    """A box of ``size`` (0–1000 scale) at a random position touching no avoided region."""
    w, h = size
    for _ in range(POSITIONS):
        x0, y0 = rng.randint(0, 1000 - w), rng.randint(0, 1000 - h)
        box = [x0, y0, x0 + w, y0 + h]
        if not any(touches(box, r) for r in avoid):
            return box
    return None


async def swap_one(env: ToolEnv, point: dict, row: dict, problem: dict, work: Path) -> tuple[dict | None, dict]:
    """The swapped call record and its images list, or None with the reason."""
    call = row["tool_calls"][0]
    applied = call["meta"]["applied_crop_box"]["normalized_1000"]
    requested = call["arguments"]["crop_box"]
    size = (int(round(requested[2] - requested[0])), int(round(requested[3] - requested[1])))
    avoid = [r["box"] for r in problem["regions"]] or [applied]
    rng = random.Random(f"{SEED}:{row['dp_id']}:{row['action']}:{row['k']}")
    info = {"avoided": "answer_regions" if problem["regions"] else "original_box", "retries": 0}
    for attempt in range(CALLS):
        box = free_box(size, avoid, rng)
        if box is None:
            return None, info | {"reason": "no_free_position"}
        images = list(point["images"])
        result = await env.run({"name": row["action"], "arguments": call["arguments"] | {"crop_box": box}},
                               images, work / f"try{attempt}")
        if result["error"]:
            raise RuntimeError(f"{row['action']} failed on {row['dp_id']} k={row['k']}: {result['observation'][:200]}")
        new_applied = result["meta"]["applied_crop_box"]["normalized_1000"]
        record = {"name": row["action"], "arguments": call["arguments"] | {"crop_box": box}, "error": None,
                  "image": result["image"], "meta": result["meta"],
                  "observation": None if result["image"] else result["observation"]}
        f, reason, _ = judge({"tool_calls": [record], "action": row["action"], "dataset": row["dataset"]}, problem)
        if f is None:   # e.g. plotqa OCR goes to the LLM judge, so the swapped output cannot be checked
            return None, info | {"reason": f"no_rule_f: {reason}"}
        if not any(touches(new_applied, r) for r in avoid) and f == 0:
            text = "" if result["image"] else result["observation"].removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
            return {"call": record, "images": images, "observation": result["observation"]}, \
                info | {"box": box, "applied": new_applied, "f": f, "f_reason": reason,
                        "output": result["image"] or result["observation"],
                        "output_empty": None if result["image"] else not text}
        info["retries"] += 1
    return None, info | {"reason": "swapped_f1_or_overlap"}


def item_for(point: dict, row: dict, swapped: dict, info: dict, tag: str) -> dict:
    """A branch_rollout continuation standing right after the swapped call, as run_tool leaves it."""
    text = row["text"]
    first = text[: text.index("</tool_call>") + len("</tool_call>")]
    state = {key: copy.deepcopy(point[key]) for key in ("system", "tools", "turns")}
    state["turns"] += [f"{point['partial'].rstrip()}\n\n{row['block']}\n\n{first}", swapped["observation"]]
    state["images"], state["partial"] = swapped["images"], ""
    return {"dp_id": row["dp_id"], "id": row["id"], "dataset": row["dataset"], "split": point["split"],
            "depth": point["depth"], "answer": point["answer"], "action": row["action"], "k": row["k"],
            "block": row["block"], "first_action": row["action"], "followed": True, "statement_chars": None,
            "state": state, "status": "generate", "pending": None, "finish": None, "prediction": None,
            "correct": None, "tool_calls": [swapped["call"]], "tokens": 0, "segments": 1,
            "text": f"{first}\n{swapped['observation']}\n", "sub_points": [], "model": tag, "swap": info}


async def prepare(args) -> None:
    store = Store(args.out)
    if store.items and not args.resume:
        raise SystemExit(f"{store.path} exists; --resume to continue it")
    no_points = (args.out / "no_points.jsonl").resolve()
    no_points.touch()
    store.args = {"points": str(no_points), "k": K, "seed": SEED, "model": None, "datasets": None, "limit": 0,
                  "actions": None, "swap": {"targets": str(args.targets), "model_tag": args.model_tag,
                                            "points": [str(p) for p in args.points], "branches": [str(b) for b in args.branches],
                                            "annotations": str(args.annotations),
                                            "teacher": {"model": os.getenv("MODEL_NAME"), "base_url": os.getenv("DIRECT_BASE_URL")}}}
    targets = {(t["dp_id"], t["action"]) for t in map(json.loads, open(args.targets)) if t["model"] == args.model_tag}
    if any(action not in TOOLS for _, action in targets):
        raise SystemExit(f"targets hold tools other than {TOOLS}")
    points = {p["dp_id"]: p for path in args.points for p in map(json.loads, open(path))}
    problems = {p["id"]: p for p in map(json.loads, open(args.annotations))}
    rows = [r for path in args.branches for r in map(json.loads, open(path)) if (r["dp_id"], r["action"]) in targets]
    skipped_path = args.out / "skipped.jsonl"
    done = set(store.items) | {f"{s['dp_id']}|{s['action']}|{s['k']}" for s in map(json.loads, open(skipped_path))} \
        if skipped_path.exists() else set(store.items)
    rows = [r for r in rows if r["tool_calls"] and r["tool_calls"][0]["name"] == r["action"]
            and f"{r['dp_id']}|{r['action']}|{r['k']}" not in done]
    if args.limit:
        rows = rows[: args.limit]
    found = {(r["dp_id"], r["action"]) for r in rows}
    print(f"{len(targets)} target calls ({len(found)} with stored continuations to swap), {len(rows)} continuations", flush=True)
    skipped = open(skipped_path, "a")
    usable = []
    for row in rows:
        reason = unusable(row["tool_calls"][0])
        if reason:
            skipped.write(json.dumps({k: row[k] for k in ("dp_id", "action", "k")} | {"reason": reason}) + "\n")
        else:
            usable.append(row)
    rows = usable
    semaphore = asyncio.Semaphore(args.concurrency)
    async with ToolEnv(sorted({r["action"] for r in rows}), copies=args.tool_copies) as env:
        async def one(row):
            async with semaphore:
                work = (args.out / "swap_work" / row["dp_id"].replace(":", "_") / row["action"] / str(row["k"])).resolve()
                swapped, info = await swap_one(env, points[row["dp_id"]], row, problems[row["id"]], work)
            if swapped is None:
                skipped.write(json.dumps({k: row[k] for k in ("dp_id", "action", "k")} | info) + "\n")
                skipped.flush()
                return
            store.items[f"{row['dp_id']}|{row['action']}|{row['k']}"] = item_for(points[row["dp_id"]], row, swapped, info, args.model_tag)
            await store.touch(every=50)
        try:
            await asyncio.gather(*(one(r) for r in rows))
        finally:
            store.save()
            skipped.close()
    print(json.dumps({"continuations": len(store.items), "status": dict(store.counts())}))
    print(f"NEXT: serve the {args.model_tag} policy, then: branch_rollout.py policy --points {no_points} --out {args.out.resolve()} "
          f"--k {K} --seed {SEED} --base-url URL --model MODEL; then swap_continue.py exclude --out {args.out}")


EXCLUDED_AGAIN = {"crop_ocr_image": "ocr_again", "imaginate_editing": "imaginate_again"}   # every model-backed tool


def exclude(args) -> None:
    store = Store(args.out)
    hit = Counter()
    for item in store.items.values():
        if item["status"] == "teacher":
            hit[EXCLUDED_AGAIN[item["pending"]["name"]]] += 1
            finish(item, EXCLUDED_AGAIN[item["pending"]["name"]])
    store.save()
    print(json.dumps(dict(hit) | {"status": dict(store.counts())}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue stored branches with a swapped tool output.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--model-tag", required=True, choices=("2B", "4B"))
    p.add_argument("--points", action="append", required=True, type=Path)
    p.add_argument("--branches", action="append", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=32, help="tool calls at a time (teacher OCR)")
    p.add_argument("--tool-copies", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first N continuations only (smoke runs)")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("exclude")
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.mkdir(parents=True, exist_ok=True)
        asyncio.run(prepare(args))
    else:
        exclude(args)


if __name__ == "__main__":
    main()
