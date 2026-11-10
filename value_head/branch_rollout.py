#!/usr/bin/env python3
"""Stage-2 step 3: at each decision point force every action (do(a)) and play K continuations out.

The policy and the teacher (backend of crop_ocr_image and imaginate_editing) cannot be served at the same
time, so continuations advance in phases over one state file:

    policy   (policy served)  start every (point, action) group, then generate; tools that need no model run
                              at once; a crop_ocr_image / imaginate_editing call parks the continuation
    teacher  (teacher served) run every parked call, continuations go back to generating
    finalize                  once every continuation is done, write branches.jsonl

Repeat policy/teacher until ``status`` shows nothing parked (at most 3 tool calls per continuation), then
finalize. Each phase saves the state as it goes and can be rerun after an interruption. In the policy phase
``--concurrency`` is the number of sequences kept running on the server: groups start and continuations advance
in one stream, each request taking a slot per sample and returning it when that sample ends.

    /root/miniconda3/envs/research/bin/python value_head/branch_rollout.py policy --points RUN/points.jsonl \\
        --out BRANCH_DIR --k 16 --base-url http://HOST:PORT/v1 --model MODEL
    /root/miniconda3/envs/research/bin/python value_head/branch_rollout.py teacher --out BRANCH_DIR
    /root/miniconda3/envs/research/bin/python value_head/branch_rollout.py status --out BRANCH_DIR
    /root/miniconda3/envs/research/bin/python value_head/branch_rollout.py finalize --out BRANCH_DIR

Actions are answer_now and every tool of the dataset by default; ``--actions`` picks others, e.g.
``--actions think_more``. The forced ``<evaluating>`` block is ``build_cavir_coldstart.render_evaluating``
with τ=0.7, seeded by (dp_id, action) and shared by the K continuations, whose first segments are sampled
in one request (n=K). After that the policy plays on by itself (``continuation=self``): tools run as in the
rollouts (``tool_env``), later decision points are filled by the policy, at most 3 tool calls in all. A
continuation whose first action is not the forced one is recorded (``followed: false``) and stopped.

think_more is played out with no tool call to the end. A segment that calls a tool is cut back to its last
``<evaluating>`` (or to the ``<tool_call>`` when no block precedes it) and generated again from there, so the
decision is resampled; after 3 such retries in a row the segment is generated with ``<tool_call>`` masked out by
``logit_bias``. Discarded segments go to ``blocked`` (and their tokens to ``blocked_tokens``, not ``tokens``); masked
segments whose block still selects a tool are counted in ``masked_tool_selections``. A tool call that gets through
the mask (spelled in plain tokens) ends the continuation as ``tool_blocked``, a failure.

``branches.jsonl`` has one line per continuation: dp_id, action, k, block, followed, finish (answer,
no_answer, length, tool_limit, tool_blocked, context, error, not_followed, parked_unfinished), prediction, correct, tool calls, tokens, the
answer-statement length after the block (answer_now), the generated text and later decision points.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_construction"))

from build_cavir_coldstart import CANONICAL_TOOL_ORDER, render_evaluating  # noqa: E402
from host import MODEL_BACKED_TOOLS  # noqa: E402
from openai import BadRequestError  # noqa: E402
from policy import MAX_TOOL_CALLS, Policy, ServerUnavailable, grade  # noqa: E402
from prompting import first_action, parse_tool_call  # noqa: E402
from tool_env import ToolEnv  # noqa: E402

TAU = 0.7
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")
SELECTION_RE = re.compile(r"selection:\s*([A-Za-z_]+)")
STOP = ["</tool_call>", "</answer>"]
NO_TOOL_ACTIONS = {"think_more"}       # played out with tool calls blocked to the end
TOOL_CALL_TOKEN = 248058               # "<tool_call>" in the policy tokenizer, one token
BLOCK_TOOL_CALL = {str(TOOL_CALL_TOKEN): -100}
RETRIES = 3                            # unmasked resamples of a decision before masking <tool_call>


def seeded(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:4], "big")


def dataset_tools(tools_field: str) -> list[str]:
    return [t for t in CANONICAL_TOOL_ORDER if t in {x["name"] for x in json.loads(tools_field)}]


class Store:
    """Continuations keyed by "dp_id|action|k", saved atomically to ``state.json``."""

    def __init__(self, out: Path):
        self.path = out / "state.json"
        data = json.loads(self.path.read_text()) if self.path.exists() else {"args": {}, "items": {}}
        self.args, self.items = data["args"], data["items"]
        self.lock = asyncio.Lock()
        self.dirty = 0

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"args": self.args, "items": self.items}, ensure_ascii=False))
        tmp.replace(self.path)
        self.dirty = 0

    async def touch(self, every: int = 200) -> None:
        async with self.lock:
            self.dirty += 1
            if self.dirty >= every:
                self.save()

    def counts(self) -> Counter:
        return Counter(item["status"] for item in self.items.values())


def apply_segment(item: dict, segment: dict) -> None:
    """Append a generated segment and decide what happens next (status: generate, tool, teacher, done)."""
    state = item["state"]
    text = segment["text"]
    if state["partial"] == "" and len(state["turns"]) > 1 and "<evaluating>" in text:
        item["sub_points"].append({"turns": copy.deepcopy(state["turns"]), "images": list(state["images"]),
                                   "partial": text[: text.index("<evaluating>")], "depth": item["depth"] + len(item["tool_calls"])})
    state["partial"] += text
    item["text"] += text
    item["tokens"] += segment["tokens"] or 0
    item["segments"] += 1
    if segment["finish_reason"] == "length":
        return finish(item, "length")
    call = parse_tool_call(text) if text.rstrip().endswith("</tool_call>") else None
    if call is None:
        return finish(item, "answer" if "</answer>" in text else "no_answer")
    if item["action"] in NO_TOOL_ACTIONS:
        return finish(item, "tool_blocked")
    if item["depth"] + len(item["tool_calls"]) >= MAX_TOOL_CALLS:
        return finish(item, "tool_limit")
    item["pending"] = call
    item["status"] = "teacher" if call["name"] in MODEL_BACKED_TOOLS and call["name"] in dataset_tools(state["tools"]) else "tool"


def finish(item: dict, how: str) -> None:
    item["status"], item["finish"], item["pending"] = "done", how, None
    if how == "answer":
        item["prediction"], item["correct"] = grade(item["state"]["partial"], item["answer"])


def block_attempt(item: dict, segment: dict, tokenizer) -> None:
    """Discard a no-tool continuation's segment that calls a tool; keep its text up to the decision before the call."""
    text = segment["text"]
    call = text.index("<tool_call>")
    block = text.rfind("<evaluating>", 0, call)
    kept = text[: block if block >= 0 else call]
    kept_tokens = len(tokenizer.encode(kept, add_special_tokens=False)) if kept else 0
    item["state"]["partial"] += kept
    item["text"] += kept
    item["tokens"] += kept_tokens
    item["blocked"].append(text[len(kept):])
    item["blocked_tokens"] += max((segment["tokens"] or 0) - kept_tokens, 0)
    item["retries"] += 1
    if segment["finish_reason"] == "length":  # cut off inside the call: the kept text is all there is
        finish(item, "length")


def wants_tool(text: str) -> bool:
    selections = SELECTION_RE.findall(text)
    return bool(selections) and selections[-1] not in ("answer_now", "think_more")


async def run_tool(item: dict, env: ToolEnv, out: Path) -> None:
    state, call = item["state"], item["pending"]
    work = out / "work" / item["dp_id"].replace(":", "_") / item["action"] / str(item["k"])
    result = await env.run(call, state["images"], work, allowed=dataset_tools(state["tools"]))
    item["tool_calls"].append({"name": call["name"], "arguments": call["arguments"], "error": result["error"],
                               "image": result["image"], "meta": result["meta"],
                               "observation": None if result["image"] else result["observation"]})
    state["turns"] += [state["partial"], result["observation"]]
    state["partial"] = ""
    item["text"] += "\n" + result["observation"] + "\n"
    item["pending"], item["status"] = None, "generate"


async def start_group(policy: Policy, store: Store, point: dict, action: str, k: int, seed: int, held: int = 0) -> list[dict]:
    """Sample the K first segments of one (point, action) group; returns its continuations."""
    tools = dataset_tools(point["tools"])
    block = render_evaluating(action, tools, TAU, random.Random(seeded(str(seed), point["dp_id"], action)))
    state = {key: point[key] for key in ("system", "tools", "turns", "images")}
    state["partial"] = f"{point['partial'].rstrip()}\n\n{block}\n\n"
    firsts = await policy.generate(state, n=k, stop=STOP, seed=seeded(str(seed), point["dp_id"], action, "n"), held=held)
    items = []
    for index, first in enumerate(firsts):
        taken = first_action(first["text"])
        if taken == "tool_call":
            # A call cut off by the token limit or malformed still shows which tool the policy chose; it then
            # fails as an outcome of that action (finish length / no_answer) instead of counting as not followed.
            named = FUNCTION_RE.search(first["text"])
            taken = named.group(1).strip() if named else "tool_call_unparsed"
        item = {"dp_id": point["dp_id"], "id": point["id"], "dataset": point["dataset"], "split": point["split"],
                "depth": point["depth"], "answer": point["answer"], "action": action, "k": index, "block": block,
                "first_action": taken, "followed": taken == action,
                "statement_chars": (len(first["text"][: first["text"].find("<answer>")].strip())
                                    if action == "answer_now" and "<answer>" in first["text"] else None),
                "state": copy.deepcopy(state), "status": "generate", "pending": None, "finish": None,
                "prediction": None, "correct": None, "tool_calls": [], "tokens": 0, "segments": 0, "text": "",
                "sub_points": []}
        if action in NO_TOOL_ACTIONS:
            item |= {"blocked": [], "blocked_tokens": 0, "masked_tool_selections": 0, "retries": 0}
        if not item["followed"]:
            item["text"], item["tokens"] = first["text"], first["tokens"]
            finish(item, "not_followed")
        elif action in NO_TOOL_ACTIONS and "<tool_call>" in first["text"]:
            block_attempt(item, first, policy.tokenizer)
        else:
            apply_segment(item, first)
        store.items[f"{point['dp_id']}|{action}|{index}"] = item
        items.append(item)
    await store.touch(every=20)
    return items


async def advance(policy: Policy, env: ToolEnv, store: Store, item: dict, out: Path, seed: int) -> None:
    """Generate (running model-free tools inline) until the continuation is done or needs the teacher."""
    while item["status"] in ("generate", "tool"):
        if item["status"] == "tool":
            try:
                await run_tool(item, env, out)
            except Exception as exc:  # tool server crash or unreadable output
                item["error"] = f"{type(exc).__name__}: {exc}"[:300]
                finish(item, "error")
            continue
        no_tool = item["action"] in NO_TOOL_ACTIONS
        masked = no_tool and item["retries"] >= RETRIES
        parts = [str(seed), item["dp_id"], item["action"], str(item["k"]), str(item["segments"])]
        if no_tool and item["retries"]:
            parts.append(f"retry{item['retries']}")
        try:
            segment = (await policy.generate(item["state"], stop=STOP, seed=seeded(*parts),
                                             logit_bias=BLOCK_TOOL_CALL if masked else None))[0]
        except BadRequestError:  # prompt over the server's max_model_len
            finish(item, "context")
            break
        except ServerUnavailable:  # server gone or failing: keep the continuation as it is and stop
            raise
        except Exception as exc:  # one broken continuation must not stop the phase
            item["error"] = f"{type(exc).__name__}: {exc}"[:300]
            finish(item, "error")
            break
        if no_tool and not masked and "<tool_call>" in segment["text"]:
            block_attempt(item, segment, policy.tokenizer)
            continue
        if no_tool:
            item["masked_tool_selections"] += masked and wants_tool(segment["text"])
            item["retries"] = 0
        apply_segment(item, segment)
    await store.touch()


async def phase_policy(args) -> None:
    store = Store(args.out)
    if not store.args:
        store.args = {"points": str(args.points), "k": args.k, "seed": args.seed, "model": args.model,
                      "datasets": args.datasets, "limit": args.limit, "actions": args.actions}
    elif ((store.args["k"], store.args["seed"], store.args["points"], store.args.get("actions"))
          != (args.k, args.seed, str(args.points), args.actions)):
        raise SystemExit(f"{store.path} was started with other settings: {store.args}")
    points = [json.loads(line) for line in open(args.points)]
    if args.datasets:
        points = [p for p in points if p["dataset"] in args.datasets]
    if args.limit:
        points = points[: args.limit]
    started = {key.rsplit("|", 1)[0] for key in store.items}
    def actions(point):
        tools = dataset_tools(point["tools"])
        if not args.actions:
            return ["answer_now"] + tools
        return [a for a in args.actions if a in ("answer_now", "think_more") or a in tools]

    groups = [(p, a) for p in points for a in actions(p) if f"{p['dp_id']}|{a}" not in started]
    policy = Policy(args.base_url, args.model, concurrency=args.concurrency)
    todo = [item for item in store.items.values() if item["status"] in ("generate", "tool")]
    print(f"{len(points)} decision points; {len(groups)} new (point, action) groups, k={args.k}; "
          f"{len(todo)} continuations to resume; state {dict(store.counts())}", flush=True)
    tools = sorted({t for item in todo if item["action"] not in NO_TOOL_ACTIONS for t in dataset_tools(item["state"]["tools"])}
                   | {t for p, a in groups if a not in NO_TOOL_ACTIONS for t in dataset_tools(p["tools"])})

    # One stream instead of two waves: a group starts as soon as K of the --concurrency slots are free,
    # and its continuations go on generating right away; a continuation's next segment needs one free slot.
    tasks, failed = [], []
    async with (ToolEnv(tools, copies=args.tool_copies) if tools else nullcontext()) as env:
        def spawn(coro):
            task = asyncio.create_task(coro)
            task.add_done_callback(lambda t: failed.append(t) if not t.cancelled() and t.exception() else None)
            tasks.append(task)

        def advance_all(items):
            for item in items:
                if item["status"] in ("generate", "tool"):
                    spawn(advance(policy, env, store, item, args.out, args.seed))

        async def group(point, action, held):
            advance_all(await start_group(policy, store, point, action, args.k, args.seed, held=held))

        try:
            advance_all(todo)
            for point, action in groups:
                held = await policy.acquire(args.k)
                if failed:  # stop starting groups once something has failed
                    await policy.release(held)
                    failed[0].result()
                spawn(group(point, action, held))
            while pending := [t for t in tasks if not t.done()]:
                await asyncio.gather(*pending)
            for task in tasks:
                task.result()
        except ServerUnavailable as exc:
            # Unfinished continuations keep their status and groups not yet sampled are not in the state, so
            # rerunning the same command once the server is back resumes from here.
            for task in tasks:
                task.cancel()
            store.save()
            await policy.close()
            raise SystemExit(f"server unreachable or failing ({exc}); state saved, rerun this command when it is back: "
                             f"{dict(store.counts())}")
    store.save()
    await policy.close()
    report(store)


async def phase_teacher(args) -> None:
    store = Store(args.out)
    parked = [item for item in store.items.values() if item["status"] == "teacher"]
    print(f"running {len(parked)} teacher-backed tool calls", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with ToolEnv(sorted(MODEL_BACKED_TOOLS), copies=args.tool_copies) as env:
        async def one(item):
            async with semaphore:
                try:
                    await run_tool(item, env, args.out)
                except Exception as exc:  # one broken call must not stop the phase
                    item["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    finish(item, "error")
            await store.touch()
        await asyncio.gather(*(one(item) for item in parked))
    store.save()
    report(store)


def report(store: Store) -> None:
    counts = store.counts()
    print(json.dumps({"status": dict(counts), "finish": dict(Counter(i["finish"] for i in store.items.values() if i["finish"]))}))
    no_tool = [i for i in store.items.values() if i["action"] in NO_TOOL_ACTIONS]
    if no_tool:
        print(json.dumps({"no_tool_continuations": len(no_tool),
                          "with_blocked_call": sum(bool(i["blocked"]) for i in no_tool),
                          "blocked_calls": sum(len(i["blocked"]) for i in no_tool),
                          "masked_tool_selections": sum(i["masked_tool_selections"] for i in no_tool)}))
    if counts.get("generate") or counts.get("tool"):
        print("NEXT: serve the policy, then run the policy phase")
    elif counts.get("teacher"):
        print("NEXT: serve the teacher, then run the teacher phase")
    else:
        print("NEXT: every continuation is done; run finalize")


def phase_finalize(args) -> None:
    store = Store(args.out)
    if args.drop_parked:  # a handful left waiting for the teacher: excluded instead of another server swap
        for item in store.items.values():
            if item["status"] == "teacher":
                item["status"], item["finish"] = "done", "parked_unfinished"
    if any(item["status"] != "done" for item in store.items.values()):
        raise SystemExit(f"not finished: {dict(store.counts())}")
    path = args.out / "branches.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists")
    with open(path, "w") as handle:
        for key in sorted(store.items):
            item = {k: v for k, v in store.items[key].items() if k not in ("state", "status", "pending", "answer")}
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {len(store.items)} continuations to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-action branch rollouts in policy/teacher phases.")
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("policy")
    p.add_argument("--points", required=True, type=Path)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=0, help="first N decision points only (smoke runs)")
    p.add_argument("--actions", nargs="+", default=None,
                   help="forced actions (answer_now, think_more, tool names); default answer_now and the dataset's tools")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=256,
                   help="sequences kept running on the server (a group's first request takes K of them)")
    p.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("teacher")
    t.add_argument("--concurrency", type=int, default=16)
    for sub_parser in (p, t):
        # Each copy runs tool calls one at a time; a code call spawns `conda run` (2-3 busy processes),
        # so 4 copies keep it within about half of a DGX Spark's 20 cores.
        sub_parser.add_argument("--tool-copies", type=int, default=4, help="processes per tool server")
    f = sub.add_parser("finalize")
    f.add_argument("--drop-parked", action="store_true",
                   help="finish continuations still waiting for the teacher as parked_unfinished (excluded from labels)")
    for name in ("policy", "teacher", "status", "finalize"):
        sub_parser = {"policy": p, "teacher": t, "finalize": f}.get(name) or sub.add_parser(name)
        sub_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "policy":
        asyncio.run(phase_policy(args))
    elif args.phase == "teacher":
        asyncio.run(phase_teacher(args))
    elif args.phase == "status":
        report(Store(args.out))
    else:
        phase_finalize(args)


if __name__ == "__main__":
    main()
