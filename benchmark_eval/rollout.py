#!/usr/bin/env python3
"""Solve benchmark problems with a served model and its tool, then branch at the first tool call.

Run with the research env against one model the user serves on the host (OpenAI-compatible vLLM server):

    python benchmark_eval/rollout.py natural --problems PROBLEMS.jsonl --out OUT --base-url URL --model NAME --tokenizer PATH
    python benchmark_eval/rollout.py branch  --out OUT --base-url URL --model NAME --tokenizer PATH
    python benchmark_eval/rollout.py predictions --out OUT

PROBLEMS (``prepare.py``): one line per problem, {index, dataset, prompt, image, ...}; ``prompt`` is VLMEvalKit's
official question text and ``image`` the original image file. The dataset fixes the one tool the model gets
(DATASET_TOOL). Nothing is resized here; the server's processor applies the model's own image limit, and a produced
image larger than that limit is shrunk to it before sending (the server would shrink it the same way).

- natural: every problem once, tools available (at most MAX_TOOL_CALLS calls). OUT/natural.jsonl; ``call_note`` marks a
  solution that wrote a call but has no executed first call (more than MAX_TOOL_CALLS calls in one message, or call
  text that does not parse), which is not branched. A request the server rejects (400, e.g. a full context) ends the
  solution without an answer and keeps what was generated.
- branch: for each natural solution with a tool call, the shared prefix is everything before its first tool call.
  execute: the first assistant message with its calls and the tool results it got, played on K times (later calls
  allowed); block: the same assistant message without its calls, continued K times with ``<tool_call>`` banned
  (``bad_words``; ``logit_bias`` is refused by servers running speculative decoding). OUT/branches.jsonl.
- predictions: OUT/predictions/{natural, execute_r<k>, block_r<k>}.jsonl, one {index, prediction, source} per problem
  in problem order; a problem without a tool call keeps its natural prediction in every file (``source`` natural).

The prompt is rendered by the server's own chat template (assistant reasoning goes in the ``reasoning`` key, the
only one this vLLM puts back into the template). Generated text is decoded from the returned token ids, because the
server's reasoning parser files a continued answer under reasoning. Sampling is VLMEvalKit's Qwen3.5 thinking-mode
API setting; ``max_tokens`` is left to the server (the rest of the context). A prediction is the text after the last
``</think>`` of the final turn without tool calls; for a block continuation, which starts after ``</think>``, it is
the continuation itself. Both steps append and skip finished problems, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "value_head"))
sys.path.insert(0, str(ROOT / "dataset_construction"))

from host import HOST_INJECTED_FIELDS, _image_index_property  # noqa: E402
from prompting import PARAM_RE, TOOL_CALL_RE  # noqa: E402
from tool_env import ToolEnv  # noqa: E402

load_dotenv(ROOT / "dataset_construction" / ".env")
API_KEY = os.environ.get("DIRECT_API_KEY", "EMPTY")   # the teacher needs it; a server started without --api-key ignores it

DATASET_TOOL = {"VStarBench": "crop_zoomin", "DocVQA_VAL": "crop_ocr_image", "ChartQA_TEST": "crop_ocr_image"}
K = 4                    # continuations per condition
MAX_TOOL_CALLS = 3       # per solution, as in the cold start rollouts (host.MAX_TOOL_CALLS)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}            # VLMEvalKit Qwen3.5 ThinkMode
EXTRA = {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "skip_special_tokens": False,
         "return_tokens_as_token_ids": True}
TOOL_CALL_TOKEN = "<tool_call>"


def seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def data_uri(path: str) -> str:
    with Image.open(path) as image:
        mime = Image.MIME[image.format]
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def to_request(messages: list[dict]) -> list[dict]:
    """Stored messages refer to images by path ({"type": "image", "path"}); the request carries them as data URIs."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = [{"type": "image_url", "image_url": {"url": data_uri(p["path"])}} if p["type"] == "image" else p
                            for p in m["content"]]
        out.append(m)
    return out


def parse_calls(text: str) -> list[dict]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        arguments = {}
        for key, value in PARAM_RE.findall(match.group(2)):
            try:
                arguments[key.strip()] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key.strip()] = value
        calls.append({"name": match.group(1).strip(), "arguments": arguments})
    return calls


def parse_turn(raw: str) -> dict:
    """A turn generated after the template's ``<think>\\n``: reasoning, the content before any call, and the calls."""
    raw = raw.split("<|im_end|>")[0]
    reasoning, closed, rest = raw.partition("</think>")
    if not closed:   # cut off while thinking
        return {"reasoning": reasoning.strip(), "content": "", "calls": []}
    return {"reasoning": reasoning.strip(), "content": rest.split(TOOL_CALL_TOKEN)[0].strip(), "calls": parse_calls(rest)}


class Runner:
    def __init__(self, args):
        self.args = args
        self.client = AsyncOpenAI(base_url=args.base_url, api_key=API_KEY, timeout=3600)
        processor = AutoProcessor.from_pretrained(args.tokenizer)
        self.tokenizer = processor.tokenizer
        if len(self.tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)) != 1:
            raise SystemExit(f"{TOOL_CALL_TOKEN} is not one token for {args.tokenizer}; bad_words would not ban the call")
        self.max_pixels = processor.image_processor.size["longest_edge"]
        self.semaphore = asyncio.Semaphore(args.concurrency)

    async def complete(self, messages: list[dict], tools: list[dict], n: int, request_seed: int, block: bool = False) -> list[dict]:
        extra = dict(EXTRA)
        if block:
            extra.update(continue_final_message=True, add_generation_prompt=False, bad_words=[TOOL_CALL_TOKEN])
        async with self.semaphore:
            reply = await self.client.chat.completions.create(
                model=self.args.model, messages=to_request(messages), tools=tools, n=n, seed=request_seed,
                logprobs=True, extra_body=extra, **SAMPLING)
        outs = []
        for choice in sorted(reply.choices, key=lambda c: c.index):
            ids = [int(t.token.removeprefix("token_id:")) for t in (choice.logprobs.content or [])]
            outs.append({"raw": self.tokenizer.decode(ids, skip_special_tokens=False), "finish": choice.finish_reason,
                         "tokens": len(ids)})
        return outs

    async def play(self, messages: list[dict], calls_used: int, env: ToolEnv, tools: list[dict], tool: str,
                   work: Path, request_seed: int) -> dict:
        """Generate and run tools until a turn without calls; returns the turns, tool results and the prediction."""
        messages = list(messages)
        turns, results = [], []
        while True:
            try:
                out = (await self.complete(messages, tools, 1, seed(request_seed, len(turns))))[0]
            except BadRequestError as exc:   # e.g. the context is full: no answer, the solution so far is kept
                return {"turns": turns, "results": results, "prediction": "", "finish": f"error: {exc}"[:500], "messages": messages}
            turn = parse_turn(out["raw"]) | out
            turns.append(turn)
            if not turn["calls"] or out["finish"] == "length":
                return {"turns": turns, "results": results, "prediction": turn["content"],
                        "finish": out["finish"], "messages": messages}
            if calls_used + len(turn["calls"]) > MAX_TOOL_CALLS:
                return {"turns": turns, "results": results, "prediction": "", "finish": "tool_limit", "messages": messages}
            ids = [f"call_{uuid.uuid4().hex[:12]}" for _ in turn["calls"]]
            messages.append({"role": "assistant", "content": turn["content"], "reasoning": turn["reasoning"],
                             "tool_calls": [{"id": i, "type": "function",
                                             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                            for i, c in zip(ids, turn["calls"])]})
            images = [p["path"] for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p["type"] == "image"]
            for call_id, call in zip(ids, turn["calls"]):
                result = await env.run(call, images, work, allowed=[tool])
                results.append({"turn": len(turns) - 1, "call": call} | {k: result.get(k) for k in ("image", "text", "error", "meta")})
                if result["image"]:
                    content = [{"type": "text", "text": f"[image_index={len(images) - 1}] Produced image is shown below."},
                               {"type": "image", "path": result["image"]}]
                else:
                    content = result["text"] or "(empty tool output)"
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            calls_used += len(turn["calls"])

    def tool_env(self, tools: set[str]) -> ToolEnv:
        model = {"model_name": self.args.model, "host_url": self.args.base_url, "api_key": API_KEY}
        return ToolEnv(sorted(tools), copies=self.args.tool_copies, max_pixels=self.max_pixels, model=model)


async def tool_schemas(env: ToolEnv) -> dict[str, dict]:
    """The model-facing schema of each tool, as host.py builds it (host-injected fields removed, image_index added)."""
    schemas = {}
    for name, sessions in env.sessions.items():
        for t in (await sessions[0].list_tools()).tools:
            if t.name != name:
                continue
            schema = json.loads(json.dumps(t.input_schema))
            for field in HOST_INJECTED_FIELDS:
                schema.get("properties", {}).pop(field, None)
            schema["required"] = [r for r in schema.get("required", []) if r not in HOST_INJECTED_FIELDS]
            if (index := _image_index_property(name)) is not None:
                schema["properties"]["image_index"] = index
            schemas[name] = {"type": "function", "function": {"name": name, "description": t.description, "parameters": schema}}
    return schemas


def done_ids(path: Path) -> set[str]:
    return {json.loads(line)["index"] for line in open(path)} if path.exists() else set()


async def run_natural(args) -> None:
    problems = [json.loads(line) for line in open(args.problems)]
    out = args.out / "natural.jsonl"
    done = done_ids(out)
    todo = [p for p in problems if p["index"] not in done]
    print(f"natural: {len(todo)} of {len(problems)} problems to solve", flush=True)
    runner = Runner(args)
    lock = asyncio.Lock()
    async with runner.tool_env({DATASET_TOOL[p["dataset"]] for p in problems}) as env:
        schemas = await tool_schemas(env)

        async def one(p):
            tool = DATASET_TOOL[p["dataset"]]
            messages = [{"role": "user", "content": [{"type": "image", "path": p["image"]}, {"type": "text", "text": p["prompt"]}]}]
            solved = await runner.play(messages, 0, env, [schemas[tool]], tool, args.out / "work" / p["index"],
                                       seed(p["index"], "natural"))
            # Any error but a rejected request (server down) stops the run; finished problems are kept and a rerun resumes.
            solved["call_note"] = call_note(solved)
            async with lock:
                with open(out, "a") as handle:
                    handle.write(json.dumps({"index": p["index"], "dataset": p["dataset"], "tool": tool} | solved,
                                            ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(p) for p in todo))
    await runner.client.close()


def call_note(solved: dict) -> str | None:
    """Why a solution that wrote a tool call has no executed first call, so it is not branched (reported, not hidden):
    its first calling message held more than MAX_TOOL_CALLS calls, or the call text did not parse."""
    if first_call_message(solved["messages"]) is not None:
        return None
    if any(turn["calls"] for turn in solved["turns"]):
        return "first_message_over_limit"
    if any(TOOL_CALL_TOKEN in turn["raw"] for turn in solved["turns"]):
        return "malformed_call"
    return None


def first_call_message(messages: list[dict]) -> int | None:
    return next((i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")), None)


async def run_branch(args) -> None:
    natural = [json.loads(line) for line in open(args.out / "natural.jsonl")]
    out = args.out / "branches.jsonl"
    done = done_ids(out)
    todo = [r for r in natural if first_call_message(r["messages"]) is not None and r["index"] not in done]
    print(f"branch: {len(todo)} solutions with a tool call to branch ({len(done)} done)", flush=True)
    runner = Runner(args)
    lock = asyncio.Lock()
    async with runner.tool_env({r["tool"] for r in natural}) as env:
        schemas = await tool_schemas(env)

        async def one(r):
            messages, tools = r["messages"], [schemas[r["tool"]]]
            j = first_call_message(messages)
            calls = len(messages[j]["tool_calls"])
            shared = messages[:j + 1 + calls]   # first call and its results, as the natural solution got them
            execute = await asyncio.gather(*(
                runner.play(shared, calls, env, tools, r["tool"], args.out / "work" / r["index"] / f"execute_r{k}",
                            seed(r["index"], "execute", k)) for k in range(K)))
            stripped = {key: value for key, value in messages[j].items() if key != "tool_calls"}
            try:
                block = await runner.complete(messages[:j] + [stripped], tools, K, seed(r["index"], "block"), block=True)
            except BadRequestError as exc:
                block = [{"raw": "", "finish": f"error: {exc}"[:500], "tokens": 0}] * K
            row = {"index": r["index"], "dataset": r["dataset"], "first_call": j,
                   "execute": [{key: e[key] for key in ("turns", "results", "prediction", "finish")} for e in execute],
                   "block": [{"raw": b["raw"], "prediction": b["raw"].split("<|im_end|>")[0].strip(), "finish": b["finish"],
                              "tokens": b["tokens"]} for b in block]}
            async with lock:
                with open(out, "a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(r) for r in todo))
    await runner.client.close()


def write_predictions(args) -> None:
    natural = [json.loads(line) for line in open(args.out / "natural.jsonl")]
    branches = {r["index"]: r for r in map(json.loads, open(args.out / "branches.jsonl"))} \
        if (args.out / "branches.jsonl").exists() else {}
    missing = [r["index"] for r in natural if first_call_message(r["messages"]) is not None and r["index"] not in branches]
    if missing:
        raise SystemExit(f"{len(missing)} solutions with a tool call have no branches yet, e.g. {missing[:3]}")
    folder = args.out / "predictions"
    folder.mkdir(exist_ok=True)
    files = {"natural": [], **{f"{c}_r{k}": [] for c in ("execute", "block") for k in range(K)}}
    for r in natural:
        files["natural"].append({"index": r["index"], "prediction": r["prediction"], "source": "natural"})
        b = branches.get(r["index"])
        for c in ("execute", "block"):
            for k in range(K):
                files[f"{c}_r{k}"].append({"index": r["index"], "prediction": b[c][k]["prediction"], "source": "branch"} if b else
                                          {"index": r["index"], "prediction": r["prediction"], "source": "natural"})
    for name, rows in files.items():
        path = folder / f"{name}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(files)} prediction files of {len(natural)} problems ({len(branches)} branched) to {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural solutions, first-call branches and prediction files.")
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("natural", "branch"):
        p = sub.add_parser(name)
        if name == "natural":
            p.add_argument("--problems", required=True, type=Path)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--tokenizer", required=True, help="local snapshot of the served model (chat template, ids)")
        p.add_argument("--concurrency", type=int, default=16, help="requests in flight on the server")
        p.add_argument("--tool-copies", type=int, default=4)
        p.add_argument("--out", required=True, type=Path)
    sub.add_parser("predictions").add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()
    if args.step == "predictions":
        write_predictions(args)
    else:
        asyncio.run(run_natural(args) if args.step == "natural" else run_branch(args))


if __name__ == "__main__":
    main()
