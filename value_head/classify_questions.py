#!/usr/bin/env python3
"""The teacher labels each problem's question as grounding or reasoning (no image, question and answer only).

grounding: the answer stands in one place of the image, so finding and reading or looking there answers it.
reasoning: the answer needs several values of the image gathered, computed, compared or counted.

    /root/miniconda3/envs/research/bin/python value_head/classify_questions.py --points P1.jsonl [--points P2.jsonl ...] \\
        --branches B1.jsonl [--branches B2.jsonl ...] --out OUT.jsonl

Problems are the ids in the (regraded) branch files, so excluded problems stay out; the question text comes from the points files. The teacher is the one in dataset_construction/.env
(DIRECT_BASE_URL, MODEL_NAME, DIRECT_API_KEY); sampling is the model's recommended
setting with a seed per problem. OUT has one line per problem: {id, dataset, question, answer, type, raw,
finish_reason, reasoning_chars}; ``type`` is null when the reply has no "type: grounding|reasoning" line, and
finish_reason "length" tells a cut-off reply from one without that line. Problems already in OUT are skipped, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / "dataset_construction" / ".env")

SYSTEM = """You sort visual questions into two types. You see only the question and its correct answer, not the image.
- grounding: the answer stands in one place of the image, so finding that place and reading or looking at it answers \
the question. Example: "What is the date on the form?"
- reasoning: the answer needs several values of the image gathered, or a computation, comparison or count. Example: \
"What is the difference between the highest and lowest values?"
End your reply with exactly one line: "type: grounding" or "type: reasoning"."""
TYPE_RE = re.compile(r"^type:\s*(grounding|reasoning)\s*$", re.MULTILINE | re.IGNORECASE)
MAX_TOKENS = 8192   # reasoning model; finish_reason "length" in the output shows a reply that still ran out
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20}}   # the teacher's generation_config


def problems(points: list[Path], branches: list[Path]) -> list[dict]:
    wanted = {json.loads(line)["id"] for path in branches for line in open(path)}
    seen = {}
    for path in points:
        for p in map(json.loads, open(path)):
            if p["id"] not in wanted:
                continue
            question = p["turns"][0].removeprefix("<image>").strip()
            seen.setdefault(p["id"], {"id": p["id"], "dataset": p["dataset"], "question": question, "answer": p["answer"]})
    return list(seen.values())


async def run(args) -> None:
    done = {json.loads(line)["id"] for line in open(args.out)} if args.out.exists() else set()
    todo = [p for p in problems(args.points, args.branches) if p["id"] not in done]
    print(f"{len(todo)} problems to classify ({len(done)} done)", flush=True)
    client = AsyncOpenAI(base_url=os.environ["DIRECT_BASE_URL"], api_key=os.environ["DIRECT_API_KEY"], timeout=3600)
    semaphore, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    with open(args.out, "a") as handle:
        async def one(p):
            async with semaphore:
                reply = await client.chat.completions.create(
                    model=os.environ["MODEL_NAME"], max_tokens=MAX_TOKENS, seed=int(hashlib.sha256(p["id"].encode()).hexdigest()[:8], 16),
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"Question: {p['question']}\nCorrect answer: {p['answer']}"}],
                    **SAMPLING)
            choice = reply.choices[0]
            raw = choice.message.content or ""
            thinking = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
            found = TYPE_RE.findall(raw)
            async with lock:
                handle.write(json.dumps(p | {"type": found[-1].lower() if found else None, "raw": raw,
                                             "finish_reason": choice.finish_reason, "reasoning_chars": len(thinking)},
                                        ensure_ascii=False) + "\n")
                handle.flush()
        try:
            await asyncio.gather(*(one(p) for p in todo))
        finally:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label questions grounding or reasoning with the teacher.")
    parser.add_argument("--points", action="append", required=True, type=Path)
    parser.add_argument("--branches", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
