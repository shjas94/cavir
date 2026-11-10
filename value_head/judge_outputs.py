#!/usr/bin/env python3
"""A judge model rules on sampled tool outputs, 3 times each, to check the faithfulness rules.

The judge sees the question, the correct answer and the tool output, never the answer regions (with them it would
redo the rule's overlap computation instead of checking it). It is asked what the rule decides for each tool:
crop_ocr_image and crop_zoomin, whether the read text or the cropped image holds the information needed to answer;
draw_bbox and mark_dots, whether a drawn mark sits on the object or region the question asks about.

    /root/miniconda3/envs/research/bin/python value_head/judge_outputs.py --sample SAMPLE.jsonl \\
        --data 2B POINTS_R1 BRANCHES_R1 --data 2B POINTS_R2 BRANCHES_R2 --data 4B ... \\
        --base-url http://HOST:PORT/v1 --model JUDGE --out OUT.jsonl

SAMPLE: one line per output, {model, dp_id, action, k, ...}; the output is ``tool_calls[0]`` of that continuation.
An output that is a tool error is not sent to the judge: its line gets verdict null and reason "tool_error_output".
Run once per judge model. The API key is DIRECT_API_KEY from dataset_construction/.env (an exported DIRECT_API_KEY
wins). OUT keeps each sample line and adds judge_model, votes (1, 0 or null per run), verdict (the value of at least 2
of the 3 runs, else null), and per run the reply, its finish_reason and the length of its reasoning in characters, so a
cut-off reply can be told from a reply without a verdict line. Lines already in OUT are skipped, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / "dataset_construction" / ".env")

RUNS = 3
MAX_TOKENS = 8192   # reasoning models; finish_reason "length" in the output shows a reply that still ran out
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "extra_body": {"top_k": 20}}
SYSTEM = ("You check the output of a visual tool against a question about an image and its correct answer. "
          'Think it through, then end your reply with exactly one line: "verdict: yes" or "verdict: no".')
ASK = {
    "crop_ocr_image": "An OCR tool read the text below from one region of the image. Does this text hold the information "
                      "needed to answer the question?",
    "crop_zoomin": "The attached image is a region a crop tool cut from the image. Does this region hold the information "
                   "needed to answer the question?",
    "draw_bbox": "The attached image is the image with one or more boxes a tool drew on it. Does a drawn box sit on the "
                 "object or region that the question asks about?",
    "mark_dots": "The attached image is the image with one or more points a tool marked on it. Does a marked point sit on "
                 "the object or region that the question asks about?",
}
VERDICT_RE = re.compile(r"^verdict:\s*(yes|no)\s*$", re.MULTILINE | re.IGNORECASE)


def key(row: dict) -> str:
    return f"{row['model']}|{row['dp_id']}|{row['action']}|{row['k']}"


def load(data: list[list[str]], sample: list[dict]) -> dict[str, dict]:
    """Question, answer and forced tool output of each sampled continuation."""
    wanted = {key(r) for r in sample}
    found = {}
    for model, points_path, branches_path in data:
        points = {p["dp_id"]: p for p in map(json.loads, open(points_path))}
        for b in map(json.loads, open(branches_path)):
            k = f"{model}|{b['dp_id']}|{b['action']}|{b['k']}"
            if k in wanted:
                call = b["tool_calls"][0]
                text = re.sub(r"</?tool_response>", "", call["observation"] or "").strip()
                point = points[b["dp_id"]]
                found[k] = {"question": point["turns"][0].removeprefix("<image>").strip(), "answer": point["answer"],
                            "image": call["image"], "text": text, "error": call["error"]}
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} sampled outputs not in the branch files, e.g. {sorted(missing)[:3]}")
    return found


def message(action: str, item: dict) -> list[dict]:
    head = f"Question: {item['question']}\nCorrect answer: {item['answer']}\n\n{ASK[action]}"
    if action == "crop_ocr_image":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{head}\n\nOCR text:\n{item['text']}"}]
    url = "data:image/png;base64," + base64.b64encode(Path(item["image"]).read_bytes()).decode()
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": head}, {"type": "image_url", "image_url": {"url": url}}]}]


async def run(args) -> None:
    sample = [json.loads(line) for line in open(args.sample)]
    done = {key(json.loads(line)) for line in open(args.out)} if args.out.exists() else set()
    items = load(args.data, sample)
    todo = [r for r in sample if key(r) not in done]
    print(f"{len(todo)} outputs to judge with {args.model} ({len(done)} done), {RUNS} runs each", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ["DIRECT_API_KEY"], timeout=3600)
    semaphore, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    with open(args.out, "a") as handle:
        async def ask(row, n):
            async with semaphore:
                reply = await client.chat.completions.create(
                    model=args.model, messages=message(row["action"], items[key(row)]), max_tokens=MAX_TOKENS,
                    seed=int(hashlib.sha256(f"{key(row)}|{n}".encode()).hexdigest()[:8], 16), **SAMPLING)
            choice = reply.choices[0]
            thinking = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
            return {"reply": choice.message.content or "", "finish_reason": choice.finish_reason, "reasoning_chars": len(thinking)}

        async def one(row):
            item = items[key(row)]
            out = row | {"judge_model": args.model, "question": item["question"], "answer": item["answer"],
                         "output": item["image"] or item["text"]}
            if item["error"]:
                out |= {"votes": [], "verdict": None, "reason": "tool_error_output", "runs": []}
            else:
                runs = await asyncio.gather(*(ask(row, n) for n in range(RUNS)))
                votes = [(1 if v[-1].lower() == "yes" else 0) if (v := VERDICT_RE.findall(r["reply"])) else None for r in runs]
                counted = [v for v in votes if v is not None]
                out |= {"votes": votes, "verdict": next((v for v in (1, 0) if counted.count(v) >= 2), None), "runs": runs}
            async with lock:
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                handle.flush()
        try:
            await asyncio.gather(*(one(r) for r in todo))
        finally:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge sampled tool outputs 3 times with one judge model.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--data", nargs=3, action="append", required=True, metavar=("MODEL", "POINTS", "BRANCHES"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
