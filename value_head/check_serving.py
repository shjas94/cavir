#!/usr/bin/env python3
"""Stage-2 step 0 checks before any large-scale generation.

parity (train_cluster env, no server): ``prompting.render`` must produce the same tokens as LLaMA
Factory's ``qwen3_5`` ``encode_multiturn`` on cold start training records. LLaMA Factory expands each
image to its token count; runs of ``<|image_pad|>`` are collapsed to one before comparing, and the
counts are reported for the server-side check.

    /root/miniconda3/envs/train_cluster/bin/python value_head/check_serving.py parity --out OUT.json

server (research env, policy vLLM server and teacher server up): against the served policy,
  1. prompt token counts equal LLaMA Factory's (images included), using ``parity --out`` results;
  2. the policy behaves as under HF generate: first turns on two pilot-set prompts per dataset open
     ``<evaluating>``, and every injected selection is followed (target >= 95%);
  3. prefix caching is active (prefix-cache hit counters in ``/metrics`` grow over an n=8 request);
  4. throughput (generated tokens per second) at the given concurrency;
  5. the tool environment: crop_zoomin on index 0 reads the full-resolution file, crop_ocr_image reaches
     the teacher.

    /root/miniconda3/envs/research/bin/python value_head/check_serving.py server \
        --base-url http://HOST:PORT/v1 --model cavir-policy --parity PARITY.json --out OUT.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompting import render  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRAIN_FILE = REPO / "sft/LlamaFactory/data/cavir_coldstart_v2_noarith/cavir_coldstart_v2_noarith.jsonl"
POLICY = REPO / "sft/LlamaFactory/saves/qwen35-2b/full/cavir_coldstart_v2_noarith_fixtools"
PILOT_FILE = REPO / "sft/LlamaFactory/data/cavir_pilot_v2/cavir_pilot_v2.jsonl"
BEHAVIOR_PER_DATASET = 2


def collapse(ids: list[int], pad: int) -> list[int]:
    return [t for i, t in enumerate(ids) if not (t == pad and i and ids[i - 1] == pad)]


def parity(args: argparse.Namespace) -> None:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, _, _ = get_infer_args(dict(
        model_name_or_path=str(args.model), template="qwen3_5", enable_thinking=True, image_max_pixels=262144,
        trust_remote_code=True, infer_backend="huggingface"))
    tok_mod = load_tokenizer(model_args)
    tokenizer, processor = tok_mod["tokenizer"], tok_mod["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    by_dataset = defaultdict(list)
    for line in open(args.data):
        record = json.loads(line)
        by_dataset[record["meta"]["dataset"]].append(record)
    rng = random.Random(args.seed)
    records = [r for rows in by_dataset.values() for r in rng.sample(rows, min(args.per_dataset, len(rows)))]

    results, failures = Counter(), []
    pad_counts = []
    for record in records:
        messages = [{"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]}
                    for m in record["conversations"]]
        processed = template.mm_plugin.process_messages(messages, record["images"], [], [], processor)
        reference = [t for src, tgt in template.encode_multiturn(tokenizer, processed, record["system"], record["tools"])
                     for t in src + tgt]
        # Prompt for the last assistant turn, as generation sees it (fresh reply after the last user turn).
        prompt_pairs = template.encode_multiturn(tokenizer, processed[:-1] + [{"role": "assistant", "content": ""}],
                                                 record["system"], record["tools"])
        prompt_ref = [t for src, tgt in prompt_pairs[:-1] for t in src + tgt] + prompt_pairs[-1][0]

        turns = [m["value"] for m in record["conversations"]]
        full = tokenizer(render(record["system"], record["tools"], turns, partial=None), add_special_tokens=False)["input_ids"]
        prompt = tokenizer(render(record["system"], record["tools"], turns[:-1]), add_special_tokens=False)["input_ids"]

        ok_full = collapse(reference, pad) == full
        ok_prompt = collapse(prompt_ref, pad) == prompt
        results["full_match"] += ok_full
        results["prompt_match"] += ok_prompt
        results["records"] += 1
        runs = [n for n in _pad_runs(reference, pad)]
        pad_counts.append({"id": record["id"], "dataset": record["meta"]["dataset"], "image_pad_runs": runs,
                           "prompt_tokens": len(prompt_ref)})
        if not (ok_full and ok_prompt):
            failures.append({"id": record["id"], "dataset": record["meta"]["dataset"], "full": ok_full,
                             "prompt": ok_prompt, "first_diff": _first_diff(collapse(reference, pad), full, tokenizer)})

    report = {"records": results["records"], "full_match": results["full_match"], "prompt_match": results["prompt_match"],
              "by_dataset": dict(Counter(r["meta"]["dataset"] for r in records)), "failures": failures[:20],
              "image_pad_counts": pad_counts, "model": str(args.model), "data": str(args.data), "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("records", "full_match", "prompt_match")}), "failures:", len(failures))
    for failure in failures[:3]:
        print(json.dumps(failure, ensure_ascii=False)[:600])


def _pad_runs(ids: list[int], pad: int) -> list[int]:
    runs, count = [], 0
    for t in ids + [None]:
        if t == pad:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    return runs


def _first_diff(a: list[int], b: list[int], tokenizer) -> dict:
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"index": index, "len_ref": len(a), "len_render": len(b),
            "ref": tokenizer.decode(a[max(0, index - 20):index + 20]), "render": tokenizer.decode(b[max(0, index - 20):index + 20])}


async def _server(args: argparse.Namespace) -> dict:
    import time

    import httpx
    from PIL import Image

    from policy import Policy, request_content  # noqa: F401
    from prompting import first_action, parse_tool_call
    from tool_env import ToolEnv

    sys.path.insert(0, str(REPO / "dataset_construction"))
    from build_cavir_coldstart import CANONICAL_TOOL_ORDER, render_evaluating

    policy = Policy(args.base_url, args.model, concurrency=args.concurrency)
    report = {}

    # 1. prompt tokens
    expected = {row["id"]: row["prompt_tokens"] for row in json.load(open(args.parity))["image_pad_counts"]}
    records = [json.loads(line) for line in open(TRAIN_FILE) if json.loads(line)["id"] in expected]

    async def count(record):
        turns = [m["value"] for m in record["conversations"]][:-1]
        state = {"system": record["system"], "tools": record["tools"], "turns": turns,
                 "images": record["images"], "partial": ""}
        text = __import__("prompting").render(state["system"], state["tools"], turns, "")
        response = await policy.client.chat.completions.create(
            model=policy.model, max_tokens=1, messages=[{"role": "user", "content": request_content(text, state["images"])}],
            extra_body={"chat_template": __import__("policy").PASSTHROUGH_TEMPLATE, "chat_template_content_format": "openai",
                        "add_generation_prompt": False, "add_special_tokens": False})
        return record["id"], response.usage.prompt_tokens, expected[record["id"]]
    counts = await asyncio.gather(*(count(r) for r in records))
    mismatches = [{"id": i, "server": a, "expected": b} for i, a, b in counts if a != b]
    report["prompt_tokens"] = {"records": len(counts), "match": len(counts) - len(mismatches), "mismatches": mismatches[:10]}
    print("1 prompt tokens:", report["prompt_tokens"]["match"], "/", len(counts))

    # 2. behavior on pilot-set prompts (noarith headers)
    headers = {}
    for line in open(TRAIN_FILE):
        record = json.loads(line)
        headers.setdefault(record["meta"]["dataset"], (record["system"], record["tools"]))
    prompts, per = [], Counter()
    for line in open(PILOT_FILE):
        row = json.loads(line)
        if per[row["dataset"]] < BEHAVIOR_PER_DATASET:
            per[row["dataset"]] += 1
            prompts.append(row)

    async def behave(row):
        system, tools = headers[row["dataset"]]
        state = {"system": system, "tools": tools, "turns": [row["conversations"][0]["value"]],
                 "images": row["images"], "partial": ""}
        first = (await policy.generate(state, stop=["<evaluating>", "</answer>", "</tool_call>"], seed=args.seed))[0]
        if not first["text"].endswith("<evaluating>"):
            return row["dataset"], "no_evaluating", []
        prefix = first["text"][: -len("<evaluating>")].rstrip()
        names = [t for t in CANONICAL_TOOL_ORDER if t in {x["name"] for x in json.loads(tools)}]
        rng = random.Random(f"{args.seed}:{row['id']}")
        outcomes = []
        for target in ["answer_now", "think_more"] + names:
            block = render_evaluating(target, names, 0.7, rng)
            probe = dict(state, partial=f"{prefix}\n\n{block}\n\n")
            text = (await policy.generate(probe, stop=["</answer>", "<parameter=", "</tool_call>"], max_tokens=256,
                                          seed=args.seed))[0]["text"]
            action = first_action(text)
            if action == "tool_call":
                call = parse_tool_call(text + "</function>\n</tool_call>") if "<function=" in text else None
                action = call["name"] if call else text[text.find("<function=") + 10:].split(">")[0]
            outcomes.append({"target": target, "action": action, "followed": action == target})
        return row["dataset"], "natural", outcomes
    behavior = await asyncio.gather(*(behave(r) for r in prompts))
    injections = [o for _, _, outs in behavior for o in outs]
    report["behavior"] = {"prompts": len(behavior), "natural": sum(p == "natural" for _, p, _ in behavior),
                          "injections": len(injections), "followed": sum(o["followed"] for o in injections),
                          "misses": [(d, o["target"], o["action"]) for d, _, outs in behavior for o in outs if not o["followed"]]}
    print("2 behavior:", {k: v for k, v in report["behavior"].items() if k != "misses"}, report["behavior"]["misses"][:5])

    # 3. prefix caching
    metrics_url = args.base_url.rsplit("/v1", 1)[0] + "/metrics"

    def hits(text):
        return sum(float(line.split()[-1]) for line in text.splitlines()
                   if not line.startswith("#") and "prefix_cache_hits" in line)
    async with httpx.AsyncClient() as http:
        before = hits((await http.get(metrics_url)).text)
        row = prompts[0]
        system, tools = headers[row["dataset"]]
        state = {"system": system, "tools": tools, "turns": [row["conversations"][0]["value"]], "images": row["images"], "partial": ""}
        await policy.generate(state, n=8, stop=["</answer>"], max_tokens=64)
        await policy.generate(state, n=8, stop=["</answer>"], max_tokens=64)
        after = hits((await http.get(metrics_url)).text)
    report["prefix_cache"] = {"hits_before": before, "hits_after": after, "active": after > before}
    print("3 prefix cache:", report["prefix_cache"])

    # 4. throughput
    start = time.time()
    outs = await asyncio.gather(*(policy.generate(dict(state, partial=""), stop=["</answer>"], max_tokens=512, seed=i)
                                  for i in range(args.concurrency)))
    elapsed = time.time() - start
    generated = sum(o[0]["tokens"] or 0 for o in outs)
    report["throughput"] = {"concurrency": args.concurrency, "generated_tokens": generated, "seconds": round(elapsed, 1),
                            "tokens_per_second": round(generated / elapsed, 1)}
    print("4 throughput:", report["throughput"])

    # 5. tool environment
    doc = next(r for r in prompts if r["dataset"] == "docvqa")
    work = args.out.parent / "check_tool_work"
    async with ToolEnv(["crop_zoomin", "crop_ocr_image"]) as env:
        images = [doc["images"][0]]
        crop = await env.run({"name": "crop_zoomin", "arguments": {"image_index": 0, "crop_box": [0, 0, 500, 500], "expansion_factor": 2}}, images, work)
        if args.skip_teacher:
            ocr = {"error": "skipped (teacher not served)", "observation": ""}
        else:
            ocr = await env.run({"name": "crop_ocr_image", "arguments": {"image_index": 0, "crop_box": [0, 0, 1000, 300]}}, images, work)
    with Image.open(doc["images"][0]) as original:
        original_size = original.size
    crop_size = Image.open(crop["image"]).size if crop["image"] else None
    report["tools"] = {"original_size": original_size, "crop_zoomin_output_size": crop_size,
                       "crop_zoomin_output_pixels": crop_size[0] * crop_size[1] if crop_size else None,
                       "crop_error": crop["error"], "ocr_error": ocr["error"], "ocr_observation": ocr["observation"][:300]}
    print("5 tools:", report["tools"])
    return report


def server(args: argparse.Namespace) -> None:
    import asyncio as _asyncio
    report = _asyncio.run(_server(args))
    report.update({"base_url": args.base_url, "model": args.model})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--data", type=Path, default=TRAIN_FILE)
    p.add_argument("--model", type=Path, default=POLICY)
    p.add_argument("--per-dataset", type=int, default=23)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("server")
    q.add_argument("--base-url", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--parity", type=Path, required=True, help="parity --out json with prompt_tokens")
    q.add_argument("--concurrency", type=int, default=32)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--skip-teacher", action="store_true", help="skip the crop_ocr_image check (teacher not served)")
    q.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    {"parity": parity, "server": server}[args.command](args)


if __name__ == "__main__":
    main()
