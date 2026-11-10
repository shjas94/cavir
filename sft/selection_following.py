#!/usr/bin/env python3
"""Selection-following check for a cold-start v2 policy.

Run with the train_cluster env from /workspace/cavir/sft/LlamaFactory:
    /root/miniconda3/envs/train_cluster/bin/python /workspace/cavir/sft/selection_following.py \
        --prompts PROMPTS.jsonl --out-dir DIR

For each held-out prompt (``make_followcheck_prompts.py``) the policy writes its first turn.
If it opens ``<evaluating>`` that is a natural decision point; if it answers without one, its
think is kept and ``<evaluating>`` is forced open. The block is then filled by us, as the router
would: once per selection (answer_now, think_more, every dataset tool), with numbers sampled to
agree with the routing rule exactly as ``build_cavir_coldstart.render_evaluating`` does. The
policy continues and its next action is compared with the injected selection: a call to that
tool, an ``<answer>``, or a new ``<think>``. Tools are not executed; only the action is judged.
Prompts are encoded with LlamaFactory's training path, as in ``sample_inference.py``.
``--headers-from`` replaces each prompt's system prompt and tool list with those of the same dataset in
the given training file, so a fixed prompt set can test a policy trained with a different tool set.

Multi-node (data parallel, one GPU per process): launch with torchrun on every node (``scripts/run_followcheck_ddp.sh``).
Rank r takes the prompts whose index i in the full list has i % world_size == r (DistributedSampler without shuffling
or padding), every seed comes from that index, and rank 0 gathers all rows and writes the same files as a single run,
so the result does not depend on how the prompts are split. Each rank also streams its own rows to
``injections.rank<r>.jsonl`` in ``--out-dir`` on its node.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import os
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from sample_inference import CANONICAL_TOOL_ORDER

SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
MAX_THINK_TOKENS = 4096
SELECTION_RE = re.compile(r"^selection: (\S+)$", re.MULTILINE)
FUNCTION_RE = re.compile(r"<function=(\w+)>")


def render_evaluating(selection: str, tools: list[str], tau: float, rng: random.Random) -> str:
    """Same sampling as build_cavir_coldstart.render_evaluating (that module needs the data env)."""
    tau_c = round(tau * 100)
    if selection in tools:
        p = rng.randint(10, 85)
        best = rng.randint(5, min(40, 99 - p))
        deltas = {tool: rng.randint(max(-30, 1 - p), best - 1) for tool in tools}
        deltas[selection] = best
    else:
        p = rng.randint(tau_c, 97) if selection == "answer_now" else rng.randint(max(5, tau_c - 45), tau_c - 1)
        deltas = {tool: rng.randint(max(-30, 1 - p), -1) for tool in tools}
    lines = ["<evaluating>", f"answer_now: p={p / 100:.2f}"]
    lines += [f"{tool}: Δ={deltas[tool] / 100:+.2f}" for tool in tools]
    lines += [f"selection: {selection}", "</evaluating>"]
    return "\n".join(lines)


def classify(continuation: str) -> str:
    """The action that starts first: 'think_more', 'answer_now', a tool name, or 'none'."""
    found = {tag: continuation.find(tag) for tag in ("<think>", "<tool_call>", "<answer>") if tag in continuation}
    if not found:
        return "none"
    first = min(found, key=found.get)
    if first == "<think>":
        return "think_more"
    if first == "<answer>":
        return "answer_now"
    match = FUNCTION_RE.search(continuation, found["<tool_call>"])
    return match.group(1) if match else "tool_call_unparsed"


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


def summarize(rows: list[dict], prompts: list[dict]) -> dict:
    kind = lambda r: "tool" if r["target"] not in ("answer_now", "think_more") else r["target"]
    groups = defaultdict(list)
    for r in rows:
        groups[("all",)].append(r)
        groups[("type", kind(r))].append(r)
        groups[("tool", r["target"])].append(r) if kind(r) == "tool" else None
        groups[("dataset", r["dataset"])].append(r)
        groups[("point", r["point"])].append(r)
        groups[("point_type", r["point"], kind(r))].append(r)
    misses = Counter(f"{kind(r)} -> {r['action'] if r['action'] in ('answer_now', 'think_more', 'none') else 'tool'}"
                     for r in rows if not r["followed"])
    return {
        "prompts": len(prompts),
        "decision_points": dict(Counter(p["point"] for p in prompts)),
        "natural_selection": dict(Counter(p["natural_selection"] for p in prompts if p["point"] == "natural")),
        "follow": {" / ".join(k): rate(v) for k, v in sorted(groups.items())},
        "misses": dict(misses.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how often the policy follows an injected selection.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "LlamaFactory/saves/qwen35-2b-base/full/cavir_coldstart_v2"))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--headers-from", type=Path,
                        help="training jsonl whose per-dataset system/tools replace the prompts' headers")
    parser.add_argument("--limit-per-dataset", type=int, default=0, help="0 = all prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.7)
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        # gloo: only Python objects are gathered at the end; ranks may finish hours apart, hence the long timeout.
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    # Ranks on one node share --out-dir, so with several ranks only their own files must be new.
    args.out_dir.mkdir(parents=True, exist_ok=world > 1)
    if world > 1 and any((args.out_dir / name).exists() for name in ("summary.json", f"injections.rank{rank}.jsonl")):
        raise SystemExit(f"{args.out_dir} already has output")

    model_args, data_args, _, _ = get_infer_args(dict(
        model_name_or_path=args.model, template="qwen3_5", enable_thinking=True, image_max_pixels=262144,
        trust_remote_code=True, infer_backend="huggingface"))
    tok_mod = load_tokenizer(model_args)
    tokenizer, processor = tok_mod["tokenizer"], tok_mod["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).cuda().eval()

    headers = {}
    if args.headers_from:
        for line in open(args.headers_from):
            record = json.loads(line)
            headers.setdefault(record["meta"]["dataset"], set()).add((record["system"], record["tools"]))
        assert all(len(v) == 1 for v in headers.values()), "a dataset has more than one header"
        headers = {dataset: next(iter(v)) for dataset, v in headers.items()}

    prompts, per_dataset = [], Counter()
    for line in open(args.prompts):
        prompt = json.loads(line)
        if headers:
            prompt["system"], prompt["tools"] = headers[prompt["dataset"]]
        if args.limit_per_dataset and per_dataset[prompt["dataset"]] >= args.limit_per_dataset:
            continue
        per_dataset[prompt["dataset"]] += 1
        prompts.append(prompt)

    def generate(ids: list[int], mm_inputs: dict, max_new_tokens: int, stops: list[str], seed: int) -> str:
        inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                  **{k: v for k, v in mm_inputs.items() if torch.is_tensor(v)}}
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = model.generate(**{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=max_new_tokens,
                                 stop_strings=stops, tokenizer=tokenizer, **SAMPLING)
        return tokenizer.decode(out[0, len(ids):], skip_special_tokens=False).replace("<|im_end|>", "")

    rows, prompt_rows = [], []
    rows_out = open(args.out_dir / ("injections.jsonl" if world == 1 else f"injections.rank{rank}.jsonl"), "w")
    start = time.time()
    for index, prompt in enumerate(prompts):
        if index % world != rank:
            continue
        tools =[t for t in CANONICAL_TOOL_ORDER if t in {x["name"] for x in json.loads(prompt["tools"])}]
        messages = [{"role": "user", "content": prompt["conversations"][0]["value"]}, {"role": "assistant", "content": ""}]
        messages = template.mm_plugin.process_messages(messages, prompt["images"], [], [], processor)
        base_ids = template.encode_multiturn(tokenizer, messages, prompt["system"], prompt["tools"])[-1][0]
        mm_inputs = template.mm_plugin.get_mm_inputs(prompt["images"], [], [], [1], [0], [0], [base_ids], processor)
        seed = args.seed * 1_000_003 + index

        first = generate(base_ids, mm_inputs, MAX_THINK_TOKENS, ["</evaluating>", "</answer>"], seed)
        natural_selection = None
        if "<evaluating>" in first:
            point, prefix = "natural", first[:first.index("<evaluating>")]
            match = SELECTION_RE.search(first)
            natural_selection = match.group(1) if match else "incomplete"
        elif "</think>" in first:
            point, prefix = "forced", first[:first.index("</think>") + len("</think>")]
        else:
            point, prefix = "no_think", ""
        prefix = prefix.rstrip() + "\n\n" if prefix else ""
        prompt_row = {"index": index, "id": prompt["id"], "dataset": prompt["dataset"], "split": prompt["split"], "point": point,
                      "natural_selection": natural_selection, "first_turn": first}
        prompt_rows.append(prompt_row)
        if point == "no_think":
            continue

        rng = random.Random(seed)
        for target in ["answer_now", "think_more"] + tools:
            block = render_evaluating(target, tools, args.tau, rng)
            ids = base_ids + tokenizer.encode(prefix + block + "\n\n", add_special_tokens=False)
            continuation = generate(ids, mm_inputs, 256, ["</answer>", "<parameter=", "<think>", "</tool_call>"], seed)
            action = classify(continuation)
            row = {"index": index, "id": prompt["id"], "dataset": prompt["dataset"], "split": prompt["split"], "point": point,
                   "target": target, "action": action, "followed": action == target, "block": block,
                   "continuation": continuation[:400]}
            rows.append(row)
            rows_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_out.flush()
        done = [r for r in rows if r["id"] == prompt["id"]]
        print(f"[rank {rank}] " * (world > 1) + f"[{index + 1}/{len(prompts)} {time.time() - start:.0f}s] {prompt['dataset']} {point} "
              f"natural={natural_selection} followed {sum(r['followed'] for r in done)}/{len(done)} "
              f"misses {[(r['target'], r['action']) for r in done if not r['followed']]}", flush=True)
    rows_out.close()

    if world > 1:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object((prompt_rows, rows), gathered, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        prompt_rows = sorted((r for part, _ in gathered for r in part), key=lambda r: r["index"])
        rows = sorted((r for _, part in gathered for r in part), key=lambda r: r["index"])  # stable: target order kept
        with open(args.out_dir / "injections.jsonl", "w") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    with open(args.out_dir / "first_turns.jsonl", "w") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows, prompt_rows)
    summary.update({"model": args.model, "prompts_file": str(args.prompts),
                    "headers_from": str(args.headers_from) if args.headers_from else None, "seed": args.seed, "tau": args.tau,
                    "sampling": SAMPLING, "world_size": world, "seconds": round(time.time() - start)})
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("prompts", "decision_points", "natural_selection", "misses")}, ensure_ascii=False))
    for key in ("all", "type / answer_now", "type / think_more", "type / tool", "point / natural", "point / forced"):
        if key in summary["follow"]:
            print(key, summary["follow"][key])


if __name__ == "__main__":
    main()
