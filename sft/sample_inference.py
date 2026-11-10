#!/usr/bin/env python3
"""Sanity-check a cold-start v2 SFT model on a few training records with Hugging Face generate.

Run with the train_cluster env from /workspace/cavir/sft/LlamaFactory:
    /root/miniconda3/envs/train_cluster/bin/python /workspace/cavir/sft/sample_inference.py --out-dir DIR

Prompts are encoded exactly as in training: LlamaFactory's ``qwen3_5`` template with
``encode_multiturn`` (earlier turns keep their <think>, unlike ``encode_oneturn``) and its
image preprocessing (``image_max_pixels`` 262,144). The model is plain transformers with
FlashAttention-2. For every assistant turn of a sampled record, the earlier turns come from
the record (teacher forcing) and only that turn is generated, then checked for format,
the routing rule, selection/action agreement and the final answer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

THINK_RE = re.compile(r"<think>\n(.*?)\n</think>", re.DOTALL)
EVAL_RE = re.compile(r"<evaluating>\n(.*?)\n</evaluating>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{(.*)\}")
CALL_RE = re.compile(r"<tool_call>\s*<function=([\w]+)>")
# Order of tool lines in <evaluating> blocks (build_cavir_coldstart.CANONICAL_TOOL_ORDER; that module
# needs the data-construction env, so the tuple is repeated here).
CANONICAL_TOOL_ORDER = (
    "crop_zoomin", "crop_ocr_image", "draw_bbox", "mark_dots", "imaginate_editing",
    "visualize_data", "math_compute", "add", "sub", "multiply", "divide", "round",
)
# One record per Case, with different source variants; Case 2 gets a useful and an override call.
VARIANTS = ["r0_text", "r1_tool", "r0_tool_override", "r2_answer_now", "r3_think_more"]


def check(text: str, tools: list[str], tau: float, gt: str | None) -> dict:
    """Format and consistency checks for one generated assistant turn."""
    thinks, blocks = THINK_RE.findall(text), EVAL_RE.findall(text)
    result = {"thinks": len(thinks), "evaluating_blocks": len(blocks), "issues": []}
    selections = []
    for block in blocks:
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        listed = [k for k in lines if k not in ("answer_now", "selection")]
        if listed != tools:
            result["issues"].append(f"tool lines {listed} != dataset tools {tools}")
        try:
            p = float(lines["answer_now"].removeprefix("p="))
            deltas = {t: float(lines[t].removeprefix("Δ=")) for t in listed}
            best = max(deltas, key=deltas.get)
            implied = best if deltas[best] > 0 else ("answer_now" if p >= tau else "think_more")
            if lines.get("selection") != implied:
                result["issues"].append(f"selection {lines.get('selection')} but routing rule implies {implied}")
        except (KeyError, ValueError) as error:
            result["issues"].append(f"unparsable evaluating block ({error})")
        selections.append(lines.get("selection"))
    result["selections"] = selections
    action = text.split("</evaluating>")[-1] if blocks else text.split("</think>")[-1]
    call, boxed = CALL_RE.search(action), BOXED_RE.search(action)
    result["action"] = f"tool:{call.group(1)}" if call else ("answer" if boxed else "none")
    if selections:
        final = selections[-1]
        if any(s != "think_more" for s in selections[:-1]):
            result["issues"].append(f"non-final selection is not think_more: {selections}")
        expected = "answer" if final == "answer_now" else f"tool:{final}"
        if final == "think_more" or result["action"] != expected:
            result["issues"].append(f"final selection {final} but action {result['action']}")
    if boxed:
        result["answer"] = boxed.group(1).strip()
        result["correct"] = None if gt is None else result["answer"] == gt
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a few training-record turns with the SFT model.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "LlamaFactory/saves/qwen35-2b-base/full/cavir_coldstart_v2"))
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "dataset_construction/legacy/llamafactory_data/cavir_coldstart_v2/cavir_coldstart_v2.jsonl"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--tau", type=float, default=0.7)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    model_args, data_args, _, _ = get_infer_args(dict(
        model_name_or_path=args.model, template="qwen3_5", enable_thinking=True, image_max_pixels=262144,
        trust_remote_code=True, infer_backend="huggingface"))
    tok_mod = load_tokenizer(model_args)
    tokenizer, processor = tok_mod["tokenizer"], tok_mod["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).cuda().eval()
    print("attention:", model.config._attn_implementation, flush=True)

    records = [json.loads(line) for line in open(args.data)]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    picked, used_datasets = [], set()
    for variant in VARIANTS:
        record = next(r for r in records if r["meta"]["variant"] == variant and r["meta"]["dataset"] not in used_datasets)
        used_datasets.add(record["meta"]["dataset"])
        picked.append(record)

    out = open(args.out_dir / "generations.jsonl", "w")
    for record in picked:
        tools = {t["name"] for t in json.loads(record["tools"])}
        order = [t for t in CANONICAL_TOOL_ORDER if t in tools]
        messages = [{"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]}
                    for m in record["conversations"]]
        gt_match = BOXED_RE.search(record["conversations"][-1]["value"])
        gt = gt_match.group(1).strip() if gt_match else None
        # The training encoding of the whole record; every generation prompt must be a prefix of it.
        full_messages = template.mm_plugin.process_messages(messages, record["images"], [], [], processor)
        full_ids = [t for source, target in template.encode_multiturn(
            tokenizer, full_messages, record["system"], record["tools"]) for t in source + target]
        for k in range(1, len(messages), 2):
            history = messages[:k]
            n_images = sum(m["content"].count("<image>") for m in history)
            images = record["images"][:n_images]
            prompt_messages = template.mm_plugin.process_messages(
                history + [{"role": "assistant", "content": ""}], images, [], [], processor)
            pairs = template.encode_multiturn(tokenizer, prompt_messages, record["system"], record["tools"])
            prompt_ids = [t for source, target in pairs[:-1] for t in source + target] + pairs[-1][0]
            assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt differs from the training encoding"
            mm_inputs = template.mm_plugin.get_mm_inputs(images, [], [], [len(images)], [0], [0], [prompt_ids], processor)
            inputs = {"input_ids": torch.tensor([prompt_ids]), "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.long),
                      **{key: value for key, value in mm_inputs.items() if torch.is_tensor(value)}}
            inputs = {key: value.cuda() for key, value in inputs.items()}
            torch.manual_seed(args.seed)
            start = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                           temperature=0.6, top_p=0.95, top_k=20)
            new_ids = generated[0, len(prompt_ids):]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            row = {
                "id": record["id"], "dataset": record["meta"]["dataset"], "variant": record["meta"]["variant"],
                "case": record["meta"]["case"], "turn": (k + 1) // 2, "turns": len(messages) // 2,
                "prompt_tokens": len(prompt_ids), "new_tokens": len(new_ids), "seconds": round(time.time() - start, 1),
                "hit_max_tokens": len(new_ids) >= args.max_new_tokens,
                "check": check(text, order, args.tau, gt if k == len(messages) - 1 else None),
                "reference_check": check(messages[k]["content"], order, args.tau, gt if k == len(messages) - 1 else None),
                "generated": text, "reference": messages[k]["content"],
                "question": record["conversations"][0]["value"].replace("<image>", "").strip()[-300:], "gt": gt,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            c = row["check"]
            print(f"[{row['variant']} {row['dataset']} turn {row['turn']}/{row['turns']}] "
                  f"{row['new_tokens']} tok {row['seconds']}s | selections {c['selections']} action {c['action']} "
                  f"| ref {row['reference_check']['selections']} {row['reference_check']['action']} "
                  f"| answer {c.get('answer')} correct {c.get('correct')} | issues {c['issues']}", flush=True)
    out.close()
    write_report(args.out_dir)


def write_report(out_dir: Path) -> None:
    """Side-by-side markdown of every generated turn and its reference."""
    lines = ["# Generated vs reference turns", ""]
    for row in map(json.loads, open(out_dir / "generations.jsonl")):
        c, r = row["check"], row["reference_check"]
        lines += [f"## {row['variant']} · {row['dataset']} · Case {row['case']} · turn {row['turn']}/{row['turns']}", "",
                  f"- Question: {row['question']}", f"- Ground truth: `{row['gt']}`",
                  f"- Generated: selections {c['selections']}, action {c['action']}, answer `{c.get('answer')}`, "
                  f"correct {c.get('correct')}, issues {c['issues'] or 'none'}, {row['new_tokens']} tokens",
                  f"- Reference: selections {r['selections']}, action {r['action']}", "",
                  "<details><summary>generated</summary>", "", "```", row["generated"], "```", "</details>", "",
                  "<details><summary>reference</summary>", "", "```", row["reference"], "```", "</details>", ""]
    (out_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
