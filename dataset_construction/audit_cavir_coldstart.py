#!/usr/bin/env python3
r"""Audit cavir_formatted.jsonl records for structural and logical defects.

Reads ``<dataset_dir>/cavir_formatted.jsonl`` (never modified) and writes
``<dataset_dir>/cavir_coldstart/audit.jsonl`` with the issues found per record,
plus ``audit_report.json`` with counts. Two passes:

* rules (default): cheap deterministic checks over every record.
* ``--llm N``: an LLM reads N records (0 = all) and reports logical
  inconsistencies the rules cannot see. Text only: it checks internal
  consistency, not whether a claim about the image is true.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_cavir_coldstart import TOOL_NAME_RE, parse_json_object, parse_tool_call_text
from reformat_dataset import ANSWER_TAG_RE, extract_boxed

GPT_THINK_RE = re.compile(r"<think>\n(.*?)\n</think>", re.DOTALL)
EVAL_RE = re.compile(r"<evaluating>\n(.*?)\n</evaluating>", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
ARITHMETIC = {"add", "sub", "multiply", "divide", "round"}

# Evidence that only exists after a tool ran (past tense / result nouns), not plans to run one.
TOOL_EVIDENCE_RE = re.compile(
    r"\b(?:the|this|that) (?:cropped (?:image|region|view)|crop|zoomed(?:-in)? (?:image|view|crop|region)|"
    r"enlarged (?:crop|image|view|region)|OCR (?:result|output|text)|tool (?:result|output))"
    r"[^.,;?]{0,15}\b(?:shows|showed|reveals|revealed|confirms|confirmed|reads|displays|displayed|indicates|indicated)\b",
    re.IGNORECASE,
)
# Reasoning or plans leaking into the short statement that precedes <answer>.
STATEMENT_REASONING_RE = re.compile(r"\b(?:let me|I need to|I will|I'll|step \d|first,|next,)\b", re.IGNORECASE)
SELF_CORRECTION_RE = re.compile(r"\b(?:wait|actually|hmm+|oops|correction|I mean|no, )\b", re.IGNORECASE)
ACTION_LEAK_RE = re.compile(
    r"\b(?:let me (?:zoom|crop|use|call|run|compute|calculate|verify with|check with|ocr)|"
    r"I(?:'ll| will) (?:use|call|zoom|crop|run|compute)|no (?:need for|tool)|answer directly|"
    r"without (?:a |any )?tool|use (?:the )?(?:crop_\w+|ocr|zoom|tool|calculator))",
    re.IGNORECASE,
)
# The templated conclusion sentence edit_think writes; a body that is only this sentence carries no reasoning.
CONCLUSION_RE = re.compile(r"Current evidence (?:points to|suggests)[^\n]*?(?:\.(?=\s|$)|$)")
# A tool call written out as text in the answer statement instead of being made.
TOOL_TEXT_RE = re.compile(r"\b(?:crop_zoomin|crop_ocr_image|draw_bbox|mark_dots|imaginate_editing|visualize_data|math_compute|"
                          r"tool_call)\b|^\[[^\]]*\]$", re.IGNORECASE)
# Before any tool result: a plan to run a perception tool, or a claim about a crop that never ran.
TOOL_INTENT_RE = re.compile(
    r"^<?(?:crop|zoom|ocr)\b|\b(?:I should|I'd like to|I would like to|I need to|let's|let me|we should|I will|I'll)\s+"
    r"(?:\w+\s+){0,3}?(?:zoom|crop|ocr)\b|\bzooming in(?:to)?\b|\bcrop(?:ping)? (?:and|to) |\bcrop zoom\b|"
    r"\b(?:cropped|zoomed(?:-in)?|enlarged) (?:view|image|region|crop)\b",
    re.IGNORECASE,
)
TOOL_NEGATION_RE = re.compile(r"\bno (?:further |additional )?(?:zooming|cropping|zoom|crop|tool)|\bwithout (?:zooming|cropping|a tool)|"
                              r"\bnot (?:needed|necessary)\b", re.IGNORECASE)


def _number(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _route(evaluating: str, tau: float) -> tuple[str, str]:
    """(selection written in the block, selection the routing rule implies)."""
    p = float(re.search(r"answer_now: p=([\d.]+)", evaluating).group(1))
    deltas = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+): Δ=([+-][\d.]+)$", evaluating, re.MULTILINE)}
    written = re.search(r"^selection: (\S+)$", evaluating, re.MULTILINE).group(1)
    best = max(deltas, key=deltas.get)
    implied = best if deltas[best] > 0 else ("answer_now" if p >= tau else "think_more")
    return written, implied


def rule_issues(record: dict[str, Any], tau: float, max_think_more: int) -> list[dict[str, str]]:
    """Deterministic checks; each issue is {"type", "detail"}."""
    issues: list[dict[str, str]] = []

    def add(kind: str, detail: str) -> None:
        issues.append({"type": kind, "detail": detail[:300]})

    tools = {tool["name"] for tool in json.loads(record["tools"])}
    conversations = record["conversations"]
    placeholders = sum(m["value"].count("<image>") for m in conversations)
    if placeholders != len(record["images"]):
        add("image_count_mismatch", f"{placeholders} placeholders, {len(record['images'])} images")

    images_so_far = 1
    seen_tool_result = False
    for index, message in enumerate(conversations):
        value = message["value"]
        if message["from"] == "human":
            if "<tool_response>" in value:
                seen_tool_result = True
                images_so_far += value.count("<image>")
            continue

        thinks = GPT_THINK_RE.findall(value)
        evaluations = EVAL_RE.findall(value)
        action = value.split("</evaluating>")[-1] if evaluations else value.split("</think>")[-1]
        call = parse_tool_call_text(action)
        tags = ANSWER_TAG_RE.findall(action)

        # --- structure: selections vs routing rule vs the action that follows
        if evaluations:
            selections = []
            for block in evaluations:
                written, implied = _route(block, tau)
                selections.append(written)
                if written != implied:
                    add("routing_rule_violation", f"turn {index}: selection {written}, rule implies {implied}")
            if len(thinks) != len(evaluations):
                add("think_eval_count_mismatch", f"turn {index}: {len(thinks)} thinks, {len(evaluations)} evaluating")
            if any(s != "think_more" for s in selections[:-1]):
                add("non_final_selection_not_think_more", f"turn {index}: {selections}")
            if selections.count("think_more") > max_think_more:
                add("too_many_think_more", f"turn {index}: {selections.count('think_more')}")
            final = selections[-1]
            if final == "answer_now" and not tags:
                add("selection_action_mismatch", f"turn {index}: answer_now without <answer>")
            elif final in tools and (call is None or call["name"] != final):
                add("selection_action_mismatch", f"turn {index}: selection {final}, call {call and call['name']}")
            elif final == "think_more":
                add("selection_action_mismatch", f"turn {index}: message ends on think_more")

        # --- tool call validity
        if call is not None:
            if call["name"] not in tools:
                add("tool_not_in_dataset_set", call["name"])
            index_arg = call["arguments"].get("image_index")
            if isinstance(index_arg, int) and not 0 <= index_arg < images_so_far:
                add("image_index_out_of_range", f"{call['name']} image_index={index_arg}, {images_so_far} images so far")
            if call["name"] in ARITHMETIC:
                a, b = (_number(str(call["arguments"].get(k, ""))) for k in ("a", "b"))
                trivial = (call["name"] in ("add", "sub") and b == 0) or (call["name"] == "add" and a == 0) or \
                          (call["name"] in ("multiply", "divide") and b == 1) or (call["name"] == "multiply" and a == 1) or \
                          (call["name"] == "round" and a is not None and a == a.to_integral_value())
                if trivial:
                    add("trivial_arithmetic", f"{call['name']}({a}, {b})")

        # --- think content
        for think in thinks:
            body = think.split("Still missing:")[0].strip() if evaluations else think
            if evaluations and len(body) < 20:
                add("empty_think_body", f"turn {index}")
            elif evaluations and len(CONCLUSION_RE.sub("", body).strip()) < 20:
                add("conclusion_only_think_body", f"turn {index}: {body}")
            if not seen_tool_result and TOOL_EVIDENCE_RE.search(think):
                add("tool_evidence_without_tool", f"turn {index}: {TOOL_EVIDENCE_RE.search(think).group(0)}")
            if evaluations and ACTION_LEAK_RE.search(body):
                add("action_leak_in_edited_think", f"turn {index}: {ACTION_LEAK_RE.search(body).group(0)}")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 15]
            repeated = [s for s, n in Counter(sentences).items() if n > 1]
            if repeated:
                add("repeated_sentence", f"turn {index}: {repeated[0]}")
            missing = think.split("Still missing:")[1].split(" Options:")[0] if "Still missing:" in think else ""
            if missing and TOOL_NAME_RE.search(missing):
                add("still_missing_names_tool", missing)
            if evaluations and think.count("Still missing:") > 1:
                add("teacher_deliberates_on_options", f"turn {index}: {think.count('Still missing:')} status sentences")

        # --- final answer statement
        if tags:
            statement = action[: action.lower().find("<answer")].strip()
            boxed = extract_boxed(tags[-1]) or ""
            if statement and SELF_CORRECTION_RE.search(statement):
                add("self_correction_in_answer_statement", statement)
            if statement and (STATEMENT_REASONING_RE.search(statement) or len(statement) > 600):
                add("reasoning_in_answer_statement", statement)
            if statement and (TOOL_TEXT_RE.search(statement) or (not seen_tool_result and TOOL_INTENT_RE.search(statement)
                                                                 and not TOOL_NEGATION_RE.search(statement))):
                add("tool_imitation_in_answer_statement", statement)
            answer_number = _number(boxed) if NUMBER_RE.fullmatch(boxed.strip()) else None
            if statement and answer_number is not None:
                numbers = {_number(n) for n in NUMBER_RE.findall(statement)}
                if numbers and answer_number not in numbers:  # weak signal: statements may cite only the inputs
                    add("statement_answer_number_mismatch", f"answer {boxed}; statement numbers {sorted(map(str, numbers))[:6]}")
    return issues


LLM_PROMPT = """You audit one training trajectory of a visual reasoning agent for LOGICAL defects. You cannot see the image, so judge only internal consistency of the text, not whether claims about the image are true.

The trajectory format is fixed and intended: each <think> is followed by an <evaluating> block whose numbers are synthetic and must be ignored; every think ends with a templated "Still missing: … Options: …" sentence; selections may deliberately differ from what the reasoning would suggest. Do NOT report any of that.

Report only these defects:
- "contradiction": two statements in the trajectory contradict each other (including think vs final answer).
- "missing_contradicts_evidence": a "Still missing:" sentence asks for something the reasoning has already established.
- "unsupported_reference": the reasoning cites evidence that does not exist in the trajectory, e.g. a crop, zoom or tool result when no such tool output appears before it.
- "ignores_tool_result": reasoning after a tool result misreports it or ignores it while relying on the opposite.
- "pointless_tool_call": a tool call whose arguments make it useless for the question (e.g. multiplying unrelated constants, 42 - 0).
- "answer_statement_defect": the sentence before <answer> contradicts the answer or contains visible self-correction.
- "incoherent_reasoning": a think that is empty of reasoning, truncated mid-thought, or not about the question.

Reply with only a JSON object: {"issues": [{"type": "<one of the types above>", "turn": <gpt message number starting at 1>, "evidence": "<short quote>"}]}. Use an empty list when there is no defect."""


async def llm_issues(client, model, record, semaphore) -> list[dict[str, Any]] | None:
    text = []
    gpt_number = 0
    for message in record["conversations"]:
        if message["from"] == "gpt":
            gpt_number += 1
            text.append(f"[gpt message {gpt_number}]\n{message['value']}")
        else:
            text.append(f"[human]\n{message['value']}")
    user = "\n\n".join(text)
    for _ in range(3):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": user}],
                    max_tokens=8192, temperature=0.6, top_p=0.95,
                    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
                )
            obj = parse_json_object(response.choices[0].message.content or "")
            if isinstance(obj.get("issues"), list):
                return [i for i in obj["issues"] if isinstance(i, dict) and "type" in i]
        except Exception:  # retry on transport or parse failure
            continue
    return None


async def main_async(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    records = [json.loads(line) for line in open(dataset_dir / "cavir_formatted.jsonl")]
    aux = dataset_dir / "cavir_coldstart"
    results = {r["id"] + "|" + r["meta"]["variant"]: {"id": r["id"], "variant": r["meta"]["variant"],
                                                      "case": r["meta"]["case"], "rule_issues": rule_issues(r, args.tau, 2)}
               for r in records}

    if args.llm is not None:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        client = AsyncOpenAI(base_url=os.environ["DIRECT_BASE_URL"], api_key=os.getenv("DIRECT_API_KEY", "EMPTY"), timeout=1800)
        sample = records if args.llm == 0 else random.Random(args.seed).sample(records, min(args.llm, len(records)))
        semaphore = asyncio.Semaphore(args.concurrency)
        found = await asyncio.gather(*(llm_issues(client, os.environ["MODEL_NAME"], r, semaphore) for r in sample))
        for record, issues in zip(sample, found):
            results[record["id"] + "|" + record["meta"]["variant"]]["llm_issues"] = issues
        await client.close()

    suffix = "" if args.llm in (None, 0) else f"_llm{args.llm}"
    with open(aux / f"audit{suffix}.jsonl", "w") as handle:
        for row in results.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = list(results.values())
    rule_types = Counter(i["type"] for row in rows for i in row["rule_issues"])
    llm_rows = [row for row in rows if "llm_issues" in row]
    llm_types = Counter(i["type"] for row in llm_rows for i in (row["llm_issues"] or []))
    report = {
        "dataset": dataset_dir.name, "records": len(rows),
        "records_with_rule_issue": sum(bool(row["rule_issues"]) for row in rows),
        "rule_issue_records_by_type": {k: sum(any(i["type"] == k for i in row["rule_issues"]) for row in rows) for k in rule_types},
        "rule_issue_records_by_variant": dict(Counter(row["variant"] for row in rows if row["rule_issues"])),
    }
    if llm_rows:
        report.update({
            "llm_audited": len(llm_rows),
            "llm_failed": sum(row["llm_issues"] is None for row in llm_rows),
            "records_with_llm_issue": sum(bool(row["llm_issues"]) for row in llm_rows),
            "llm_issue_records_by_type": {k: sum(any(i["type"] == k for i in (row["llm_issues"] or [])) for row in llm_rows) for k in llm_types},
            "llm_issue_records_by_variant": dict(Counter(row["variant"] for row in llm_rows if row["llm_issues"])),
        })
    (aux / f"audit_report{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cavir_formatted.jsonl for structural and logical defects.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--llm", type=int, default=None, help="LLM-audit N random records (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
