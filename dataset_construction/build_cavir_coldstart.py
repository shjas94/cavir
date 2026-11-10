#!/usr/bin/env python3
r"""Build CaVIR cold-start v2 data: insert ``<evaluating>`` blocks into rollouts.

Reads ``<dataset_dir>/reformatted_results.jsonl`` (ShareGPT, already filtered
for correctness and format) and writes ``<dataset_dir>/cavir_formatted.jsonl``.
Every input trajectory passed ``filter_out.py``, so its final boxed answer is
the ground truth; no other file is read.

Pipeline (one teacher model, served over an OpenAI-compatible endpoint):

1. Classify every rollout. R0 has no tool call. Otherwise each tool call is
   judged (the judge sees the ground truth); the first call judged not useful
   decides R2 (answer was already settled) or R3 (doubt was resolvable without
   the tool); all calls useful gives R1.
2. Plan which rollout becomes which variant (``MIX``); a rollout is used once.
3. Build each variant. Decision-point thinks are edited without the ground
   truth, the selection or the original action; ``<evaluating>`` blocks are
   rendered from a template whose numbers are sampled to agree with the
   routing rule. Past the switch point the teacher regenerates the rest of the
   trajectory, tool calls are executed for real, and the result is kept only
   when its final answer matches the ground truth.
4. Write every survivor and report counts per variant, Case and selection
   type. Nothing is downsampled here: the target mix is decided once all
   datasets are merged.

Auxiliary files go to ``<dataset_dir>/cavir_coldstart/``: the judgment cache
(``judgments.jsonl``, reused on reruns), the report, and the images/scripts of
regenerated tool calls. ``--pilot N`` runs the whole pipeline on N random
rollouts, adds a leak probe, and writes
``pilot_formatted.jsonl``/``pilot_report.json`` there instead.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import sys
import uuid
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from dataset import _preprocess_image
from filter_out import (
    TOOL_ERROR_TEXT_RE,
    answers_match,
    normalize_answer,
)
from host import (
    CODE_EXECUTION_TOOLS,
    DIRECT_IMAGE_PRODUCING_TOOLS,
    GENERATION_PROMPT,
    IMAGE_INPUT_TOOLS,
    MODEL_BACKED_TOOLS,
    _code_tool_feedback,
    _file_b64,
    _fit_to_image_budget,
    _parse_tool_payload,
    _resolve_image_index,
    discover_tool_servers,
)
from reformat_dataset import (
    ANSWER_TAG_RE,
    DATASET_TOOL_NAMES,
    HOST_INJECTED_ARGUMENTS,
    SPECIAL_TAG_RE,
    _escape_observation,
    escape_image_placeholder,
    extract_boxed,
    render_tool_call,
    sanitize_reasoning,
)

logger = logging.getLogger("build_cavir_coldstart")

INPUT_FILENAME = "reformatted_results.jsonl"
OUTPUT_FILENAME = "cavir_formatted.jsonl"
AUX_DIRNAME = "cavir_coldstart"

# Order in which tools are listed in <evaluating> blocks and "Options:" sentences.
# The arithmetic tool names (add/sub/multiply/divide/round) are deliberately absent: listing them
# here would make TOOL_NAME_RE reject plain words such as "round" in Still missing sentences.
CANONICAL_TOOL_ORDER = ("crop_zoomin", "crop_ocr_image", "draw_bbox", "mark_dots", "math_compute")
# Tools that need numbers to work on; never chosen for an override when the
# context has fewer than two numbers.
NUMERIC_TOOLS = {"math_compute"}

MAX_THINK_MORE = 2          # per turn; the next decision is forced to answer_now
LLM_ATTEMPTS = 3            # retries of one LLM step whose output fails validation
GEN_MAX_TOKENS = 16384
JUDGE_MAX_TOKENS = 8192
EDIT_MAX_TOKENS = 4096

# variant -> (source class, case, share). A share only splits its source pool
# among that source's variants (e.g. R0 goes 15:5:5:3 to its four variants);
# it is not enforced on the output, whose mix is set after merging datasets.
MIX = {
    "r0_text":            ("R0", 1, 0.15),
    "r1_tool":            ("R1", 2, 0.30),
    "r0_tool_override":   ("R0", 2, 0.05),
    "r2_answer_now":      ("R2", 3, 0.17),
    "r0_answer_now":      ("R0", 3, 0.05),
    "r1_answer_override": ("R1", 3, 0.08),
    "r3_think_more":      ("R3", 4, 0.10),
    "r0_split":           ("R0", 4, 0.03),
    "r1_think_override":  ("R1", 4, 0.07),
}
REGENERATED = {"r0_tool_override", "r2_answer_now", "r1_answer_override", "r3_think_more", "r1_think_override"}

GEN_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}
JUDGE_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0}
EDIT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 0.0}

THINK_BLOCK_RE = re.compile(r"^\s*<think>\n?(.*?)\n?</think>\s*(.*)$", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)
EXTRA_TAG_RE = re.compile(r"</?(?:evaluating|action)\s*>", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Whole words only, so a tool name inside a longer word is not flagged.
TOOL_NAME_RE = re.compile(r"\b(?:" + "|".join(CANONICAL_TOOL_ORDER) + r")\b", re.IGNORECASE)
STATUS_SENTENCES_RE = re.compile(r"\s*Still missing:.*?Options:[^\n]*", re.DOTALL)


# --------------------------------------------------------------------------- #
#  Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = r"""You are a multimodal reasoning agent. Answer questions by combining visual reasoning, logical reasoning, and tools, and weigh whether each step is worth its cost.

Principles:
- Gather evidence before answering; do not guess when key information is missing.
- Call a tool only when its expected gain in correctness outweighs its cost.
- Always use a returned observation in your next reasoning step.
- Do not repeat a call that yields no new information.

Output format. Every response begins with a <think> block. No text may appear outside the allowed tags, except a short final statement before <answer>.

When you weigh your next step, close the <think> block with an <evaluating> block:
<evaluating>
answer_now: p=<probability that answering now is correct>
<tool_name>: Δ=<expected change of that probability if the tool is called>
(one line per available tool)
selection: <tool_name | answer_now | think_more>
</evaluating>
Routing: select the tool with the largest Δ if that Δ is positive; otherwise answer_now if p is high enough; otherwise think_more and open a new <think> block to keep reasoning. If a question is simple, you may answer directly without an <evaluating> block.

To call a tool:
<think>
...
</think>
<evaluating>
...
selection: tool_name
</evaluating>
<tool_call>
<function=tool_name>
<parameter=first_argument>
value
</parameter>
</function>
</tool_call>

To give the final answer:
<think>
...
</think>
<evaluating>
...
selection: answer_now
</evaluating>
<answer>
\boxed{...}
</answer>"""

JUDGE_PROMPT = """You audit one tool call inside a visual question answering trajectory. The trajectory reached the correct answer; you are told what it is.

You see the question, the correct answer, the input image, the reasoning right before the call, the call, its result, and the reasoning right after it. Decide whether THIS call was useful.

- useful: the result supplied information (a value, text, visual detail or computation) that the later reasoning actually relied on to reach the correct answer, and that the agent could not reliably get from what it already had.
- not useful, "confident_before": before the call the reasoning had already reached the correct answer with confidence; the result only re-confirmed it or changed nothing.
- not useful, "resolvable_without_tool": before the call the reasoning still had real doubt, but looking more carefully at the images already available, or a little more reasoning or mental arithmetic, would have resolved it without the tool.

A result that is ignored, redundant, unreadable or wrong is not useful. Judge the call, not the final answer.

Reply with only a JSON object:
{"useful": true or false, "unnecessary_type": "confident_before" or "resolvable_without_tool" or null, "reason": "<one sentence>"}
unnecessary_type is null exactly when useful is true."""

EDIT_PROMPT = """You edit the private reasoning of a visual reasoning agent so that it does not reveal what the agent will do next.

Change only two kinds of sentences, always as whole sentences:
1. Action sentences: any sentence that announces, plans or hints at the next step, e.g. "Let me zoom in to check.", "I'll compute it with a tool.", "I can answer directly.", "No tool is needed.", "Let me think more carefully."
2. Final-conclusion sentences: sentences that commit to the final answer, e.g. "So the answer is 12.", "Final answer: yes."

Leave every observation and reasoning sentence exactly as it is. Do not fix wording and do not add facts.
- Replacement for an action sentence: "" (delete it). If it also carries an observation, keep only the observation.
- Replacement for a conclusion sentence: the same content stated as current evidence without committing, e.g. "Current evidence points to 12."

Then write one sentence that starts with "Still missing:" and names the least certain piece of information in the reasoning. Always write it, even when the reasoning looks confident. Never say that nothing is missing, never name a tool, and never suggest an action.

Reply with only a JSON object:
{"edits": [{"original": "<sentence copied character for character from the reasoning>", "replacement": "<new sentence or empty string>"}], "still_missing": "Still missing: ..."}
Use an empty edits list if no sentence needs to change."""

SPLIT_PROMPT = """The reasoning below is split into numbered segments. Choose where to cut it into two parts so that the first part (segments 1..k) contains only observations and partial reasoning and has NOT yet reached the conclusion; the answer must be settled only in the second part. Prefer a cut near the middle.

Reply with only a JSON object: {"split_after": k} where 1 <= k <= N-1 and N is the number of segments."""

LEAK_PROMPT = """Below is the private reasoning of a visual reasoning agent, cut right before it acts. Guess what it does next.

Reply with only a JSON object: {"guess": "<one option copied exactly>"}"""


class BuildError(ValueError):
    """A rollout cannot be turned into the requested variant."""


# --------------------------------------------------------------------------- #
#  Runtime state
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    client: AsyncOpenAI
    model: str
    dataset: str
    dataset_dir: Path
    aux_dir: Path
    tau: float
    seed: int
    max_tool_calls: int
    eval_tools: list[str]                     # dataset tool set in canonical order
    sessions: dict[str, ClientSession] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)


# --------------------------------------------------------------------------- #
#  Parsing ShareGPT rollouts
# --------------------------------------------------------------------------- #

def _parse_param_value(value: str) -> Any:
    # reformat_dataset.render_tool_call json-dumps every non-string argument.
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_call_text(text: str) -> dict[str, Any] | None:
    """Parse the first Qwen3.5 XML ``<tool_call>`` in ``text``."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    arguments = {key.strip(): _parse_param_value(value) for key, value in PARAM_RE.findall(match.group(2))}
    return {"name": match.group(1).strip(), "arguments": arguments}


def parse_sample(item: dict[str, Any]) -> dict[str, Any]:
    """Split a ShareGPT rollout into turns of (think, action, observation)."""
    conversations = item.get("conversations") or []
    if len(conversations) < 2 or len(conversations) % 2 or conversations[0]["from"] != "human":
        raise BuildError("unexpected conversation layout")
    query = conversations[0]["value"].removeprefix("<image>").strip()
    turns = []
    for index in range(1, len(conversations), 2):
        gpt = conversations[index]
        if gpt["from"] != "gpt":
            raise BuildError("unexpected conversation layout")
        match = THINK_BLOCK_RE.match(gpt["value"])
        if not match:
            raise BuildError("gpt turn without a leading <think> block")
        think, action = match.group(1).strip(), match.group(2).strip()
        observation = conversations[index + 1]["value"] if index + 1 < len(conversations) else None
        call = parse_tool_call_text(action) if observation is not None else None
        if observation is not None and call is None:
            raise BuildError("tool turn without a parsable tool call")
        turns.append({
            "think": think,
            "action": action,
            "call": call,
            "observation": observation,
            "obs_image": observation is not None and "<image>" in observation,
        })
    final = turns[-1]
    tags = ANSWER_TAG_RE.findall(final["action"])
    answer = extract_boxed(tags[-1]) if tags else None
    if final["call"] is not None or not answer:
        raise BuildError("last turn is not a boxed answer")
    return {
        "id": item["id"],
        "query": query,
        "images": list(item["images"]),
        "tools_field": item["tools"],
        "turns": turns,
        "calls": [turn["call"] for turn in turns if turn["call"] is not None],
        # filter_out.py kept only trajectories whose answer matched the ground truth.
        "answer": normalize_answer(answer),
    }


def exclusion_reason(sample: dict[str, Any], max_tool_calls: int) -> str | None:
    """Trajectories with too many calls or a repeated identical call are dropped."""
    calls = sample["calls"]
    if len(calls) > max_tool_calls:
        return "too_many_tool_calls"
    keys = [json.dumps(call, sort_keys=True, ensure_ascii=False) for call in calls]
    if len(set(keys)) != len(keys):
        return "repeated_tool_call"
    return None


def is_correct(prediction: str, ground_truth: str) -> bool:
    return answers_match(normalize_answer(prediction), ground_truth)[0]


# --------------------------------------------------------------------------- #
#  Images
# --------------------------------------------------------------------------- #

def image_b64(images: list[str], index: int, cache: dict[int, str]) -> str:
    """Raw base64 of an image the rollout referenced.

    Index 0 is resized exactly as host.py resized it for the original rollout,
    so tool coordinates and the teacher's view match what produced the data.
    Later indices are tool outputs, already sized by the tool.
    """
    if index not in cache:
        if index == 0:
            cache[index] = _preprocess_image(images[0])
        else:
            cache[index] = base64.b64encode(Path(images[index]).read_bytes()).decode("ascii")
    return cache[index]


def image_part(b64: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


# --------------------------------------------------------------------------- #
#  LLM access
# --------------------------------------------------------------------------- #

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def chat(
    run: Run,
    messages: list[dict[str, Any]],
    *,
    sampling: dict[str, float],
    max_tokens: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    prefill: bool = False,
) -> dict[str, Any]:
    """One chat completion. ``prefill`` continues the last assistant message."""
    extra_body: dict[str, Any] = {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": True},
    }
    if prefill:
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False
    payload: dict[str, Any] = {
        "model": run.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
        **sampling,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    response = await run.client.chat.completions.create(**payload)
    run.stats["llm_requests"] += 1
    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    tool_calls = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = None
        tool_calls.append({"name": call.function.name, "arguments": arguments})
    return {
        "content": message.content or "",
        "reasoning": reasoning or "",
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """The last top-level JSON object in ``text``."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        end = text.rfind("}", 0, end)
    raise BuildError("no JSON object in reply")


async def ask_json(run, system, user_content, validate, *, thinking, sampling, max_tokens, step):
    """Ask for a JSON reply and return ``validate(obj)``, retrying invalid replies."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    last_error = ""
    for _ in range(LLM_ATTEMPTS):
        reply = await chat(run, messages, sampling=sampling, max_tokens=max_tokens, thinking=thinking)
        try:
            if reply["finish_reason"] == "length":
                raise BuildError("reply truncated")
            return validate(parse_json_object(reply["content"]))
        except (BuildError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            run.stats[f"invalid_{step}"] += 1
    raise BuildError(f"{step} failed: {last_error}")


# --------------------------------------------------------------------------- #
#  Classification (sees the ground truth)
# --------------------------------------------------------------------------- #

def _observation_text(observation: str) -> str:
    match = TOOL_RESPONSE_RE.search(observation)
    return (match.group(1) if match else observation).strip()


async def judge_call(run, sample, turn_index, image_index, cache):
    """Judge whether the tool call of ``turns[turn_index]`` was useful."""
    turn, after = sample["turns"][turn_index], sample["turns"][turn_index + 1]
    content = [
        {"type": "text", "text": f"QUESTION:\n{sample['query']}\n\nCORRECT ANSWER: {sample['answer']}\n\nINPUT IMAGE:"},
        image_part(image_b64(sample["images"], 0, cache)),
        {"type": "text", "text": (
            f"\n\nREASONING BEFORE THE CALL:\n{turn['think']}\n\nTOOL CALL:\n{turn['action']}\n\nTOOL RESULT:"
        )},
    ]
    if turn["obs_image"]:
        content.append(image_part(image_b64(sample["images"], image_index, cache)))
    else:
        content.append({"type": "text", "text": "\n" + _observation_text(turn["observation"])})
    content.append({"type": "text", "text": f"\n\nREASONING AFTER THE CALL:\n{after['think']}"})

    def validate(obj):
        useful = obj["useful"]
        kind = obj.get("unnecessary_type")
        if not isinstance(useful, bool):
            raise BuildError("useful must be a boolean")
        if useful:
            kind = None
        elif kind not in ("confident_before", "resolvable_without_tool"):
            raise BuildError(f"bad unnecessary_type {kind!r}")
        return {"useful": useful, "unnecessary_type": kind, "reason": str(obj.get("reason", ""))}

    return await ask_json(run, JUDGE_PROMPT, content, validate, thinking=True,
                          sampling=JUDGE_SAMPLING, max_tokens=JUDGE_MAX_TOKENS, step="judge")


async def classify(run, sample) -> dict[str, Any]:
    """R0-R3 class plus per-call judgments; ``switch_turn`` is the first unnecessary call."""
    if not sample["calls"]:
        return {"id": sample["id"], "class": "R0", "calls": [], "switch_turn": None}
    cache: dict[int, str] = {}
    jobs, image_index = [], 0
    for index, turn in enumerate(sample["turns"][:-1]):
        if turn["obs_image"]:
            image_index += 1
        jobs.append(judge_call(run, sample, index, image_index, cache))
    judgments = await asyncio.gather(*jobs)
    for index, judgment in enumerate(judgments):
        if not judgment["useful"]:
            label = "R2" if judgment["unnecessary_type"] == "confident_before" else "R3"
            return {"id": sample["id"], "class": label, "calls": judgments, "switch_turn": index}
    return {"id": sample["id"], "class": "R1", "calls": judgments, "switch_turn": None}


# --------------------------------------------------------------------------- #
#  Think editing (never sees the ground truth, selection or original action)
# --------------------------------------------------------------------------- #

def options_sentence(tools: list[str]) -> str:
    items = ["answer now", "think more", *tools]
    return f"Options: {', '.join(items[:-1])}, or {items[-1]}."


def apply_edits(think: str, edits: list[dict[str, Any]]) -> str:
    """Replace whole sentences; every original must occur verbatim in ``think``."""
    text, removed = think, 0
    for edit in edits:
        original = str(edit["original"]).strip()
        replacement = str(edit.get("replacement") or "").strip()
        if not original:
            raise BuildError("empty original sentence")
        if original not in text:
            raise BuildError(f"sentence not found: {original[:60]!r}")
        text = text.replace(original, replacement, 1)
        removed += len(original)
    # Observations must survive: an edit that rewrites most of a long think is rejected.
    if len(think) > 300 and removed > 0.5 * len(think):
        raise BuildError("edits rewrite most of the reasoning")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


async def edit_think(run, sample, think: str) -> str:
    """Neutralize action and conclusion sentences and append the options sentence."""
    think = EXTRA_TAG_RE.sub("", think)
    # The teacher sometimes imitates the appended sentences it saw in earlier thinks.
    think = STATUS_SENTENCES_RE.sub("", think).strip()

    def validate(obj):
        edits = obj["edits"]
        if not isinstance(edits, list):
            raise BuildError("edits must be a list")
        missing = str(obj["still_missing"]).strip()
        if not missing.startswith("Still missing:") or len(missing) < 20:
            raise BuildError("bad still_missing sentence")
        if TOOL_NAME_RE.search(missing):
            raise BuildError("still_missing names a tool")
        edited = apply_edits(think, edits)
        body = f"{edited}\n\n" if edited else ""
        return f"{body}{missing} {options_sentence(run.eval_tools)}"

    user = f"QUESTION:\n{sample['query']}\n\nREASONING:\n{think}"
    return await ask_json(run, EDIT_PROMPT, user, validate, thinking=False,
                          sampling=EDIT_SAMPLING, max_tokens=EDIT_MAX_TOKENS, step="edit")


async def split_think(run, think: str) -> tuple[str, str]:
    """Cut an R0 think where the first part has not concluded yet."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", think) if part.strip()]
    units, joiner = (paragraphs, "\n\n") if len(paragraphs) >= 2 else (
        [part for part in SENTENCE_SPLIT_RE.split(think) if part.strip()], " ")
    if len(units) < 2:
        raise BuildError("think too short to split")
    numbered = "\n\n".join(f"[{index}] {unit}" for index, unit in enumerate(units, start=1))

    def validate(obj):
        cut = int(obj["split_after"])
        if not 1 <= cut <= len(units) - 1:
            raise BuildError(f"split_after out of range: {cut}")
        return cut

    user = f"N = {len(units)}\n\n{numbered}"
    cut = await ask_json(run, SPLIT_PROMPT, user, validate, thinking=False,
                         sampling=EDIT_SAMPLING, max_tokens=EDIT_MAX_TOKENS, step="split")
    return joiner.join(units[:cut]), joiner.join(units[cut:])


# --------------------------------------------------------------------------- #
#  <evaluating> template
# --------------------------------------------------------------------------- #

def route(p: float, deltas: dict[str, float], tau: float) -> str:
    """Routing rule: best positive Δ, else answer_now if p >= τ, else think_more."""
    best = max(deltas, key=deltas.get)
    if deltas[best] > 0:
        return best
    return "answer_now" if p >= tau else "think_more"


def render_evaluating(selection: str, tools: list[str], tau: float, rng: random.Random) -> str:
    """Numbers are sampled after the selection so that ``route`` reproduces it.

    Sampling happens in integer hundredths so that the printed two-decimal
    values satisfy the rule exactly; p + Δ always stays inside [0.01, 0.99].
    """
    tau_c = round(tau * 100)
    if selection in tools:
        p = rng.randint(10, 85)
        best = rng.randint(5, min(40, 99 - p))
        deltas = {tool: rng.randint(max(-30, 1 - p), best - 1) for tool in tools}
        deltas[selection] = best
    else:
        p = rng.randint(tau_c, 97) if selection == "answer_now" else rng.randint(max(5, tau_c - 45), tau_c - 1)
        deltas = {tool: rng.randint(max(-30, 1 - p), -1) for tool in tools}
    if route(p, deltas, tau_c) != selection:
        raise AssertionError(f"evaluating numbers disagree with selection {selection}")
    lines = ["<evaluating>", f"answer_now: p={p / 100:.2f}"]
    lines += [f"{tool}: Δ={deltas[tool] / 100:+.2f}" for tool in tools]
    lines += [f"selection: {selection}", "</evaluating>"]
    return "\n".join(lines)


def render_turn(run: Run, turn: dict[str, Any], rng: random.Random) -> str:
    """gpt message text: (think, evaluating)* followed by the action."""
    if len(turn["thinks"]) != len(turn["selections"]):
        raise BuildError("every think block needs exactly one selection")
    blocks = []
    for think, selection in zip(turn["thinks"], turn["selections"]):
        blocks.append(f"<think>\n{think}\n</think>")
        blocks.append(render_evaluating(selection, run.eval_tools, run.tau, rng))
    blocks.append(turn["action"])
    return "\n\n".join(blocks)


def render_answer(statement: str, boxed: str) -> str:
    block = f"<answer>\n\\boxed{{{escape_image_placeholder(boxed)}}}\n</answer>"
    return f"{statement}\n\n{block}" if statement else block


# --------------------------------------------------------------------------- #
#  Teacher regeneration
# --------------------------------------------------------------------------- #

def teacher_tools(sample) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool} for tool in json.loads(sample["tools_field"])]


def teacher_history(sample, done_turns, images, cache) -> list[dict[str, Any]]:
    """OpenAI messages for the finished tool turns, as host.py would have sent them.

    ``images`` is the image history of the trajectory being built, which past
    the switch point differs from the original rollout's.
    """
    messages = [
        {"role": "system", "content": GENERATION_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"QUESTION:\n{sample['query']}\nINPUT_IMAGE:\n"},
            image_part(image_b64(images, 0, cache)),
        ]},
    ]
    image_index = 0
    for turn in done_turns:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning": "\n\n".join(turn["teacher_thinks"]),
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": turn["call"]["name"],
                             "arguments": json.dumps(turn["call"]["arguments"], ensure_ascii=False)},
            }],
        })
        if turn["obs_image"]:
            image_index += 1
            content: Any = [
                {"type": "text", "text": f"[image_index={image_index}] Produced image is shown below."},
                image_part(image_b64(images, image_index, cache)),
            ]
        else:
            content = _observation_text(turn["observation"])
        messages.append({"role": "tool", "tool_call_id": call_id, "name": turn["call"]["name"], "content": content})
    return messages


def split_reasoning(reply: dict[str, Any], prefilled_think: bool) -> tuple[str, str]:
    """(new think text, text after it) from a reply that is expected to open with a think.

    Reasoning parsers differ on where text goes when the prompt already opened
    or closed a think block, so both fields are inspected.
    """
    reasoning, content = reply["reasoning"], reply["content"]
    if reasoning and "</think>" not in content:
        return reasoning, content
    combined = reasoning + content
    if "</think>" in combined:
        think, _, rest = combined.partition("</think>")
        return think.removeprefix("<think>"), rest
    if not prefilled_think and reasoning:
        return reasoning, content
    raise BuildError("reply does not close its think block")


def checked_think(think: str) -> str:
    """A new think block must carry reasoning; the model sometimes closes it at once."""
    if len(think.strip()) < 20:
        raise BuildError("empty think block")
    return sanitize_reasoning(think)


def parse_action(run, reply, text) -> dict[str, Any]:
    """Classify the text after a think as a tool call, an answer or nothing."""
    call = reply["tool_calls"][0] if reply["tool_calls"] else parse_tool_call_text(text)
    if call is not None:
        if call["name"] not in run.eval_tools or not isinstance(call["arguments"], dict):
            raise BuildError(f"invalid tool call {call['name']!r}")
        arguments = {k: v for k, v in call["arguments"].items() if k not in HOST_INJECTED_ARGUMENTS}
        return {"kind": "tool", "call": {"name": call["name"], "arguments": arguments}}
    tags = ANSWER_TAG_RE.findall(text)
    if tags:
        boxed = extract_boxed(tags[-1])
        if not boxed or "<think" in text or "<tool_call" in text:
            raise BuildError("malformed answer")
        statement = text[: text.lower().find("<answer")]
        statement = escape_image_placeholder(EXTRA_TAG_RE.sub("", SPECIAL_TAG_RE.sub("", statement)).strip())
        return {"kind": "answer", "statement": statement, "boxed": boxed}
    if not text.strip():
        return {"kind": "none"}
    raise BuildError("text after think is neither a tool call nor an answer")


async def teacher_step(run, sample, history, thinks, step, cache) -> tuple[str | None, dict[str, Any]]:
    """Generate one decision of the current turn.

    step: None (fresh turn), "think_more" (open a new think block),
    "answer_now" (answer after the given thinks) or a tool name (call it).
    Returns (new think or None, action).
    """
    tools = teacher_tools(sample)
    last_error = ""
    for _ in range(LLM_ATTEMPTS):
        try:
            if step is None:
                reply = await chat(run, history, sampling=GEN_SAMPLING, max_tokens=GEN_MAX_TOKENS,
                                   thinking=True, tools=tools)
                if reply["finish_reason"] == "length":
                    raise BuildError("reply truncated")
                think, rest = split_reasoning(reply, prefilled_think=False)
                return checked_think(think), parse_action(run, reply, rest)

            current = {"role": "assistant", "reasoning": "\n\n".join(thinks)}
            if step == "think_more":
                # think_more continues without tools (Case 4); with a tool list in
                # the prompt the teacher almost always reaches for one instead.
                current["content"] = "<think>\n"
                reply = await chat(run, [*history, current], sampling=GEN_SAMPLING, max_tokens=GEN_MAX_TOKENS,
                                   thinking=True, prefill=True)
                if reply["finish_reason"] == "length":
                    raise BuildError("reply truncated")
                think, rest = split_reasoning(reply, prefilled_think=True)
                action = parse_action(run, reply, rest)
                if action["kind"] == "tool":
                    raise BuildError("tool call after think_more")
                return checked_think(think), action

            current["content"] = ""
            if step == "answer_now":
                # Tools stay out of this request: with a tool list in the prompt the
                # model starts a tool call right after </think>, which tool_choice
                # "none" then drops, leaving an empty reply.
                reply = await chat(run, [*history, current], sampling=GEN_SAMPLING, max_tokens=GEN_MAX_TOKENS,
                                   thinking=True, prefill=True)
                if reply["finish_reason"] == "length":
                    raise BuildError("reply truncated")
                text = reply["reasoning"] + reply["content"]
                if "<think" in text or "</think" in text or "<tool_call" in text:
                    raise BuildError("answer continuation left the answer format")
                action = parse_action(run, {"tool_calls": []}, text)
                if action["kind"] != "answer":
                    raise BuildError("forced answer produced no answer")
                return None, action

            reply = await chat(run, [*history, current], sampling=GEN_SAMPLING, max_tokens=GEN_MAX_TOKENS,
                               thinking=True, tools=tools, prefill=True,
                               tool_choice={"type": "function", "function": {"name": step}})
            if not reply["tool_calls"]:
                reply["tool_calls"] = [{"name": step, "arguments": parse_json_object(reply["content"])}]
            action = parse_action(run, reply, "")
            if action["kind"] != "tool" or action["call"]["name"] != step:
                raise BuildError("forced tool call not produced")
            return None, action
        except BuildError as exc:
            last_error = str(exc)
            run.stats["invalid_teacher_step"] += 1
    raise BuildError(f"teacher step {step or 'free'} failed: {last_error}")


async def execute_tool(run, sample, call, images, cache) -> dict[str, Any]:
    """Run a regenerated tool call for real; returns observation fields."""
    name = call["name"]
    arguments = dict(call["arguments"])
    if name in IMAGE_INPUT_TOOLS:
        index = _resolve_image_index(arguments, images)
        # Tools read the original file at index 0, as host.py does in the rollouts; image_b64's index 0 is the
        # downscaled copy the model and the teacher see.
        arguments["image"] = _file_b64(images[0]) if index == 0 else image_b64(images, index, cache)
        if name in MODEL_BACKED_TOOLS:
            arguments["model_name"] = run.model
            arguments["api_key"] = os.getenv("DIRECT_API_KEY", "")
            arguments["host_url"] = os.getenv("DIRECT_BASE_URL", "")
    else:
        arguments.pop("image_index", None)
    if name in CODE_EXECUTION_TOOLS:
        arguments["task_cache_dir"] = str(run.aux_dir / "code_execution" / sample["id"] / uuid.uuid4().hex[:8])

    result = await run.sessions[name].call_tool(name, arguments)
    run.stats["tool_executions"] += 1
    first = result.content[0]
    error = bool(getattr(result, "is_error", False))
    image_path, text = None, None

    if name in DIRECT_IMAGE_PRODUCING_TOOLS:
        if error or not hasattr(first, "data"):
            raise BuildError(f"{name} failed: {getattr(first, 'text', '')[:200]}")
        out_dir = run.aux_dir / "tool_results" / sample["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"{name}_{uuid.uuid4().hex[:12]}.png"
        image_path.write_bytes(base64.b64decode(_fit_to_image_budget(first.data)))  # sized as host.py does
    elif name in CODE_EXECUTION_TOOLS:
        payload = _parse_tool_payload(first.text)
        text = _code_tool_feedback(payload, name)
        error = error or bool(payload.get("execution_error"))
    else:
        text = getattr(first, "text", str(first))

    if image_path is not None:
        images.append(str(image_path.resolve()))
        return {"observation": "<tool_response>\n<image>\n</tool_response>", "obs_image": True}
    text = (text or "").strip()
    if error or not text or TOOL_ERROR_TEXT_RE.match(text):
        raise BuildError(f"{name} failed: {text[:200]}")
    return {"observation": f"<tool_response>\n{_escape_observation(text)}\n</tool_response>", "obs_image": False}


async def regenerate(run, sample, done_turns, images, first_thinks, first_step, cache) -> list[dict[str, Any]]:
    """Teacher-generate every turn from the switch point to the final answer.

    ``first_thinks`` are the (already edited) thinks of the switch turn and
    ``first_step`` the forced selection there. Selections of later decisions
    are read off the teacher's own actions. Returns the new turns; thinks the
    teacher wrote are marked for editing.
    """
    turns: list[dict[str, Any]] = []
    n_calls = sum(1 for turn in done_turns if turn.get("call"))
    thinks, needs_edit, selections, step = list(first_thinks), [False] * len(first_thinks), [], first_step
    while True:
        history = teacher_history(sample, [*done_turns, *turns], images, cache)
        if step is not None and step not in ("think_more", "answer_now") and n_calls >= run.max_tool_calls:
            step = "answer_now"
        if step in ("think_more", "answer_now") or step in run.eval_tools:
            selections.append(step)
        new_think, action = await teacher_step(run, sample, history, thinks, step, cache)
        if new_think is not None:
            thinks.append(new_think)
            needs_edit.append(True)

        if action["kind"] == "none":
            step = "answer_now" if selections.count("think_more") >= MAX_THINK_MORE else "think_more"
            continue
        if action["kind"] == "tool" and step not in run.eval_tools:
            if n_calls >= run.max_tool_calls:
                step = "answer_now"
                continue
            selections.append(action["call"]["name"])
        elif action["kind"] == "answer" and step != "answer_now":
            selections.append("answer_now")

        turn = {"thinks": thinks, "needs_edit": needs_edit, "selections": selections, "teacher_thinks": list(thinks)}
        if action["kind"] == "answer":
            turn["action"] = render_answer(action["statement"], action["boxed"])
            turn["boxed"] = action["boxed"]
            turns.append(turn)
            return turns

        n_calls += 1
        observation = await execute_tool(run, sample, action["call"], images, cache)
        turn.update(call=action["call"], action=render_tool_call(action["call"]), **observation)
        turns.append(turn)
        thinks, needs_edit, selections, step = [], [], [], None


# --------------------------------------------------------------------------- #
#  Variant builders
# --------------------------------------------------------------------------- #

async def kept_tool_turns(run, sample, count) -> list[dict[str, Any]]:
    """The first ``count`` original tool turns with edited thinks and tool selections."""
    originals = sample["turns"][:count]
    edited = await asyncio.gather(*(edit_think(run, sample, turn["think"]) for turn in originals))
    return [
        {
            "thinks": [think],
            "selections": [turn["call"]["name"]],
            "teacher_thinks": [think],
            "action": turn["action"],
            "call": turn["call"],
            "observation": turn["observation"],
            "obs_image": turn["obs_image"],
        }
        for turn, think in zip(originals, edited)
    ]


def override_tool(run, sample, rng) -> str:
    """A random dataset tool; numeric tools only when the context has numbers."""
    context = sample["query"] + " " + sample["turns"][-1]["think"]
    numbers = set(NUMBER_RE.findall(context))
    candidates = [tool for tool in run.eval_tools if len(numbers) >= 2 or tool not in NUMERIC_TOOLS]
    return rng.choice(candidates or run.eval_tools)


async def build_variant(run, sample, variant, judgment, rng) -> dict[str, Any]:
    """Build one output record; raises BuildError when the variant cannot be made."""
    final = sample["turns"][-1]
    if variant == "r0_text":
        return finish_record(run, sample, variant, None, sample["images"], rng, regenerated=False)

    if variant == "r0_answer_now":
        think = await edit_think(run, sample, final["think"])
        turns = [{"thinks": [think], "selections": ["answer_now"], "action": final["action"]}]
        return finish_record(run, sample, variant, turns, sample["images"], rng, regenerated=False)

    if variant == "r0_split":
        first, second = await split_think(run, final["think"])
        thinks = await asyncio.gather(edit_think(run, sample, first), edit_think(run, sample, second))
        turns = [{"thinks": list(thinks), "selections": ["think_more", "answer_now"], "action": final["action"]}]
        return finish_record(run, sample, variant, turns, sample["images"], rng, regenerated=False)

    if variant == "r1_tool":
        turns = await kept_tool_turns(run, sample, len(sample["turns"]) - 1)
        think = await edit_think(run, sample, final["think"])
        turns.append({"thinks": [think], "selections": ["answer_now"], "action": final["action"]})
        return finish_record(run, sample, variant, turns, sample["images"], rng, regenerated=False)

    # Regenerated variants: keep the tool turns before the switch point, force
    # the selection there, and let the teacher write the rest.
    if variant == "r0_tool_override":
        switch, step = 0, override_tool(run, sample, rng)
    elif variant in ("r2_answer_now", "r3_think_more"):
        switch, step = judgment["switch_turn"], "answer_now" if variant == "r2_answer_now" else "think_more"
    else:
        switch = rng.randrange(len(sample["calls"]))
        step = "answer_now" if variant == "r1_answer_override" else "think_more"

    done = await kept_tool_turns(run, sample, switch)
    kept_images = 1 + sum(1 for turn in done if turn["obs_image"])
    images = sample["images"][:kept_images]
    cache: dict[int, str] = {}
    switch_think = await edit_think(run, sample, sample["turns"][switch]["think"])
    new_turns = await regenerate(run, sample, done, images, [switch_think], step, cache)

    boxed = new_turns[-1]["boxed"]
    if not is_correct(boxed, sample["answer"]):
        raise BuildError("regenerated answer is wrong")
    for turn in new_turns:
        pairs = list(zip(turn["thinks"], turn["needs_edit"]))
        edited = iter(await asyncio.gather(*(edit_think(run, sample, think) for think, flag in pairs if flag)))
        turn["thinks"] = [next(edited) if flag else think for think, flag in pairs]
    return finish_record(run, sample, variant, done + new_turns, images, rng, regenerated=True)


def finish_record(run, sample, variant, turns, images, rng, *, regenerated) -> dict[str, Any]:
    """ShareGPT record in the input's layout plus a ``meta`` field."""
    source, case, _ = MIX[variant]
    conversations = [{"from": "human", "value": f"<image>\n{sample['query']}"}]
    if turns is None:  # Case 1 keeps the rollout untouched
        for turn in sample["turns"]:
            conversations.append({"from": "gpt", "value": f"<think>\n{turn['think']}\n</think>\n\n{turn['action']}"})
            if turn["observation"] is not None:
                conversations.append({"from": "human", "value": turn["observation"]})
    else:
        for turn in turns:
            conversations.append({"from": "gpt", "value": render_turn(run, turn, rng)})
            if turn.get("observation") is not None:
                conversations.append({"from": "human", "value": turn["observation"]})
    placeholders = sum(message["value"].count("<image>") for message in conversations)
    if placeholders != len(images):
        raise BuildError(f"{placeholders} <image> placeholders for {len(images)} images")
    return {
        "id": sample["id"],
        "system": SYSTEM_PROMPT,
        "tools": sample["tools_field"],
        "conversations": conversations,
        "images": list(images),
        "meta": {"dataset": run.dataset, "variant": variant, "case": case, "source": source,
                 "regenerated": regenerated},
    }


# --------------------------------------------------------------------------- #
#  Planning and balancing
# --------------------------------------------------------------------------- #

def plan_variants(pools: dict[str, list[str]], rng: random.Random) -> dict[str, list[str]]:
    """Split every source pool over its variants by share; every id is used exactly once.

    No source limits another: a scarce R1 pool does not shrink how many R0 or
    R2 rollouts get built.
    """
    by_source: dict[str, list[str]] = {}
    for variant, (source, _, _) in MIX.items():
        by_source.setdefault(source, []).append(variant)

    assignment = {}
    for source, variants in by_source.items():
        pool = list(pools.get(source, []))
        rng.shuffle(pool)
        total_share = sum(MIX[v][2] for v in variants)
        # Cumulative rounding so the slices cover the whole pool with no remainder.
        start, cumulative = 0, 0.0
        for variant in variants:
            cumulative += MIX[variant][2]
            end = round(len(pool) * cumulative / total_share)
            assignment[variant] = pool[start:end]
            start = end
    return assignment


def selection_type(selection: str) -> str:
    return selection if selection in ("answer_now", "think_more") else "tool"


def mix_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and shares per variant, Case and selection type (one per <evaluating> block).

    Selection-following has to be learned for every selection type, so these
    are the numbers the merged-dataset mix is decided on.
    """
    total = len(records)
    variants = Counter(record["meta"]["variant"] for record in records)
    cases = Counter(f"case{record['meta']['case']}" for record in records)
    selections = Counter(
        selection_type(selection)
        for record in records
        for message in record["conversations"]
        if message["from"] == "gpt"
        for selection in re.findall(r"^selection: (\S+)$", message["value"], re.MULTILINE)
    )
    blocks = sum(selections.values())

    def with_share(counter, denominator, keys):
        return {key: {"count": counter.get(key, 0),
                      "share": round(counter.get(key, 0) / denominator, 3) if denominator else 0.0}
                for key in keys}

    return {
        "records": total,
        "variants": with_share(variants, total, MIX),
        "cases": with_share(cases, total, ["case1", "case2", "case3", "case4"]),
        "selections": with_share(selections, blocks, ["tool", "answer_now", "think_more"]),
    }


# --------------------------------------------------------------------------- #
#  Pilot leak probe
# --------------------------------------------------------------------------- #

async def leak_probe(run, record) -> list[tuple[str, str]]:
    """(true selection, guessed selection) for every evaluating block of a record."""
    options = ["answer_now", "think_more", *run.eval_tools]
    pairs = []
    for message in record["conversations"]:
        if message["from"] != "gpt":
            continue
        for think, selection in re.findall(
            r"<think>\n(.*?)\n</think>\n\n<evaluating>.*?selection: (\S+)\n</evaluating>", message["value"], re.DOTALL
        ):
            def validate(obj):
                guess = str(obj["guess"]).strip()
                if guess not in options:
                    raise BuildError(f"unknown option {guess!r}")
                return guess

            user = f"OPTIONS: {', '.join(options)}\n\nREASONING:\n{think}"
            guess = await ask_json(run, LEAK_PROMPT, user, validate, thinking=False,
                                   sampling=EDIT_SAMPLING, max_tokens=EDIT_MAX_TOKENS, step="leak")
            pairs.append((selection, guess))
    return pairs


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #

async def connect_tools(run: Run, server_dir: Path, stack: AsyncExitStack) -> None:
    """Start the MCP servers that own the dataset's tools."""
    discovered = discover_tool_servers(server_dir)
    scripts: dict[Path, list[str]] = {}
    for name in run.eval_tools:
        scripts.setdefault(discovered[name], []).append(name)
    for script, names in scripts.items():
        params = StdioServerParameters(command=sys.executable, args=[str(script)], cwd=str(script.parent))
        streams = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(*streams))
        await session.initialize()
        for name in names:
            run.sessions[name] = session


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_judgments(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, filter(str.strip, handle))}


def record_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


async def build_dataset(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    dataset = dataset_dir.name
    if dataset not in DATASET_TOOL_NAMES:
        raise ValueError(f"Unknown dataset directory '{dataset}'. Expected one of {sorted(DATASET_TOOL_NAMES)}")
    aux_dir = dataset_dir / AUX_DIRNAME
    aux_dir.mkdir(parents=True, exist_ok=True)

    model = os.getenv("MODEL_NAME")
    base_url = os.getenv("DIRECT_BASE_URL")
    if not model or not base_url:
        raise ValueError("MODEL_NAME and DIRECT_BASE_URL must be set (dataset_construction/.env)")
    run = Run(
        client=AsyncOpenAI(base_url=base_url, api_key=os.getenv("DIRECT_API_KEY", "EMPTY"), timeout=1800),
        model=model,
        dataset=dataset,
        dataset_dir=dataset_dir,
        aux_dir=aux_dir,
        tau=args.tau,
        seed=args.seed,
        max_tool_calls=args.max_tool_calls,
        eval_tools=[tool for tool in CANONICAL_TOOL_ORDER if tool in DATASET_TOOL_NAMES[dataset]],
    )
    report: dict[str, Any] = {"dataset": dataset, "pilot": args.pilot, "tau": args.tau, "seed": args.seed,
                              "model": model, "target_mix": {v: MIX[v][2] for v in MIX}}

    # ---- Load and parse --------------------------------------------------- #
    raw = load_samples(dataset_dir / INPUT_FILENAME)
    if args.pilot:
        raw = random.Random(args.seed).sample(raw, min(args.pilot, len(raw)))
    samples, excluded = {}, Counter()
    for item in raw:
        try:
            sample = parse_sample(item)
        except (BuildError, KeyError) as exc:
            excluded[f"parse: {exc}"] += 1
            continue
        reason = exclusion_reason(sample, args.max_tool_calls)
        if reason:
            excluded[reason] += 1
            continue
        samples[sample["id"]] = sample
    report["input"] = {"records": len(raw), "usable": len(samples), "excluded": dict(excluded)}
    logger.info("[%s] %d usable rollouts (%d excluded)", dataset, len(samples), sum(excluded.values()))

    semaphore = asyncio.Semaphore(args.concurrency)

    # ---- 1. Classification (cached) --------------------------------------- #
    judgments_path = aux_dir / "judgments.jsonl"
    judgments = load_judgments(judgments_path)
    todo = [sid for sid in samples if sid not in judgments]
    logger.info("[%s] classifying %d rollouts (%d cached)", dataset, len(todo), len(samples) - len(todo))
    write_lock = asyncio.Lock()

    async def classify_one(sid):
        async with semaphore:
            try:
                result = await classify(run, samples[sid])
            except Exception as exc:  # one failed rollout must not stop the run
                run.stats["judge_failed"] += 1
                logger.warning("[%s] judging %s failed: %s", dataset, sid, exc)
                return
        judgments[sid] = result
        async with write_lock:
            with judgments_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    await asyncio.gather(*(classify_one(sid) for sid in todo))
    pools: dict[str, list[str]] = {}
    for sid in samples:
        if sid in judgments:
            pools.setdefault(judgments[sid]["class"], []).append(sid)
    report["classes"] = {label: len(pools.get(label, [])) for label in ("R0", "R1", "R2", "R3")}
    logger.info("[%s] classes %s", dataset, report["classes"])

    # ---- 2. Plan ---------------------------------------------------------- #
    assignment = plan_variants(pools, record_rng(args.seed, dataset, "plan"))
    report["planned"] = {variant: len(ids) for variant, ids in assignment.items()}

    # ---- 3. Build --------------------------------------------------------- #
    survivors: dict[str, list[dict[str, Any]]] = {variant: [] for variant in MIX}
    failures: dict[str, Counter] = {variant: Counter() for variant in MIX}

    async def build_one(variant, sid):
        async with semaphore:
            rng = record_rng(args.seed, dataset, variant, sid)
            try:
                record = await build_variant(run, samples[sid], variant, judgments[sid], rng)
            except BuildError as exc:
                failures[variant][str(exc).split(":")[0]] += 1
                return
            except Exception as exc:  # unexpected server/tool errors: count and move on
                failures[variant][f"error: {type(exc).__name__}"] += 1
                logger.warning("[%s] %s/%s failed: %r", dataset, variant, sid, exc)
                return
        survivors[variant].append(record)

    async with AsyncExitStack() as stack:
        if any(assignment[variant] for variant in REGENERATED):
            await connect_tools(run, args.server_dir.resolve(), stack)
        jobs = [build_one(variant, sid) for variant, ids in assignment.items() for sid in ids]
        logger.info("[%s] building %d records", dataset, len(jobs))
        await asyncio.gather(*jobs)

    report["built"] = {
        variant: {
            "planned": len(assignment[variant]),
            "survived": len(survivors[variant]),
            "survival_rate": round(len(survivors[variant]) / len(assignment[variant]), 3) if assignment[variant] else None,
            "failures": dict(failures[variant]),
        }
        for variant in MIX
    }

    # ---- 4. Write every survivor ------------------------------------------ #
    records = [record for variant in MIX for record in survivors[variant]]
    record_rng(args.seed, dataset, "shuffle").shuffle(records)
    if args.pilot:
        pairs = []
        probe_records = [record for record in records if record["meta"]["case"] != 1]
        async def probe(record):
            async with semaphore:
                try:
                    pairs.extend(await leak_probe(run, record))
                except BuildError:
                    run.stats["leak_probe_failed"] += 1
        await asyncio.gather(*(probe(record) for record in probe_records))
        by_selection: dict[str, list[int]] = {}
        for truth, guess in pairs:
            by_selection.setdefault(selection_type(truth), []).append(int(truth == guess))
        report["leak_probe"] = {
            "blocks": len(pairs),
            "accuracy": round(sum(t == g for t, g in pairs) / len(pairs), 3) if pairs else None,
            "chance": round(1 / (2 + len(run.eval_tools)), 3),
            "accuracy_by_selection": {k: round(sum(v) / len(v), 3) for k, v in by_selection.items()},
            # "true -> guessed" counts; tool names are folded into "tool".
            "confusion": dict(Counter(f"{selection_type(t)} -> {selection_type(g)}" for t, g in pairs)),
        }
        output_path = aux_dir / "pilot_formatted.jsonl"
        report_path = aux_dir / "pilot_report.json"
    else:
        output_path = dataset_dir / OUTPUT_FILENAME
        report_path = aux_dir / "report.json"

    total = len(records)
    report["final"] = mix_report(records)
    report["written"] = total
    report["stats"] = dict(run.stats)

    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    await run.client.close()
    logger.info("[%s] wrote %d records -> %s (report: %s)", dataset, total, output_path, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CaVIR cold-start v2 data with <evaluating> blocks.")
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help=f"outputs/<dataset> directory containing {INPUT_FILENAME}.")
    parser.add_argument("--server-dir", type=Path, default=Path(__file__).resolve().parent / "server")
    parser.add_argument("--tau", type=float, default=0.7, help="answer_now threshold τ of the routing rule.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32, help="Rollouts processed at the same time.")
    parser.add_argument("--max-tool-calls", type=int, default=3,
                        help="Rollouts with more calls are dropped; regenerated turns may not exceed it.")
    parser.add_argument("--pilot", type=int, default=0,
                        help="Run on N random rollouts, add a leak probe, and write pilot files instead.")
    args = parser.parse_args()
    if not 0.06 <= args.tau <= 0.97:
        parser.error("--tau must be within [0.06, 0.97]")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    asyncio.run(build_dataset(parse_args()))


if __name__ == "__main__":
    main()
