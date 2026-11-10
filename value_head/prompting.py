"""Render policy prompts exactly as cold start SFT encoded them, and parse what the policy generates.

The text matches LLaMA Factory's ``qwen3_5`` template with ``encode_multiturn`` (earlier turns keep their
``<think>``): system prompt followed by the Qwen3.5 tool block, ``user``/``assistant`` turns, and tool results
as ``user`` turns holding ``<tool_response>``. Each ``<image>`` placeholder becomes one ``<|image_pad|>``
between vision markers; the serving engine expands it to the image's token count. ``check_serving.py``
verifies the match token by token. Pure Python, so it imports in both the ``research`` and the
``train_cluster`` env.
"""

from __future__ import annotations

import json
import re

IMAGE_PLACEHOLDER = "<image>"
VISION_SPAN = "<|vision_start|><|image_pad|><|vision_end|>"

# LLaMA Factory tool_utils.QWEN35_TOOL_PROMPT
QWEN35_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>{tool_text}"
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: "
    "an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def tool_block(tools_field: str) -> str:
    tools = json.loads(tools_field)
    if not tools:
        return ""
    text = "".join("\n" + json.dumps(tool.get("function", tool) if tool.get("type") == "function" else tool,
                                     ensure_ascii=False) for tool in tools)
    return QWEN35_TOOL_PROMPT.format(tool_text=text)


def render(system: str, tools_field: str, turns: list[str], partial: str | None = "") -> str:
    """Prompt text for ``turns`` (user, assistant, user, ...; the first is the question).

    ``partial`` is the unfinished assistant message the policy continues from ("" starts a fresh
    reply). With ``partial=None`` the turns are rendered as a finished conversation, which ends with
    an assistant turn; used only to compare against training encodings.
    """
    text = f"<|im_start|>system\n{system}{tool_block(tools_field)}<|im_end|>\n"
    for index, content in enumerate(turns):
        content = content.replace(IMAGE_PLACEHOLDER, VISION_SPAN)
        if index % 2 == 0:
            text += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            text += f"{content}<|im_end|>\n"
    if partial is not None:
        if len(turns) % 2 == 0:
            raise ValueError("a partial assistant message must follow a user turn")
        text += partial
    return text


def parse_tool_call(text: str) -> dict | None:
    """First ``<tool_call>`` in ``text`` as {name, arguments}; values are JSON-decoded when they parse."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    arguments = {}
    for key, value in PARAM_RE.findall(match.group(2)):
        try:
            arguments[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key.strip()] = value
    return {"name": match.group(1).strip(), "arguments": arguments}


def first_action(text: str) -> str:
    """Which tag starts first after an ``<evaluating>`` block: think_more, answer_now, tool_call or none."""
    found = {tag: text.find(tag) for tag in ("<think>", "<tool_call>", "<answer>") if tag in text}
    if not found:
        return "none"
    return {"<think>": "think_more", "<tool_call>": "tool_call", "<answer>": "answer_now"}[min(found, key=found.get)]
