#!/usr/bin/env python3
r"""Convert filtered CaViR rollouts into multimodal ShareGPT SFT data (Qwen3.5)."""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import math
import random
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as qwen_vl_smart_resize

INPUT_FILENAME = "filtered_results.jsonl"
OUTPUT_FILENAME = "reformatted_results.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_MAX_CONTEXT_LENGTH = 32000
DEFAULT_IMAGE_MAX_PIXELS = 262144
DEFAULT_IMAGE_MIN_PIXELS = 32 * 32
DEFAULT_MAX_CLASS_RATIO = 2.0

# Matches LLaMA Factory's Qwen35ToolUtils: tool calls are XML, not JSON.
SYSTEM_PROMPT = r"""You are a multimodal reasoning agent. Answer questions by combining visual reasoning, logical reasoning, and tools.

Principles:
- Gather evidence before answering; do not guess when key information is missing.
- Call a tool only when it gives information or computation you cannot get reliably on your own.
- Always use a returned observation in your next reasoning step.
- Do not repeat a call that yields no new information.
- Prefer the simplest tool that works.

Output format. Every response begins with exactly one <think> block. No text may appear outside the allowed tags.

To call a tool:
<think>
...
</think>
<tool_call>
<function=tool_name>
<parameter=first_argument>
value
</parameter>
<parameter=second_argument>
value
</parameter>
</function>
</tool_call>

To give the final answer:
<think>
...
</think>
<answer>
\boxed{...}
</answer>"""

ANSWER_TAG_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"^\s*<think\s*>(.*?)</think\s*>\s*$", re.IGNORECASE | re.DOTALL)
# Qwen3.5 tool calls introduce <function=...> and <parameter=...> as structural tags.
SPECIAL_TAG_RE = re.compile(
    r"</?(?:think|tool_call|tool_response|observation|answer)\s*>"
    r"|</?\s*(?:function|parameter)\s*(?:=[^<>]*)?>",
    re.IGNORECASE,
)
STRUCTURAL_TAG_RE = re.compile(
    r"</?\s*(?:function|parameter|tool_call)\s*(?:=[^<>]*)?>",
    re.IGNORECASE,
)

IMAGE_RESULT_TOOLS = {"crop_zoomin", "mark_dots", "draw_bbox"}
IMAGE_INPUT_TOOLS = {"crop_zoomin", "mark_dots", "draw_bbox", "crop_ocr_image"}
HOST_INJECTED_ARGUMENTS = {"image", "model_name", "api_key", "host_url", "task_cache_dir"}
FALLBACK_REASONING = "I will use the available evidence to answer the question."

# One tool set for every dataset. Its order is build_cavir_coldstart's
# CANONICAL_TOOL_ORDER, the order the tools appear in the prompt and the <evaluating> block.
TOOL_SET = ("crop_zoomin", "crop_ocr_image", "draw_bbox", "mark_dots", "math_compute")
DATASET_TOOL_NAMES = {name: TOOL_SET for name in (
    "docvqa", "infographics_vqa", "chartqa_1", "chartqa_2", "plotqa", "gqa_1", "gqa_2", "geometry3k", "geomverse_cot",
    "hopchain_fintabnet", "chartnet_qa_01_difficulty", "coco_count_spatial_qa",
)}

# Copied verbatim from LLaMA Factory's QWEN35_TOOL_PROMPT (tool_format="qwen3_5").
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


class ReformatError(ValueError):
    """A rollout cannot be represented safely in the target format."""


def _image_index_schema() -> dict[str, Any]:
    return {
        "type": "integer",
        "description": (
            "Index of the image to operate on. 0 is the original input image; "
            "each successful image-producing tool call appends its output as "
            "the next index. Never pass raw image bytes."
        ),
        "default": 0,
    }


async def _load_tools_async(server_dir: Path) -> list[dict[str, Any]]:
    """Read actual MCP 2.x tool descriptions and schemas from every server."""
    scripts = sorted(server_dir.glob("*_tool.py"))
    if not scripts:
        raise FileNotFoundError(f"No *_tool.py files found under {server_dir}")

    definitions: list[dict[str, Any]] = []
    names: set[str] = set()
    async with AsyncExitStack() as stack:
        for script in scripts:
            streams = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(script)],
                        cwd=str(server_dir),
                    )
                )
            )
            session = await stack.enter_async_context(ClientSession(*streams))
            await session.initialize()

            for tool in (await session.list_tools()).tools:
                if tool.name in names:
                    raise ValueError(f"Duplicate MCP tool name: {tool.name}")
                names.add(tool.name)

                parameters = copy.deepcopy(tool.input_schema) or {}
                properties = parameters.get("properties", {})
                for field in HOST_INJECTED_ARGUMENTS:
                    properties.pop(field, None)
                parameters["required"] = [
                    name
                    for name in parameters.get("required", [])
                    if name not in HOST_INJECTED_ARGUMENTS
                ]
                if tool.name in IMAGE_INPUT_TOOLS:
                    properties["image_index"] = _image_index_schema()
                parameters["properties"] = properties

                definitions.append(
                    {
                        "name": tool.name,
                        "description": (tool.description or "").strip(),
                        "parameters": parameters,
                    }
                )
    return definitions


def load_tool_definitions(server_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every MCP tool definition, keyed by its unique name."""
    server_dir = server_dir.resolve()
    if not server_dir.is_dir():
        raise NotADirectoryError(f"Tool server directory does not exist: {server_dir}")
    definitions = asyncio.run(_load_tools_async(server_dir))
    return {definition["name"]: definition for definition in definitions}


def dataset_tools_field(
    dataset_name: str,
    definitions: dict[str, dict[str, Any]],
) -> str:
    """Return the instructed dataset-specific ShareGPT ``tools`` JSON string."""
    requested = DATASET_TOOL_NAMES.get(dataset_name)
    if requested is None:
        raise ValueError(
            f"Unsupported dataset directory '{dataset_name}'. "
            f"Expected one of: {', '.join(sorted(DATASET_TOOL_NAMES))}"
        )
    missing = [name for name in requested if name not in definitions]
    if missing:
        raise ValueError(
            f"Tool definition(s) missing for {dataset_name}: {', '.join(missing)}"
        )
    selected = [definitions[name] for name in requested]
    return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


class Qwen35ContextCounter:
    """Measure contexts exactly as LLaMA Factory's ``qwen3_5`` template does."""

    def __init__(
        self,
        model_name: str,
        image_max_pixels: int,
        image_min_pixels: int,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self._image_lengths: dict[str, int] = {}
        self._tools_prompts: dict[str, str] = {}
        self._system_lengths: dict[tuple[str, str], int] = {}

    def _image_token_length(self, image_path: str) -> int:
        cached = self._image_lengths.get(image_path)
        if cached is not None:
            return cached

        with Image.open(image_path) as image:
            width, height = image.size

        pixels = width * height
        if pixels > self.image_max_pixels:
            scale = math.sqrt(self.image_max_pixels / pixels)
            width, height = int(width * scale), int(height * scale)
        elif pixels < self.image_min_pixels:
            scale = math.sqrt(self.image_min_pixels / pixels)
            width, height = int(width * scale), int(height * scale)

        # Match LLaMA Factory's Qwen2VLPlugin._preprocess_image, which Qwen3VLPlugin
        # inherits and the qwen3_5 template reuses. These guards keep thin
        # tool-generated crops valid.
        width, height = max(width, 28), max(height, 28)
        if width / height > 200:
            width = height * 180
        if height / width > 200:
            height = width * 180

        patch_size = int(self.image_processor.patch_size)
        merge_size = int(self.image_processor.merge_size)
        try:
            resized_height, resized_width = qwen_vl_smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=int(self.image_processor.size["shortest_edge"]),
                max_pixels=int(self.image_processor.size["longest_edge"]),
            )
        except ValueError as exc:
            raise ReformatError(
                f"cannot calculate Qwen3.5 tokens for {image_path}: {exc}"
            ) from exc
        image_length = (
            resized_height // patch_size
            * (resized_width // patch_size)
            // (merge_size**2)
        )
        self._image_lengths[image_path] = image_length
        return image_length

    def _tools_prompt(self, tools_field: str) -> str:
        cached = self._tools_prompts.get(tools_field)
        if cached is not None:
            return cached
        tools = json.loads(tools_field)
        tool_text = ""
        for tool in tools:
            # qwen3_5 unwraps the OpenAI function envelope instead of adding one.
            unwrapped = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_text += "\n" + json.dumps(unwrapped, ensure_ascii=False)
        prompt = QWEN35_TOOL_PROMPT.format(tool_text=tool_text)
        self._tools_prompts[tools_field] = prompt
        return prompt

    def _system_length(self, system: str, tools_field: str) -> int:
        # Identical for every record of a dataset, so it is tokenized once.
        key = (system, tools_field)
        cached = self._system_lengths.get(key)
        if cached is None:
            chunk = f"<|im_start|>system\n{system}{self._tools_prompt(tools_field)}<|im_end|>\n"
            cached = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            self._system_lengths[key] = cached
        return cached

    def count(self, item: dict[str, Any]) -> int:
        """Return the complete untruncated SFT sequence length.

        Equal to tokenizing every chunk with each ``<image>`` expanded to
        ``vision_start + image_token * n + vision_end``, without building or
        tokenizing those expansions: the three are special tokens, which the
        tokenizer splits out before BPE, so an expansion is exactly ``n + 2``
        tokens and the text around it tokenizes the same on its own.
        """
        conversations, images = item["conversations"], item["images"]
        placeholders = sum(message["value"].count("<image>") for message in conversations)
        if placeholders > len(images):
            raise ReformatError("more <image> placeholders than image paths")
        if placeholders < len(images):
            raise ReformatError("more image paths than <image> placeholders")

        texts: list[str] = []
        for message in conversations:
            if message["from"] == "human":
                chunk = f"<|im_start|>user\n{message['value']}<|im_end|>\n<|im_start|>assistant\n"
            elif message["from"] == "gpt":
                chunk = f"{message['value']}<|im_end|>\n"
            else:
                raise ReformatError(
                    f"qwen3_5 context counter cannot encode role {message['from']!r}"
                )
            texts.extend(part for part in chunk.split("<image>") if part)

        image_tokens = sum(self._image_token_length(path) + 2 for path in images)
        # One batched call per record; the Rust tokenizer encodes the batch in parallel.
        text_tokens = sum(
            len(ids) for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        )
        return self._system_length(item["system"], item["tools"]) + text_tokens + image_tokens


def _balanced_braced(text: str, open_brace: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index], index + 1
    return None


def extract_boxed(text: str) -> str | None:
    """Extract the final balanced ``\boxed{...}`` content."""
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(positions):
        open_brace = text.find("{", start)
        balanced = _balanced_braced(text, open_brace)
        if balanced is not None:
            return balanced[0].strip()
    return None


def escape_image_placeholder(text: str) -> str:
    """Prevent source text from being mistaken for a multimodal placeholder."""
    return re.sub(r"<image\s*>", "&lt;image&gt;", text, flags=re.IGNORECASE)


def sanitize_final_answer(record: dict[str, Any]) -> str:
    """Return only a canonical ``<answer>\boxed{...}</answer>`` block."""
    for message in reversed(record.get("result") or []):
        content = message.get("content") if isinstance(message, dict) else None
        response = content.get("llm_content") if isinstance(content, dict) else None
        if not isinstance(response, str):
            continue
        tags = ANSWER_TAG_RE.findall(response)
        if not tags:
            continue
        answer = extract_boxed(tags[-1])
        if answer:
            answer = escape_image_placeholder(answer)
            return f"<answer>\n\\boxed{{{answer}}}\n</answer>"
    raise ReformatError("no valid <answer>\\boxed{...}</answer> final response")


def sanitize_reasoning(value: Any) -> str:
    """Normalize an existing reasoning field for one ``<think>`` block."""
    reasoning = str(value or "").strip()
    outer = THINK_TAG_RE.fullmatch(reasoning)
    if outer:
        reasoning = outer.group(1).strip()
    reasoning = SPECIAL_TAG_RE.sub("", reasoning).strip()
    reasoning = escape_image_placeholder(reasoning)
    return reasoning or FALLBACK_REASONING


def format_assistant(reasoning: str, payload: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{payload}"


def render_tool_call(call: dict[str, Any]) -> str:
    """Render one call in the XML syntax of LLaMA Factory's ``qwen3_5`` tool format."""
    name = call["name"]
    if STRUCTURAL_TAG_RE.search(name):
        raise ReformatError(f"tool name is not XML-safe: {name!r}")

    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in call["arguments"].items():
        if not isinstance(key, str) or STRUCTURAL_TAG_RE.search(key):
            raise ReformatError(f"argument name is not XML-safe: {key!r}")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        if STRUCTURAL_TAG_RE.search(value):
            raise ReformatError(
                f"argument {key!r} contains a tool-call tag and cannot be encoded"
            )
        lines.append(f"<parameter={key}>")
        lines.append(escape_image_placeholder(value))
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def parse_tool_call(content: dict[str, Any]) -> dict[str, Any]:
    """Build a canonical tool call from the original model-emitted arguments."""
    calls = content.get("tool_call")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ReformatError("tool trace must contain exactly one tool call")

    function = calls[0].get("function")
    if not isinstance(function, dict):
        raise ReformatError("tool call has no function object")

    tool_name = content.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name or tool_name == "None":
        raise ReformatError("tool trace has no valid resolved tool name")

    raw_arguments = function.get("arguments", "{}")
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            try:
                arguments = ast.literal_eval(raw_arguments)
            except (SyntaxError, ValueError) as exc:
                raise ReformatError(f"invalid tool arguments: {exc}") from exc
    else:
        raise ReformatError("tool arguments must be an object or serialized object")

    if not isinstance(arguments, dict):
        raise ReformatError("tool arguments must resolve to an object")
    arguments = {
        key: value
        for key, value in arguments.items()
        if key not in HOST_INJECTED_ARGUMENTS
    }
    return {"name": tool_name, "arguments": arguments}


def _resolve_artifact(record: dict[str, Any], content: dict[str, Any]) -> Path:
    """Resolve and validate an image-producing tool's persisted artifact."""
    tool_name = content["tool_name"]
    base_value = record.get("tool_results_dir")
    if not isinstance(base_value, str) or not base_value:
        raise ReformatError(f"{tool_name} is missing tool_results_dir")
    base = Path(base_value).resolve()

    result_value = content.get("tool_call_result")
    if not isinstance(result_value, str) or not result_value:
        raise ReformatError(f"{tool_name} has no image result path")
    candidate = Path(result_value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()

    if not candidate.is_relative_to(base):
        raise ReformatError(f"{tool_name} image is outside tool_results_dir")
    if not candidate.is_file():
        raise ReformatError(f"{tool_name} image does not exist: {candidate}")
    return candidate


def _escape_observation(text: str) -> str:
    escaped = (
        text.replace("<observation>", "&lt;observation&gt;")
        .replace("</observation>", "&lt;/observation&gt;")
        .replace("<tool_response>", "&lt;tool_response&gt;")
        .replace("</tool_response>", "&lt;/tool_response&gt;")
        .strip()
    )
    escaped = STRUCTURAL_TAG_RE.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        escaped,
    )
    return escape_image_placeholder(escaped)


def format_observation(
    record: dict[str, Any],
    content: dict[str, Any],
    images: list[str],
) -> str:
    """Wrap only the tool result in a human ``<tool_response>`` message."""
    tool_name = content["tool_name"]
    if tool_name in IMAGE_RESULT_TOOLS:
        artifact = _resolve_artifact(record, content)
        images.append(str(artifact))
        return "<tool_response>\n<image>\n</tool_response>"

    result = _escape_observation(str(content.get("tool_call_result") or ""))
    if not result:
        raise ReformatError(f"{tool_name} has an empty observation")
    return f"<tool_response>\n{result}\n</tool_response>"


def convert_record(
    record: dict[str, Any],
    tools_field: str,
) -> tuple[dict[str, Any], bool]:
    """Convert one filtered rollout into a ShareGPT item."""
    record_id = record.get("id")
    query = record.get("query")
    image_path = record.get("image_path")
    if not isinstance(record_id, str) or not record_id:
        raise ReformatError("record has no id")
    if not isinstance(query, str) or not query.strip():
        raise ReformatError(f"{record_id}: record has no query")
    if not isinstance(image_path, str) or not Path(image_path).is_file():
        raise ReformatError(f"{record_id}: input image does not exist: {image_path}")

    images = [str(Path(image_path).resolve())]
    conversations: list[dict[str, str]] = [
        {
            "from": "human",
            "value": f"<image>\n{escape_image_placeholder(query.strip())}",
        }
    ]
    pending_reasoning: list[str] = []
    used_tool = False

    for message in record.get("result") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, dict):
            continue

        if "llm_reasoning" in content:
            next_reasoning = sanitize_reasoning(content["llm_reasoning"])
            pending_reasoning.append(next_reasoning)
            continue

        if "tool_call" not in content:
            continue

        call = parse_tool_call(content)
        reasoning = "\n\n".join(pending_reasoning)
        conversations.append(
            {
                "from": "gpt",
                "value": format_assistant(reasoning, render_tool_call(call)),
            }
        )
        conversations.append(
            {
                "from": "human",
                "value": format_observation(record, content, images),
            }
        )
        pending_reasoning.clear()
        used_tool = True

    final_reasoning = "\n\n".join(pending_reasoning)
    conversations.append(
        {
            "from": "gpt",
            "value": format_assistant(final_reasoning, sanitize_final_answer(record)),
        }
    )

    return (
        {
            "id": record_id,
            "system": SYSTEM_PROMPT,
            "tools": tools_field,
            "conversations": conversations,
            "images": images,
        },
        used_tool,
    )


def _dataset_seed(seed: int, directory: Path) -> int:
    digest = hashlib.sha256(f"{seed}:{directory.name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def balance_counts(
    tool_total: int,
    no_tool_total: int,
    max_class_ratio: float,
) -> tuple[int, int]:
    """Pick per-class counts whose imbalance stays within ``max_class_ratio``.

    The minority class is kept whole and the majority class is truncated to at
    most ``max_class_ratio`` times its size. A perfect 1:1 mixture is therefore
    not required: with the default ratio of 2.0 anything between 1:2 and
    2:1 is accepted, and more skewed inputs are trimmed back to that bound.
    """
    minority = min(tool_total, no_tool_total)
    if minority == 0:
        return (0, 0)

    majority_limit = int(minority * max_class_ratio)
    return (min(tool_total, majority_limit), min(no_tool_total, majority_limit))


def reformat_file(
    source: Path,
    seed: int,
    tools_field: str,
    context_counter: Qwen35ContextCounter,
    max_context_length: int,
    max_class_ratio: float = DEFAULT_MAX_CLASS_RATIO,
) -> dict[str, Any]:
    """Convert and balance one dataset's filtered records."""
    tool_items: list[tuple[dict[str, Any], int]] = []
    no_tool_items: list[tuple[dict[str, Any], int]] = []
    skipped: list[tuple[str, str]] = []
    overlong = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            try:
                converted, used_tool = convert_record(record, tools_field)
                context_length = context_counter.count(converted)
            except ReformatError as exc:
                skipped.append((str(record.get("id") or line_number), str(exc)))
                continue
            if context_length >= max_context_length:
                overlong += 1
                continue
            (tool_items if used_tool else no_tool_items).append(
                (converted, context_length)
            )

    rng = random.Random(_dataset_seed(seed, source.parent))
    rng.shuffle(tool_items)
    rng.shuffle(no_tool_items)
    tool_count, no_tool_count = balance_counts(
        len(tool_items),
        len(no_tool_items),
        max_class_ratio,
    )
    if tool_count == 0 or no_tool_count == 0:
        raise ValueError(
            f"{source}: cannot make a mixture within 1:{max_class_ratio:g} "
            f"(tool={len(tool_items)}, no_tool={len(no_tool_items)})"
        )

    selected = tool_items[:tool_count] + no_tool_items[:no_tool_count]
    rng.shuffle(selected)

    output = source.parent / OUTPUT_FILENAME
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item, _ in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(output)

    for record_id, reason in skipped:
        print(f"[reformat] skipped {source.parent.name}/{record_id}: {reason}", file=sys.stderr)

    return {
        "dataset": source.parent.name,
        "input": str(source),
        "output": str(output),
        "available_tool": len(tool_items),
        "available_no_tool": len(no_tool_items),
        "selected_tool": tool_count,
        "selected_no_tool": no_tool_count,
        "written": len(selected),
        "skipped": len(skipped),
        "overlong": overlong,
        "max_selected_context_length": max(
            (length for _, length in selected), default=0
        ),
    }


def discover_inputs(input_dir: Path) -> list[Path]:
    direct = input_dir / INPUT_FILENAME
    if direct.is_file():
        return [direct]
    files = sorted(input_dir.rglob(INPUT_FILENAME))
    if not files:
        raise FileNotFoundError(f"No {INPUT_FILENAME} found under {input_dir}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert filtered CaViR rollouts to balanced multimodal ShareGPT "
            "reformatted_results.jsonl files for the qwen3_5 template."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "A single dataset directory, or any parent of them: every "
            f"{INPUT_FILENAME} found below it is converted and its "
            f"{OUTPUT_FILENAME} is written next to it."
        ),
    )
    parser.add_argument(
        "--server-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "server",
        help="Directory containing the MCP *_tool.py server files.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LENGTH,
        help="Exclude samples whose complete tokenized context is at least this long.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=DEFAULT_IMAGE_MAX_PIXELS,
        help="LLaMA Factory image_max_pixels used by the training configuration.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=DEFAULT_IMAGE_MIN_PIXELS,
        help="LLaMA Factory image_min_pixels used by the training configuration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-class-ratio",
        type=float,
        default=DEFAULT_MAX_CLASS_RATIO,
        help=(
            "Largest tolerated tool:no_tool imbalance. 1.0 forces an exact 1:1 "
            "mixture; the default 2.0 accepts anything between 1:2 and 2:1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.max_class_ratio < 1.0:
        raise ValueError("--max-class-ratio must be at least 1.0")
    if args.max_context_length <= 0:
        raise ValueError("--max-context-length must be greater than zero")
    if args.image_min_pixels <= 0 or args.image_max_pixels < args.image_min_pixels:
        raise ValueError("image pixel limits must satisfy 0 < min <= max")

    definitions = load_tool_definitions(args.server_dir)
    context_counter = Qwen35ContextCounter(
        args.model_name,
        args.image_max_pixels,
        args.image_min_pixels,
    )
    summaries = []
    failures: list[tuple[str, str]] = []
    for source in discover_inputs(input_dir):
        dataset = source.parent.name
        try:
            tools_field = dataset_tools_field(dataset, definitions)
            summaries.append(
                reformat_file(
                    source,
                    args.seed,
                    tools_field,
                    context_counter,
                    args.max_context_length,
                    args.max_class_ratio,
                )
            )
        except (ValueError, OSError) as exc:
            # One unusable dataset must not discard the datasets that follow it.
            failures.append((dataset, str(exc)))
    for summary in summaries:
        print(
            f"[reformat] {summary['dataset']}: wrote {summary['written']} "
            f"(tool={summary['selected_tool']}, "
            f"no_tool={summary['selected_no_tool']}), "
            f"overlong={summary['overlong']}, skipped={summary['skipped']}, "
            f"max_tokens={summary['max_selected_context_length']} -> "
            f"{summary['output']}"
        )
    for dataset, reason in failures:
        print(f"[reformat] FAILED {dataset}: {reason}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()