"""
AgentFly - A Hierarchical AI Agent System

This module implements a hierarchical AI system with two main components:
1. META-PLANNER: Breaks down high-level questions into executable tasks
2. EXECUTOR: Executes individual tasks using available tools

The system uses OpenAI models and MCP (Model Context Protocol) for tool integration.
"""

from __future__ import annotations
import ast
import asyncio
import argparse
import base64
import contextlib
import copy
import json
import logging
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import colorlog
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor

from dataset import HFDataset, collate_fn, smart_resize

DATASET_PATHS = {
    "docvqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/docvqa/docvqa.jsonl",
    "chartqa_1": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_1/chartqa_1_cognitive.jsonl",
    "chartqa_2": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_2/chartqa_2_cognitive.jsonl",
    "infographics_vqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-VLM-Dataset-v2/snapshots/51f4f4d219315c3283950994d4eb3d7fc30aa87b/infographicsvqa_cot/infographicsvqa_cot_trimmed.jsonl",
    "plotqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/mulberry_1/mulberry_plotqa_reasoning.jsonl",
    "gqa_1": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/gqa_1/gqa_1.jsonl",
    "gqa_2": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/gqa_2/gqa_2.jsonl",
    "geometry3k": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/geometry3k/geometry3k.jsonl",
    "geomverse_cot": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-VLM-Dataset-v2/snapshots/51f4f4d219315c3283950994d4eb3d7fc30aa87b/geomverse_cot/geomverse_cot.jsonl",
    # Nemotron-Image-Training-v3.5; images from fetch_v35_images.py
    "hopchain_fintabnet": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3.5/snapshots/70b27e021e64bd97641597dc1f127a208b58be0e/hopchain_fintabnet/hopchain_fintabnet.jsonl",
    "chartnet_qa_01_difficulty": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3.5/snapshots/70b27e021e64bd97641597dc1f127a208b58be0e/chartnet_qa_01_difficulty/chartnet_qa_01_difficulty.jsonl",
    "coco_count_spatial_qa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3.5/snapshots/70b27e021e64bd97641597dc1f127a208b58be0e/coco_count_spatial_qa/coco_count_spatial_qa.jsonl",
}

LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3

# Data-loading pipeline sizing. Both are kept small on purpose: every prepared
# sample holds a multi-MB base64 image, so deep prefetch buffers translate
# directly into resident memory.
LOADER_WORKERS = int(os.getenv("LOADER_WORKERS", "8"))
QUEUE_MAXSIZE = int(os.getenv("LOADER_QUEUE_MAXSIZE", "16"))

# Tools that operate on an image. Instead of accepting raw base64 image bytes
# from the model (expensive in context length and prone to hallucination),
# these tools are exposed with an `image_index` parameter: 0 is always the
# original input image, and every successful call to a tool in
# IMAGE_PRODUCING_TOOLS appends a newly produced image to the per-task history
# (1 = first such image, 2 = second, ...). The host resolves `image_index` to
# the actual base64 bytes before invoking the tool.
IMAGE_INPUT_TOOLS = {"crop_zoomin", "mark_dots", "draw_bbox", "crop_ocr_image", "imaginate_editing"}
DIRECT_IMAGE_PRODUCING_TOOLS = {"crop_zoomin", "mark_dots", "draw_bbox"}
CODE_VISUALIZATION_TOOLS = {"visualize_data"}
IMAGE_PRODUCING_TOOLS = DIRECT_IMAGE_PRODUCING_TOOLS | CODE_VISUALIZATION_TOOLS
# Image tools that call back into the serving model and therefore need the
# host's connection settings injected.
MODEL_BACKED_TOOLS = {"crop_ocr_image", "imaginate_editing"}
# Tools whose workspace the host owns: it injects a per-task `task_cache_dir`
# so generated code, execution logs and result images land under the run's
# output directory instead of a path the model has to invent.
CODE_EXECUTION_TOOLS = {"math_compute", "visualize_data"}
# Fields that the host injects itself and must never be generated by the
# model; they are stripped from the schema shown to the model.
HOST_INJECTED_FIELDS = {"image", "model_name", "api_key", "host_url", "task_cache_dir"}
# Argument keys under which a model may hallucinate raw image payloads.
IMAGE_PAYLOAD_KEYS = ("image", "image_base64", "img", "img_b64", "image_data")

# Vision-token accounting used by LLMBackend._count_tokens. Counting tokens by
# letting the multimodal processor tokenize the real conversation is extremely
# expensive: it base64-decodes every image in the history and turns it into a
# float32 patch tensor on EVERY turn (and up to 32 of those run concurrently in
# the default thread pool), which is the dominant source of memory growth.
# The image token count is therefore derived analytically from the image's
# pixel dimensions instead, the same way Qwen-VL's processor computes it.
IMAGE_PATCH_SIZE = 14
IMAGE_MERGE_SIZE = 2
# Used when an image's dimensions cannot be read cheaply (non-PNG payload).
DEFAULT_IMAGE_TOKENS = 1024


GENERATION_PROMPT = """
Given a QUESTION and an INPUT_IMAGE, answer the question.

Tool use:
- Call a tool when it gives something you cannot get reliably on your own: small or dense text, a detail too fine to read at this resolution, objects to count, the positions of several objects or how they are placed relative to each other, or a calculation that is error-prone to do mentally.
- When a key fact is genuinely uncertain, resolve it with the tool that fits instead of guessing, and base your next reasoning step on its result.
- Do not call a tool only to double-check an answer you are already confident about; answer directly instead.

Generate the final answer in short-form with following format.
<answer>\\boxed{...}</answer>
"""


# ---------------------------------------------------------------------------
#   Image / argument helpers
# ---------------------------------------------------------------------------
def _image_index_property(tool_name: str) -> Dict[str, Any] | None:
    """
    Synthetic `image_index` schema property for a tool, or None if it takes no image.

    The raw image bytes are never asked from the model: image-consuming tools
    instead select an image by its index in the per-task image history, which
    the host resolves before invoking the tool.
    """
    if tool_name not in IMAGE_INPUT_TOOLS:
        return None
    return {
        "type": "integer",
        "description": (
            "Index of the image to operate on, NOT the raw image data. "
            "0 is the original input image. Each successful call to crop_zoomin, "
            "mark_dots, or draw_bbox appends a new image to the history "
            "(1 = first such image, 2 = second, etc.); use that index to operate "
            "on a previously produced image instead of the original."
        ),
        "default": 0,
    }


def _resolve_image_index(t_args: Dict[str, Any], image_history: List[str]) -> int:
    """
    Pop `image_index` from tool arguments and resolve it to a valid history index.

    Args:
        t_args: Tool arguments emitted by the model (mutated: `image_index` is removed).
        image_history: Per-task image history (index 0 = original input image).

    Returns:
        The resolved index; out-of-range or non-integer values fall back to 0.
    """
    try:
        index = int(t_args.pop("image_index", 0))
    except (TypeError, ValueError):
        return 0
    return index if 0 <= index < len(image_history) else 0



def _raw_b64(image_ref: str) -> str:
    """Return the raw base64 payload of a possibly data-URI-prefixed string."""
    return image_ref.split(",", 1)[1] if "," in image_ref else image_ref


def _png_dimensions(image_ref: str) -> tuple[int, int] | None:
    """
    Read (width, height) from a PNG data-URI / base64 payload by decoding only
    its first few bytes (signature + IHDR), so the full image is never
    materialized in memory just to learn its size.
    """
    head = _raw_b64(image_ref)[:64]
    head = head[: len(head) // 4 * 4]
    try:
        raw = base64.b64decode(head)
    except ValueError:
        return None
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def _estimate_image_tokens(image_ref: str) -> int:
    """Estimate an image's prompt token cost without running the processor."""
    size = _png_dimensions(image_ref)
    if size is None or size[0] <= 0 or size[1] <= 0:
        return DEFAULT_IMAGE_TOKENS
    width, height = size
    try:
        h_bar, w_bar = smart_resize(height, width, factor=IMAGE_PATCH_SIZE * IMAGE_MERGE_SIZE)
    except ValueError:
        return DEFAULT_IMAGE_TOKENS
    return (h_bar // IMAGE_PATCH_SIZE) * (w_bar // IMAGE_PATCH_SIZE) // (IMAGE_MERGE_SIZE ** 2)


def _file_b64(image_path: str) -> str:
    """Raw base64 of an image file at its original resolution (no re-encoding)."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _fit_to_image_budget(b64_string: str) -> str:
    """
    Resize a tool-produced image to the same token budget as the input image
    and return it as PNG base64.

    Tools read the full-resolution original, so a crop of a large region, or
    an annotated full page, can be far larger than what the model is allowed
    to see. Sizing it like the input keeps a crop's gain in detail (a small
    region fills the budget) without handing the model the whole page at full
    resolution.
    """
    with Image.open(BytesIO(base64.b64decode(_raw_b64(b64_string)))) as image:
        try:
            height, width = smart_resize(image.height, image.width, factor=IMAGE_PATCH_SIZE * IMAGE_MERGE_SIZE)
        except ValueError:  # extreme aspect ratio: leave the image as the tool made it
            return _raw_b64(b64_string)
        resized = image.resize((width, height))
    with BytesIO() as buffer:
        resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _base64_to_pil_and_save(b64_string: str, save_path: str) -> Image.Image:
    """
    Decode a base64 image string, save it to `save_path` and return an
    in-memory copy (the source buffer is closed deterministically).
    """
    image_bytes = base64.b64decode(_raw_b64(b64_string).strip())
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as pil_image:
        pil_image.load()
        pil_image.save(save_path)
        return pil_image.copy()


def _sanitize_tool_call_arguments(raw_arguments: str) -> str:
    """
    Strip any raw image payload the model may have hallucinated into a tool
    call's arguments before it is persisted into conversation history, so a
    single bad generation doesn't bloat every subsequent turn's context.
    """
    if not raw_arguments:
        return raw_arguments
    try:
        parsed = ast.literal_eval(raw_arguments)
    except (SyntaxError, ValueError):
        return raw_arguments
    if not isinstance(parsed, dict):
        return raw_arguments
    if not any(key in parsed for key in IMAGE_PAYLOAD_KEYS):
        return raw_arguments
    return json.dumps({k: v for k, v in parsed.items() if k not in IMAGE_PAYLOAD_KEYS})


def _parse_tool_payload(payload: str) -> Dict[str, Any]:
    """
    Parse a structured tool result into a mapping.

    The tool server serializes dict results as JSON, so JSON is tried first;
    `ast.literal_eval` stays as a fallback for Python-repr style payloads. A
    non-mapping payload is rejected so callers keep their error handling.
    """
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"expected an object, got {type(parsed).__name__}")
    raise ValueError(f"unparsable tool result: {payload}")


def _code_tool_feedback(
    parsed_result: Dict[str, Any], tool_name: str = "code tool"
) -> str:
    """
    Model-facing text of a code-execution result — never empty.

    When a code tool rejects a script (contract violation, forbidden import,
    filename/write failure) nothing is executed, so `execution_result` is empty
    and the reason lives in `file_write_log`. Forwarding the empty string would
    leave the model without any feedback and it would repeat the same mistake,
    so the rejection reason is surfaced instead.
    """
    execution_result = (parsed_result.get("execution_result") or "").strip()
    execution_error = (parsed_result.get("execution_error") or "").strip()
    write_log = (parsed_result.get("file_write_log") or "").strip()

    if execution_result:
        # The script ran: its output already contains the failure details.
        return execution_result
    # Nothing ran: the write log carries the actionable rejection message.
    return write_log or execution_error or f"{tool_name} produced no output."


def _tool_call_with_parsed_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shallow copy of a tool call whose `function.arguments` is a mapping.

    Tool call arguments travel as a JSON string (the OpenAI wire format), but
    the local chat template iterates them with `|items`, so they must be parsed
    before rendering. Used for token counting only.
    """
    new_call = dict(tool_call)
    function = new_call.get("function")
    if not isinstance(function, dict):
        return new_call
    function = dict(function)
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                function["arguments"] = parse(arguments)
                break
            except (SyntaxError, ValueError):
                continue
    new_call["function"] = function
    return new_call


# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class LLMBackend:
    """
    Backend that talks to a local OpenAI-compatible server (e.g., your vLLM at
    http://localhost:port/v1) and returns tool_calls parsed from the stream.
    """

    def __init__(
        self,
        model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4",
        server_url: str | None = None,
        api_key: str | None = None,
        tokenizer_name_or_path: str | None = None,
    ):
        self.model = model
        self.server_url = server_url or os.getenv("DIRECT_BASE_URL", "http://localhost:port/v1")
        self.api_key = api_key or os.getenv("DIRECT_API_KEY", "EMPTY")

        # A single client (and its httpx connection pool) is reused across
        # every chat() call; creating one per call would leak sockets/memory.
        self.client = AsyncOpenAI(base_url=self.server_url, api_key=self.api_key)

        # Tokens are counted locally instead of via the server's /tokenize
        # endpoint: that would be a blocking HTTP round trip on every turn, adding
        # load to the engine on top of generation traffic. The processor is
        # only used to render the chat template (tokenize=False); the text
        # tokenizer does the counting, so no image is ever decoded.
        tokenizer_path = tokenizer_name_or_path or os.getenv("TOKENIZER_NAME_OR_PATH") or self.model
        self.processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    async def aclose(self):
        """Close the underlying HTTP client/connection pool."""
        await self.client.close()

    @staticmethod
    def _normalize_messages_for_template(msg: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """
        Build a lightweight, image-free copy of the conversation for token
        counting, and return it together with the number of tokens the removed
        images would contribute.

        Two normalizations are required by the chat template: plain-string
        `content` becomes a list of content parts (the template iterates it, so
        a string would be iterated character by character), and tool call
        arguments are parsed from their JSON string form into a mapping.

        Every "image_url" part is DROPPED and replaced by an analytic token
        estimate: feeding the real images to the processor would decode each of
        them into a float32 patch tensor on every single turn, which dominates
        the process's memory usage. Only the count matters here.
        """
        normalized: List[Dict[str, Any]] = []
        image_tokens = 0
        for message in msg:
            # Shallow copy only: with images removed there is no multimodal
            # content left for the template to rewrite in place.
            new_message = dict(message)
            content = message.get("content")

            if isinstance(content, str):
                new_message["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_tokens += _estimate_image_tokens((part.get("image_url") or {}).get("url", ""))
                    else:
                        parts.append(part)
                new_message["content"] = parts

            if message.get("tool_calls"):
                new_message["tool_calls"] = [
                    _tool_call_with_parsed_arguments(tc) for tc in message["tool_calls"]
                ]

            normalized.append(new_message)
        return normalized, image_tokens

    def _count_tokens(self, msg: List[Dict[str, Any]], tools_schema) -> int:
        """
        Count the tokens the chat template would produce for the given
        messages/tools, mirroring the server's /tokenize endpoint but without
        an HTTP round trip and without ever decoding an image.
        """
        normalized, image_tokens = self._normalize_messages_for_template(msg)
        rendered = self.processor.apply_chat_template(
            normalized,
            tools=tools_schema,
            add_generation_prompt=True,
            tokenize=False,
            # Template-only flags must be passed as top-level kwargs;
            # "chat_template_kwargs" is a vLLM server-side param name that
            # apply_chat_template does not understand.
            enable_thinking=True,
        )
        if isinstance(rendered, (list, tuple)):
            rendered = rendered[0]

        input_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if input_ids and isinstance(input_ids[0], list):  # unwrap batch dimension
            input_ids = input_ids[0]
        return len(input_ids) + image_tokens

    async def adjust_max_length(self, messages, max_tokens, tools_schema) -> int:
        """
        Return how many tokens are left for generation (0 if the prompt alone
        already exceeds `max_tokens`).
        """
        # Tokenization is CPU-bound; run it in a worker thread so it doesn't
        # block the event loop while other concurrent tasks are waiting on I/O.
        total = await asyncio.to_thread(self._count_tokens, messages, tools_schema)
        return max(0, max_tokens - total)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 32768,
    ) -> Dict[str, Any]:

        max_tokens = await self.adjust_max_length(messages=messages, max_tokens=max_tokens, tools_schema=tools)
        if max_tokens == 0:
            return {"content": None, "reasoning": None, "tool_calls": None, "finish_reason": "context_length_exceeded"}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "tool_choice": tool_choice or "auto",
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"preserve_thinking": True}
            }
        }
        if tools:
            payload["tools"] = tools

        stream = await self.client.chat.completions.create(**payload)

        aggregated_content = ""
        aggregated_reasoning = ""
        tool_call_buffers: Dict[int, Dict[str, Any]] = {}
        finish_reason = None

        try:
            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

                if getattr(delta, "content", None):  # content outside <think> block
                    aggregated_content += delta.content

                if getattr(delta, "reasoning", None):  # corresponds to thinking process
                    aggregated_reasoning += delta.reasoning

                for tc in getattr(delta, "tool_calls", None) or []:
                    buffer = tool_call_buffers.setdefault(
                        getattr(tc, "index", 0) or 0,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if getattr(tc, "id", None):
                        buffer["id"] = tc.id
                    func = getattr(tc, "function", None)
                    if getattr(func, "name", None):
                        buffer["function"]["name"] = func.name
                    if getattr(func, "arguments", None):
                        buffer["function"]["arguments"] += func.arguments
        finally:
            # Release the streaming response's HTTP connection/buffers instead
            # of relying on GC, which can lag after retried/partial streams.
            close = getattr(stream, "close", None)
            if close is not None:
                await close()

        return {
            "content": aggregated_content.strip() if aggregated_content else None,
            "reasoning": aggregated_reasoning.strip() if aggregated_reasoning else None,
            "tool_calls": list(tool_call_buffers.values()) or None,
            "finish_reason": finish_reason,
        }


# ---------------------------------------------------------------------------
#   CaViRAgent
# ---------------------------------------------------------------------------
def _tool_name_from_decorator(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    """Return the MCP tool name declared on a function, if any."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "mcp"
        ):
            continue

        for keyword in decorator.keywords:
            if (
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return keyword.value.value
        return node.name
    return None


def discover_tool_servers(server_dir: str | Path) -> Dict[str, Path]:
    """
    Discover ``@mcp.tool()`` definitions without importing server modules.

    Parsing the source avoids starting every server merely to learn which file
    owns a requested tool. Duplicate tool names are rejected because selection
    by name would otherwise be ambiguous.
    """
    directory = Path(server_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Tool server directory does not exist: {directory}")

    discovered: Dict[str, Path] = {}
    for script in sorted(directory.glob("*_tool.py")):
        try:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Could not inspect tool server {script}: {exc}") from exc

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            tool_name = _tool_name_from_decorator(node)
            if tool_name is None:
                continue
            previous = discovered.get(tool_name)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate tool name '{tool_name}' in {previous} and {script}."
                )
            discovered[tool_name] = script

    if not discovered:
        raise ValueError(f"No @mcp.tool() definitions found in {directory}")
    return discovered


class CaViRAgent:
    """
    Main client class that orchestrates the Agent system.

    Handles tool connections, and processes user queries through
    multiple generations and tool calling cycles.
    """

    def __init__(self):
        self.llm_agent = LLMBackend(
            model=os.getenv("MODEL_NAME", "rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"),
            server_url=os.getenv("DIRECT_BASE_URL"),
            api_key=os.getenv("DIRECT_API_KEY"),
        )
        self.sessions: Dict[str, ClientSession] = {}
        # The tool schema is identical for every sample; built once and cached.
        self._tools_schema_cache: List[Dict[str, Any]] | None = None

    def _resolve_tool_name(self, requested: str) -> str:
        """Map a model-emitted tool name onto a connected tool (exact, then fuzzy)."""
        if requested in self.sessions:
            return requested
        for name in self.sessions:
            if requested in name or name in requested:
                return name
        raise KeyError(f"No matching tool for '{requested}'. Available: {list(self.sessions.keys())}")

    # ---------- Tool management ----------
    async def connect_to_tools(
        self, tool_names: List[str], server_dir: str | Path
    ) -> None:
        """
        Start only the servers needed by ``tool_names`` and expose those tools.

        Raises:
            ValueError: If no tools are requested or a requested name is unknown.
            RuntimeError: If a server does not expose a tool discovered in its source.
        """
        from contextlib import AsyncExitStack

        if not tool_names:
            raise ValueError("At least one tool name must be provided.")
        duplicate_requests = sorted(
            name for name in set(tool_names) if tool_names.count(name) > 1
        )
        if duplicate_requests:
            raise ValueError(
                f"Duplicate requested tool names: {', '.join(duplicate_requests)}"
            )

        discovered = discover_tool_servers(server_dir)
        unknown = [name for name in tool_names if name not in discovered]
        if unknown:
            raise ValueError(
                f"Unknown tool(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(discovered))}"
            )

        # Preserve the order from run.sh while starting each required process
        # only once when several selected tools share the same server file.
        scripts: Dict[Path, List[str]] = {}
        for name in tool_names:
            scripts.setdefault(discovered[name], []).append(name)

        self.exit_stack = AsyncExitStack()

        for script, selected_names in scripts.items():
            params = StdioServerParameters(
                command=sys.executable, args=[str(Path(script))]
            )
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()

            server_tools = {
                tool.name: tool for tool in (await session.list_tools()).tools
            }
            missing = [name for name in selected_names if name not in server_tools]
            if missing:
                raise RuntimeError(
                    f"{script} did not expose discovered tool(s): {', '.join(missing)}"
                )

            for name in selected_names:
                self.sessions[name] = session

        print("Connected tools:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        """
        Build the OpenAI function-calling schema for all connected tools.

        Fields the host injects itself (raw "image" bytes, model_name, api_key,
        host_url, task_cache_dir) are stripped so the model can never be
        asked/tempted to generate them. Tools that need an image instead get a
        synthetic "image_index" integer parameter (see IMAGE_INPUT_TOOLS /
        IMAGE_PRODUCING_TOOLS) that the host resolves against the per-task
        image history before invoking the tool.

        The result never changes between samples, so it is cached.
        """
        if self._tools_schema_cache is not None:
            return self._tools_schema_cache

        result: List[Dict[str, Any]] = []
        # self.sessions maps every tool name to its session, so the same
        # session appears once per tool it exposes; list each session once.
        for session in dict.fromkeys(self.sessions.values()):
            for tool in (await session.list_tools()).tools:
                # A selected tool may share its server process with tools that
                # were not requested in run.sh. Keep those unselected sibling
                # tools out of the model-facing schema.
                if self.sessions.get(tool.name) is not session:
                    continue
                schema = copy.deepcopy(tool.input_schema) or {}
                properties = schema.get("properties", {})
                required = [r for r in schema.get("required", []) if r not in HOST_INJECTED_FIELDS]
                for field in HOST_INJECTED_FIELDS:
                    properties.pop(field, None)

                image_index_property = _image_index_property(tool.name)
                if image_index_property is not None:
                    properties["image_index"] = image_index_property

                schema["properties"] = properties
                schema["required"] = required
                result.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })

        self._tools_schema_cache = result
        return result

    # ---------- Main processing ----------
    async def process_query(self, args, query: str, img_b64: str, image_path: str, task_id: str, gt: str,
                            source_id: str, rollout: int, semaphore: asyncio.Semaphore | None = None):
        """
        Args:
            query: User's question or request
            img_b64: Optional base64 string of an image to include in the query
            image_path: Optional path to an image file to include in the query
            task_id: Unique identifier for this query session (source_id, or source_id__r<k> with --rollouts > 1)
            source_id: The dataset sample id; rollout: which of its --rollouts generations (from 0)
            gt: Ground truth answer for evaluation purposes
            semaphore: Optional asyncio semaphore for controlling concurrency.
                When concurrency is already bounded by the caller (e.g. a fixed
                worker pool), this can be left as None.
        """
        # Fall back to a no-op context manager when no semaphore is supplied so
        # concurrency isn't double-limited by both a semaphore and the caller's
        # worker pool.
        async with (semaphore or contextlib.nullcontext()):
            tools_schema = await self._tools_schema()
            tool_result_dir = f"{args.output_dir}/tool_results/{task_id}"
            os.makedirs(tool_result_dir, exist_ok=True)
            # Created lazily by the code-execution server, only if the model
            # actually runs code for this task.
            code_exec_dir = f"{args.output_dir}/code_execution/{task_id}"

            # The input image's data URI is built once and shared by reference
            # between the conversation history and image_history, so its base64
            # payload is never held in memory more than once. Raw base64
            # (needed by the image tools) is derived transiently via _raw_b64.
            input_image_url = f"data:image/png;base64,{img_b64}"
            llm_agent_message: List[Dict[str, Any]] = [
                {"role": "system", "content": GENERATION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"QUESTION:\n{query}\nINPUT_IMAGE:\n"},
                        {"type": "image_url", "image_url": {"url": input_image_url}},
                    ],
                },
            ]
            # Input fields are stored once at the top level of the output
            # record. `result` contains only generated assistant/tool traces.
            result_message: List[Dict[str, Any]] = []
            result_idx = 0
            # Per-task image history: index 0 is always the original input
            # image; every successful image-producing tool call appends its
            # result. Entries are the SAME data-URI string objects stored in
            # the conversation messages (shared by reference).
            image_history: List[str] = [input_image_url]
            # Full-resolution original for tools, loaded only if a tool uses it.
            original_image_b64: str | None = None

            def append_result(entry: Dict[str, Any]):
                nonlocal result_idx
                entry["idx"] = result_idx
                result_message.append(entry)
                result_idx += 1

            def sanitize_args(call_args: Dict[str, Any], image_index: Any = None) -> Dict[str, Any]:
                sanitized = dict(call_args)
                if "image" in sanitized:
                    sanitized["image"] = f"<image_index={image_index}>" if image_index is not None else "<image data omitted>"
                return sanitized

            def record_tool(call, call_id, tool_name, args_repr, trace_result, reason,
                            message_content=None, **extra):
                """
                Emit the tool message the model sees and the matching trace
                entry. Every tool_call in an assistant message MUST be followed
                by a tool message with the same id, otherwise the server
                rejects the next request.
                """
                llm_agent_message.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": trace_result if message_content is None else message_content,
                })
                append_result({
                    "role": "assistant",
                    "content": {
                        "tool_call": [call],
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                        "tool_call_args": args_repr,
                        "tool_call_result": trace_result,
                        "finish_reason": reason,
                        **extra,
                    },
                })

            # Main generating and tool-calling loop
            tool_called = 0
            while True:
                stop_generation = False
                llm_agent_reply = await self.llm_agent.chat(llm_agent_message, tools_schema)
                llm_content = llm_agent_reply["content"]
                llm_reasoning = llm_agent_reply["reasoning"]
                finish_reason = llm_agent_reply["finish_reason"]
                tool_calls = llm_agent_reply["tool_calls"]

                for call in tool_calls or []:
                    # Stable id so the assistant message and its matching tool
                    # messages agree, and no hallucinated image payload is
                    # persisted into the conversation history.
                    call["id"] = call.get("id") or str(uuid.uuid4())
                    call["function"]["arguments"] = _sanitize_tool_call_arguments(
                        call["function"].get("arguments", "")
                    )

                # A SINGLE assistant message per turn: one model generation =
                # reasoning + content + all tool_calls. Splitting these into
                # separate messages breaks the chat template and the
                # reasoning<->tool-call association.
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": llm_content or ""}
                if llm_reasoning:
                    assistant_msg["reasoning"] = llm_reasoning
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                llm_agent_message.append(assistant_msg)

                # for text inside <think>. corresponds to reasoning process
                if llm_reasoning:
                    append_result({
                        "role": "assistant",
                        "content": {"llm_reasoning": llm_reasoning, "finish_reason": finish_reason},
                    })

                for call in tool_calls or []:
                    call_id = call["id"]
                    t_name = call["function"]["name"]
                    raw_args = call["function"].get("arguments", "{}")

                    tool_called += 1
                    if tool_called > MAX_TOOL_CALLS:
                        finish_reason = "Tool call limit exceeded"
                        record_tool(call, call_id, "None", "None",
                                    "Exceeded maximum tool calls limit.", finish_reason)
                        stop_generation = True
                        continue

                    try:
                        t_args = ast.literal_eval(raw_args)  # validate json
                        if not isinstance(t_args, dict):
                            # Anything but an object would break the
                            # host-injected fields below and the call itself.
                            raise ValueError(f"expected an object, got {type(t_args).__name__}")
                    except (SyntaxError, ValueError) as e:
                        finish_reason = "error"
                        record_tool(call, call_id, t_name, raw_args,
                                    f"[tool argument parse error] {e}: {raw_args}", finish_reason)
                        stop_generation = True
                        continue

                    try:
                        resolved = self._resolve_tool_name(t_name)
                    except KeyError as e:
                        finish_reason = "error"
                        record_tool(call, call_id, t_name, str(sanitize_args(t_args)),
                                    f"[tool resolution error] {e}", finish_reason)
                        stop_generation = True
                        continue

                    # Everything below keys off the RESOLVED tool name, not the
                    # name the model emitted: _resolve_tool_name also accepts
                    # approximate names (e.g. "math_compute_tool" -> "math_compute"),
                    # and keying off the requested name would silently skip the
                    # host-injected image / task_cache_dir arguments.
                    used_image_index = None
                    if resolved in IMAGE_INPUT_TOOLS:
                        # Resolve `image_index` (0 = original input image, N =
                        # the Nth produced image) to the base64 bytes the tool
                        # needs; out-of-range or non-integer falls back to 0.
                        used_image_index = _resolve_image_index(t_args, image_history)
                        if used_image_index == 0:
                            # The model only sees a downscaled input; tools read
                            # the original file so zooming and OCR can recover
                            # detail the model cannot perceive on its own.
                            if original_image_b64 is None:
                                original_image_b64 = await asyncio.to_thread(_file_b64, image_path)
                            t_args["image"] = original_image_b64
                        else:
                            t_args["image"] = _raw_b64(image_history[used_image_index])
                        if resolved in MODEL_BACKED_TOOLS:
                            t_args["model_name"] = os.getenv("MODEL_NAME", "rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm")
                            t_args["api_key"] = os.getenv("DIRECT_API_KEY", "")
                            t_args["host_url"] = os.getenv("DIRECT_BASE_URL", "")
                    if resolved in CODE_EXECUTION_TOOLS:
                        t_args["task_cache_dir"] = code_exec_dir

                    result_msg = await self.sessions[resolved].call_tool(resolved, t_args)

                    save_path = None
                    extra: Dict[str, Any] = {}
                    message_content = None
                    tool_error = getattr(result_msg, "is_error", False)
                    first = result_msg.content[0]

                    # Coordinates the tool actually applied (a reversed or
                    # rounded box is not the one the model emitted), reported
                    # through the result's `_meta` and traced next to the raw
                    # arguments. Never shown to the model.
                    for meta_key, meta_value in (getattr(first, "meta", None) or {}).items():
                        if meta_key.startswith("applied_") or meta_key == "output_size":
                            extra[meta_key] = meta_value

                    if resolved in DIRECT_IMAGE_PRODUCING_TOOLS:
                        if tool_error or not hasattr(first, "data"):
                            # Tool failed internally; `first` is a TextContent error
                            tool_result = getattr(first, "text", "Unknown image tool error")
                            tool_error = True
                        else:
                            # ImageContent.data is base64 at the tool's own size.
                            tool_result = await asyncio.to_thread(_fit_to_image_budget, first.data)
                            save_path = f"{tool_result_dir}/{resolved}_{call_id}.png"
                            _base64_to_pil_and_save(tool_result, save_path)
                            # Store the produced image once as a data URI and
                            # share that same object with the tool message, so
                            # later calls can reference it by index without a
                            # second copy of its payload. The produced image is
                            # surfaced as a real image content-part: a base64
                            # string in plain-text tool content would NOT be
                            # decoded by the vision encoder.
                            image_history.append(f"data:image/png;base64,{tool_result}")
                            new_index = len(image_history) - 1
                            message_content = [
                                {"type": "text", "text": f"[image_index={new_index}] Produced image is shown below."},
                                {"type": "image_url", "image_url": {"url": image_history[new_index]}},
                            ]
                    elif resolved in MODEL_BACKED_TOOLS:
                        tool_result = first.text
                    elif resolved in CODE_VISUALIZATION_TOOLS:
                        try:
                            parsed_result = _parse_tool_payload(first.text)
                            if parsed_result.get("file_write_log"):
                                extra["code_file_log"] = parsed_result["file_write_log"]
                            if parsed_result.get("script_path"):
                                extra["code_script_path"] = parsed_result["script_path"]
                            if parsed_result.get("plot_path"):
                                extra["code_plot_path"] = parsed_result["plot_path"]

                            execution_error = parsed_result.get("execution_error", "")
                            if execution_error:
                                tool_error = True
                                extra["code_execution_error"] = execution_error

                            plot_content = next(
                                (
                                    content
                                    for content in result_msg.content[1:]
                                    if hasattr(content, "data")
                                ),
                                None,
                            )
                            feedback = _code_tool_feedback(parsed_result, resolved)
                            if tool_error or plot_content is None:
                                if not tool_error:
                                    tool_error = True
                                    extra["code_execution_error"] = (
                                        "visualize_data returned no plot image"
                                    )
                                    feedback = (
                                        f"{feedback}\n"
                                        "❌ visualize_data returned no plot image."
                                    )
                                tool_result = feedback
                            else:
                                tool_result = plot_content.data
                                save_path = parsed_result["plot_path"]
                                image_history.append(
                                    f"data:image/png;base64,{tool_result}"
                                )
                                new_index = len(image_history) - 1
                                message_content = [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"{feedback}\n"
                                            f"[image_index={new_index}] "
                                            "Generated visualization is shown below."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_history[new_index]
                                        },
                                    },
                                ]
                        except (SyntaxError, ValueError, AttributeError, KeyError):
                            tool_result = (
                                f"Error parsing visualize_data result: "
                                f"{getattr(first, 'text', first)}"
                            )
                            tool_error = True
                    elif resolved in CODE_EXECUTION_TOOLS:
                        try:
                            parsed_result = _parse_tool_payload(first.text)
                            # Keep the write log out of the model's context but
                            # in the trace, so the generated .py file that
                            # produced this output stays traceable.
                            if parsed_result.get("file_write_log"):
                                extra["code_file_log"] = parsed_result["file_write_log"]
                            if parsed_result.get("script_path"):
                                extra["code_script_path"] = parsed_result["script_path"]
                            # Never an empty message: a rejected script produces
                            # no execution output, and the model still needs the
                            # reason to fix its next call.
                            tool_result = _code_tool_feedback(parsed_result, resolved)
                            # The MCP call itself succeeds even when the
                            # generated script crashes, so the failure has to be
                            # taken from the tool payload; otherwise a failed
                            # run is traced with finish_reason None.
                            execution_error = parsed_result.get("execution_error", "")
                            if execution_error:
                                tool_error = True
                                extra["code_execution_error"] = execution_error
                        except (SyntaxError, ValueError, AttributeError):
                            tool_result = f"Error parsing tool result: {first.text}"
                            tool_error = True
                    else:
                        # Plain-text tools (e.g. the math calculators): the
                        # result is already the scalar the model asked for, and
                        # errors arrive as text with is_error set.
                        tool_result = getattr(first, "text", str(first))

                    record_tool(
                        call,
                        call_id,
                        resolved,
                        str(sanitize_args(t_args, used_image_index)),
                        save_path or tool_result,
                        "tool result error" if tool_error else None,
                        message_content=message_content,
                        **extra,
                    )

                # for text after </think>. corresponds to final answer
                if llm_content:
                    append_result({
                        "role": "assistant",
                        "content": {"llm_content": llm_content, "finish_reason": finish_reason},
                    })

                if stop_generation:
                    break
                if finish_reason is not None and finish_reason not in ["tool_calls", "Tool call limit exceeded"]:
                    break

            async with aiofiles.open(os.path.join(args.output_dir, args.output_json), "a") as f:
                await f.write(json.dumps({
                    "id": task_id,
                    "source_id": source_id,
                    "rollout": rollout,
                    "query": query,
                    "image_path": image_path,
                    "gt": gt,
                    "result": result_message,
                    "tool_results_dir": tool_result_dir,
                    "code_execution_dir": code_exec_dir,
                    "finish_reason": finish_reason
                }, default=str) + "\n")
                await f.flush()

    async def cleanup(self):
        """Clean up resources and close tool server connections."""
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()
        await self.llm_agent.aclose()


def parse_args():
    parser = argparse.ArgumentParser(description="CaViR Dataset Construction Host")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset to construct")
    parser.add_argument("--concurrency", type=int, default=64, help="Number of concurrent queries to process")
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        required=True,
        help="MCP tool names to enable; owning server files are discovered automatically",
    )
    parser.add_argument(
        "--server-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "server"),
        help="Directory containing *_tool.py MCP server modules",
    )
    parser.add_argument("--output-json", type=str, required=True, help="Path to output JSON file for the constructed dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save intermediate tool call results and reasoning traces "
             "(default: outputs/<output-name> next to this script)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of the output directory under outputs/ (default: the dataset name); ignored when --output-dir is given",
    )
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to process (default: -1 (all))")
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Text file with one dataset id per line; only these samples are rolled out",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Generations per sample. With more than one, rollout k of sample ID is recorded as ID__r<k> (k from 0); "
             "every record also holds source_id and rollout",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        name = args.output_name or args.dataset_name
        args.output_dir = str(Path(__file__).resolve().parent / "outputs" / name)
    # Saved tool-result paths are built from this directory and reformat reads them as written,
    # so a relative --output-dir would yield paths that do not resolve later.
    args.output_dir = str(Path(args.output_dir).resolve())
    return args


async def main_async(args):
    load_dotenv()
    # Results are appended, so a second run into the same directory would mix two rollouts.
    output_path = os.path.join(args.output_dir, args.output_json)
    if os.path.exists(output_path):
        raise SystemExit(f"{output_path} already exists; use a new --output-dir")
    dataset = HFDataset(args.dataset_name, DATASET_PATHS[args.dataset_name])
    if args.ids_file:
        wanted = set(Path(args.ids_file).read_text().split())
        indices = [index for index, item_id in enumerate(dataset.item_ids()) if item_id in wanted]
        if len(indices) != len(wanted):
            raise SystemExit(f"{len(wanted) - len(indices)} ids in {args.ids_file} are not in {args.dataset_name}")
        dataset = Subset(dataset, indices)
    agent = CaViRAgent()
    await agent.connect_to_tools(args.tools, args.server_dir)

    # The CPU-bound image decode/resize/base64 work in __getitem__ runs in
    # DataLoader worker processes so it never blocks the event loop. The worker
    # count is deliberately small and decoupled from --concurrency: every
    # in-flight sample holds a multi-MB base64 image and each worker forks a
    # full copy of this process, while preprocessing is far faster than
    # generation.
    num_workers = min(LOADER_WORKERS, max(1, (os.cpu_count() or 2)))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    total_samples = (len(loader) if args.sample == -1 else min(len(loader), args.sample)) * args.rollouts

    # Producer/consumer: one producer pulls prepared batches from the
    # DataLoader (each blocking `next` in a worker thread) into a bounded
    # queue, and a fixed pool of `concurrency` workers drains it, so data
    # loading and generation overlap instead of alternating in waves. The queue
    # is capped at a small absolute number of samples because each queued item
    # pins a multi-MB base64 image in memory.
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    async def producer():
        loader_iter = iter(loader)
        produced = 0
        try:
            while args.sample == -1 or produced < args.sample:
                batch = await asyncio.to_thread(next, loader_iter, None)
                if batch is None:  # dataset exhausted
                    break
                for k in range(args.rollouts):
                    await queue.put((batch[0], k))
                produced += 1
        finally:
            # One sentinel per worker so every consumer terminates cleanly.
            for _ in range(args.concurrency):
                await queue.put(None)

    with tqdm(total=total_samples, desc="Processing dataset") as pbar:
        async def worker():
            while True:
                entry = await queue.get()
                try:
                    if entry is None:
                        return
                    item, k = entry
                    await agent.process_query(
                        args,
                        item['question'],
                        item['image'],
                        item['image_path'],
                        item['id'] if args.rollouts == 1 else f"{item['id']}__r{k}",
                        item['gt'],
                        item['id'],
                        k,
                    )
                    pbar.update(1)
                except Exception:
                    logger.exception("process_query failed for a sample")
                finally:
                    queue.task_done()

        try:
            workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            await asyncio.gather(producer(), *workers)
        finally:
            await agent.cleanup()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args=args))
