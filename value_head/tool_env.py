"""Run a policy's tool call exactly as ``host.py`` ran tools for the cold start rollouts.

Same MCP servers and the same host-side handling (index 0 reads the original image file at full
resolution, produced images are fitted to the input image budget, model-backed tools call the teacher
from ``dataset_construction/.env``, code tools get a per-call workspace, failures are shown to the model
as text). The observation text follows the training format (``reformat_dataset.format_observation``):
an image result is ``<tool_response>\\n<image>\\n</tool_response>``, a text result is escaped.

Unlike ``build_cavir_coldstart.execute_tool``, index 0 is the full-resolution file, as in the rollouts.
"""

from __future__ import annotations

import base64
import math
import os
import sys
import uuid
from contextlib import AsyncExitStack
from io import BytesIO
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset_construction"
sys.path.insert(0, str(DATA_DIR))

from dotenv import load_dotenv  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from PIL import Image  # noqa: E402

from host import (  # noqa: E402
    CODE_EXECUTION_TOOLS, CODE_VISUALIZATION_TOOLS, DIRECT_IMAGE_PRODUCING_TOOLS, IMAGE_INPUT_TOOLS,
    MODEL_BACKED_TOOLS, _code_tool_feedback, _file_b64, _fit_to_image_budget, _parse_tool_payload,
    _resolve_image_index, discover_tool_servers,
)
from reformat_dataset import HOST_INJECTED_ARGUMENTS, _escape_observation  # noqa: E402

IMAGE_OBSERVATION = "<tool_response>\n<image>\n</tool_response>"

load_dotenv(DATA_DIR / ".env")


class ToolEnv:
    """MCP sessions for a dataset's tools; use as ``async with ToolEnv(names) as env``.

    Each tool server handles one call at a time (code execution takes about a second), so ``copies``
    processes of every server are started and calls are spread over them round robin.
    """

    def __init__(self, tool_names: list[str], copies: int = 1, max_pixels: int | None = None, model: dict | None = None):
        """``max_pixels`` keeps produced images at the size the tool made them unless they exceed it (e.g.
        the evaluated model's own processor limit) instead of fitting them to the cold start budget;
        ``model`` ({model_name, host_url, api_key}) sends model-backed tools to that server instead of the teacher."""
        self.tool_names = tool_names
        self.copies = copies
        self.max_pixels = max_pixels
        self.model = model
        self.sessions: dict[str, list[ClientSession]] = {}
        self.turn: dict[str, int] = {}

    async def __aenter__(self) -> "ToolEnv":
        self.stack = AsyncExitStack()
        discovered = discover_tool_servers(DATA_DIR / "server")
        scripts: dict[Path, list[str]] = {}
        for name in self.tool_names:
            scripts.setdefault(discovered[name], []).append(name)
        for script, names in scripts.items():
            for _ in range(self.copies):
                params = StdioServerParameters(command=sys.executable, args=[str(script)], cwd=str(script.parent))
                streams = await self.stack.enter_async_context(stdio_client(params))
                session = await self.stack.enter_async_context(ClientSession(*streams))
                await session.initialize()
                for name in names:
                    self.sessions.setdefault(name, []).append(session)
        return self

    def session(self, name: str) -> ClientSession:
        index = self.turn.get(name, 0)
        self.turn[name] = index + 1
        return self.sessions[name][index % len(self.sessions[name])]

    async def __aexit__(self, *exc) -> None:
        await self.stack.aclose()

    async def run(self, call: dict, images: list[str], work_dir: Path, allowed: list[str] | None = None) -> dict:
        """Execute ``call`` ({name, arguments}); ``images`` (paths, [0] = original) gains any produced image.

        ``allowed`` restricts the call to the dataset's tool set. Returns {observation, image (path or None),
        error (str or None), meta}.
        """
        # The servers run with cwd = their own folder, so a relative work_dir would put their outputs there.
        work_dir = Path(work_dir).resolve()
        name = call["name"]
        allowed = allowed or self.tool_names
        if name not in self.sessions or name not in allowed:
            return _text_result(f"Unknown tool '{name}'. Available: {', '.join(allowed)}", error="unknown_tool")
        arguments = {k: v for k, v in call["arguments"].items() if k not in HOST_INJECTED_ARGUMENTS}
        if name in IMAGE_INPUT_TOOLS:
            index = _resolve_image_index(arguments, images)
            arguments["image"] = _file_b64(images[index])
            if name in MODEL_BACKED_TOOLS:
                arguments.update(self.model or {"model_name": os.getenv("MODEL_NAME", ""), "api_key": os.getenv("DIRECT_API_KEY", ""),
                                                "host_url": os.getenv("DIRECT_BASE_URL", "")})
        if name in CODE_EXECUTION_TOOLS:
            arguments["task_cache_dir"] = str(work_dir / "code_execution" / uuid.uuid4().hex[:12])

        result = await self.session(name).call_tool(name, arguments)
        if not result.content:
            return _text_result(f"{name} returned no output", error="tool_error")
        first = result.content[0]
        error = bool(getattr(result, "is_error", False))
        meta = {k: v for k, v in (getattr(first, "meta", None) or {}).items()
                if k.startswith("applied_") or k == "output_size"}

        if name in DIRECT_IMAGE_PRODUCING_TOOLS:
            if error or not hasattr(first, "data"):
                return _text_result(getattr(first, "text", "Unknown image tool error"), error="tool_error", meta=meta)
            path = work_dir / "tool_results" / f"{name}_{uuid.uuid4().hex[:12]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.max_pixels is None:
                path.write_bytes(base64.b64decode(_fit_to_image_budget(first.data)))
            else:
                _save_within(base64.b64decode(first.data), self.max_pixels, path)
            images.append(str(path))
            return {"observation": IMAGE_OBSERVATION, "image": str(path), "error": None, "meta": meta}
        if name in CODE_VISUALIZATION_TOOLS:
            try:
                payload = _parse_tool_payload(first.text)
            except (SyntaxError, ValueError, AttributeError):
                return _text_result(f"Error parsing visualize_data result: {getattr(first, 'text', first)}", error="tool_error")
            feedback = _code_tool_feedback(payload, name)
            plot = next((part for part in result.content[1:] if hasattr(part, "data")), None)
            if payload.get("execution_error") or plot is None or not payload.get("plot_path"):
                return _text_result(feedback if plot is not None else f"{feedback}\n❌ visualize_data returned no plot image.",
                                    error="tool_error")
            images.append(str(payload["plot_path"]))
            return {"observation": IMAGE_OBSERVATION, "image": str(payload["plot_path"]), "error": None, "meta": meta}
        if name in CODE_EXECUTION_TOOLS:
            try:
                payload = _parse_tool_payload(first.text)
            except (SyntaxError, ValueError, AttributeError):
                return _text_result(f"Error parsing tool result: {first.text}", error="tool_error")
            return _text_result(_code_tool_feedback(payload, name), error="tool_error" if payload.get("execution_error") else None)
        return _text_result(getattr(first, "text", str(first)), error="tool_error" if error else None, meta=meta)


def _save_within(png: bytes, max_pixels: int, path: Path) -> None:
    """Write the PNG as is, or shrunk to ``max_pixels`` (aspect kept) when larger: the server would shrink it anyway."""
    with Image.open(BytesIO(png)) as image:
        if image.width * image.height <= max_pixels:
            path.write_bytes(png)
            return
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS).save(path, format="PNG")


def _text_result(text: str, error: str | None = None, meta: dict | None = None) -> dict:
    escaped = _escape_observation(text or "") or "(empty tool output)"
    return {"observation": f"<tool_response>\n{escaped}\n</tool_response>", "image": None, "error": error, "meta": meta or {},
            "text": text}
