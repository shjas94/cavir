"""Generate with the cold start policy served by vLLM, and play a trajectory out to its final answer.

Prompts are rendered by ``prompting.render`` (the training format) and sent through the chat endpoint with a
pass-through chat template, so the server adds nothing: text parts are copied verbatim and each image part
becomes one ``<|vision_start|><|image_pad|><|vision_end|>`` span. Images are resized here the way LLaMA
Factory resized them for SFT (``image_max_pixels`` 262,144), so the server's processor sees the same image.

A state is {system, tools, turns, images, partial}: ``turns`` alternates user and assistant messages
(tool results are user turns), ``images`` are file paths in placeholder order ([0] = the question image),
``partial`` is the assistant message being written. ``branch_rollout.py`` plays states out to an answer.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import math
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from openai import APIConnectionError, APIError, AsyncOpenAI, InternalServerError
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_construction"))

from filter_out import answers_match, normalize_answer  # noqa: E402
from prompting import ANSWER_RE, VISION_SPAN, parse_tool_call, render  # noqa: E402
from reformat_dataset import extract_boxed  # noqa: E402

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}   # as in selection_following.py
IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS = 262144, 32 * 32           # LLaMA Factory defaults used in SFT
SEGMENT_MAX_TOKENS = 8192   # room for long code-tool calls
MAX_TOOL_CALLS = 3                                             # host.MAX_TOOL_CALLS
# Images here are local files: dataset images and tool outputs (visualize_data plots can be huge
# from a large figsize/dpi). They are shrunk to the SFT budget below, so PIL's bomb guard is lifted.
Image.MAX_IMAGE_PIXELS = None
PASSTHROUGH_TEMPLATE = (
    "{%- for message in messages %}{%- for part in message['content'] %}"
    "{%- if part['type'] == 'text' %}{{- part['text'] }}"
    "{%- else %}{{- '<|vision_start|><|image_pad|><|vision_end|>' }}{%- endif %}"
    "{%- endfor %}{%- endfor %}"
)


@lru_cache(maxsize=4096)
def image_data_uri(path: str) -> str:
    """The image as LLaMA Factory's qwen2_vl plugin preprocessed it, PNG data URI."""
    with Image.open(path) as image:
        image.load()
        if image.width * image.height > IMAGE_MAX_PIXELS:
            factor = math.sqrt(IMAGE_MAX_PIXELS / (image.width * image.height))
            image = image.resize((int(image.width * factor), int(image.height * factor)))
        if image.width * image.height < IMAGE_MIN_PIXELS:
            factor = math.sqrt(IMAGE_MIN_PIXELS / (image.width * image.height))
            image = image.resize((int(image.width * factor), int(image.height * factor)))
        image = image.convert("RGB") if image.mode != "RGB" else image
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_content(text: str, images: list[str]) -> list[dict]:
    pieces = text.split(VISION_SPAN)
    if len(pieces) - 1 != len(images):
        raise ValueError(f"{len(pieces) - 1} image placeholders but {len(images)} images")
    content = []
    for index, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if index < len(images):
            content.append({"type": "image_url", "image_url": {"url": image_data_uri(images[index])}})
    return content


def grade(text: str, answer: str) -> tuple[str | None, bool]:
    """Boxed prediction from the last ``<answer>`` in ``text`` and whether it matches the normalized answer."""
    tags = ANSWER_RE.findall(text)
    boxed = extract_boxed(tags[-1]) if tags else None
    if boxed is None:
        return None, False
    return boxed, answers_match(normalize_answer(boxed), answer)[0]


POLICY_DIR = Path(__file__).resolve().parents[1] / "sft/LlamaFactory/saves/qwen35-2b/full/cavir_coldstart_v2_noarith_fixtools"


class ServerUnavailable(Exception):
    """The server could not be reached, dropped the connection while streaming, or failed on its side:
    a 5xx response or an error event inside the stream."""


class Policy:
    """Client of the served policy.

    The text is decoded here from the generated token ids (returned through ``logprobs`` with
    ``return_tokens_as_token_ids``) instead of read from ``message.content``: server-side tool or reasoning
    parsers (e.g. ``--enable-auto-tool-choice``) remove ``<tool_call>`` blocks from the content.

    ``concurrency`` counts sequences running on the server, not requests: a request with n
    samples takes n slots and gives each back as soon as that sample finishes (streamed), so a new request can
    start while the longest sample of an earlier one is still running.
    """

    def __init__(self, base_url: str, model: str, concurrency: int = 64):
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)
        self.model = model
        self.slots = self.free = concurrency
        self.changed = asyncio.Condition()
        self.tokenizer = AutoTokenizer.from_pretrained(str(POLICY_DIR))

    async def acquire(self, n: int) -> int:
        """Wait for n free slots (at most all of them) and take them; returns how many were taken."""
        n = min(n, self.slots)
        async with self.changed:
            await self.changed.wait_for(lambda: self.free >= n)
            self.free -= n
        return n

    async def close(self) -> None:
        """Close the client before the event loop ends. openai leaves an inner stream generator behind after [DONE];
        finalized while the loop is shutting down it logs "error occurred during closing of asynchronous generator".
        Collecting it here, with the loop still running, finalizes it quietly."""
        await self.client.close()
        gc.collect()
        await asyncio.sleep(0.1)

    async def release(self, n: int) -> None:
        async with self.changed:
            self.free += n
            self.changed.notify_all()

    async def generate(self, state: dict, *, n: int = 1, stop: list[str], max_tokens: int = SEGMENT_MAX_TOKENS,
                       seed: int | None = None, logit_bias: dict[str, float] | None = None,
                       held: int = 0) -> list[dict]:
        """Continue ``state['partial']``; returns n {text, finish_reason, tokens}. The stop string is kept.

        ``logit_bias`` maps token ids (as strings) to a bias added to their logits; -100 bans a token.
        ``held``: slots the caller already took with ``acquire`` for this request (0 = take them here)."""
        taken = held or await self.acquire(n)
        ids, finish = [[] for _ in range(n)], [None] * n
        stream = None
        try:
            text = render(state["system"], state["tools"], state["turns"], state["partial"])
            stream = await self.client.chat.completions.create(
                model=self.model, n=n, max_tokens=max_tokens, stop=stop, seed=seed,
                **({"logit_bias": logit_bias} if logit_bias else {}),
                temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                messages=[{"role": "user", "content": request_content(text, state["images"])}],
                logprobs=True, stream=True,
                extra_body={"top_k": SAMPLING["top_k"], "chat_template": PASSTHROUGH_TEMPLATE,
                            "chat_template_content_format": "openai", "add_generation_prompt": False,
                            "include_stop_str_in_output": True, "skip_special_tokens": False,
                            "add_special_tokens": False, "return_tokens_as_token_ids": True},
            )
            async for chunk in stream:
                for choice in chunk.choices:
                    if choice.logprobs and choice.logprobs.content:
                        ids[choice.index] += [int(t.token.removeprefix("token_id:")) for t in choice.logprobs.content]
                    if choice.finish_reason and finish[choice.index] is None:
                        finish[choice.index] = choice.finish_reason
                        if taken:
                            taken -= 1
                            await self.release(1)
        except Exception as exc:
            # A drop mid-stream surfaces as the HTTP library's own error, not as openai's APIConnectionError; a
            # server-side failure inside the stream is a plain APIError. Request errors (400 and other 4xx) stay as they are.
            if (isinstance(exc, (APIConnectionError, InternalServerError)) or type(exc) is APIError
                    or type(exc).__module__.startswith("httpx")):
                raise ServerUnavailable(f"{type(exc).__name__}: {exc}") from exc
            raise
        finally:
            if stream is not None:  # release the connection now, not when the garbage collector gets to it
                try:
                    await stream.close()
                except Exception:
                    pass
            await self.release(taken)
        outputs = []
        for choice_ids, reason in zip(ids, finish):
            text = self.tokenizer.decode(choice_ids, skip_special_tokens=False).removesuffix("<|im_end|>")
            outputs.append({"text": text, "finish_reason": reason, "tokens": len(choice_ids)})
        return outputs
