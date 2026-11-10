import math
from PIL import Image, ImageDraw
import io
import base64
import mcp.types as mcp_types
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from openai import AsyncOpenAI
from collections.abc import Sequence
from typing import Annotated, Any, Dict
from pydantic import Field, WithJsonSchema

"""
Tool server for cropping and zooming an image based on a specified crop box and expansion factor.
"""

IMAGINATION_PROMPT = """
You are an image edit simulation model that imagines the result of editing an image based on a given edit operation. 
You do not perform actual image editing, but instead provide a detailed description of what the edited image would look like after applying the specified edit operation to the input image.
Given the Edit Operation and Input Image, describe the detailed description of the expected result image after applying the edit operation. 
"""

OCR_PROMPT = """
You are an OCR model that extracts text from images.
"""


mcp = MCPServer("image_processing")

InjectedImage = Annotated[
    str,
    Field(
        description=(
            "Host-injected base64 image selected by image_index. "
            "The model must never generate or pass this value."
        )
    ),
]
Coordinate1000 = Annotated[
    float,
    WithJsonSchema({"type": "number", "minimum": 0.0, "maximum": 1000.0}),
]
Box1000 = Annotated[
    tuple[
        Coordinate1000,
        Coordinate1000,
        Coordinate1000,
        Coordinate1000,
    ],
    Field(
        description=(
            "Bounding box [x1, y1, x2, y2] in the 0-to-1000 normalized coordinate "
            "system. (x1, y1) is the top-left corner and (x2, y2) is the "
            "bottom-right corner. x increases left-to-right and y increases "
            "top-to-bottom: x=0 is the left edge, x=1000 is the right edge, "
            "y=0 is the top edge, and y=1000 is the bottom edge. Never use "
            "pixel coordinates."
        )
    ),
]
Point1000 = Annotated[
    tuple[Coordinate1000, Coordinate1000],
    Field(
        description=(
            "Exactly two coordinates [x, y] normalized to the 0-to-1000 range. "
            "The origin [0, 0] is the top-left corner; x increases left-to-right "
            "and y increases top-to-bottom. [1000, 1000] is the bottom-right "
            "corner. Never use pixel coordinates."
        )
    ),
]
Boxes1000 = Annotated[
    list[Box1000],
    Field(
        min_length=1,
        description=(
            "One or more bounding boxes in [x1, y1, x2, y2] format, where "
            "(x1, y1) is top-left and (x2, y2) is bottom-right, using the "
            "0-to-1000 normalized coordinate system."
        ),
    ),
]
Points1000 = Annotated[
    list[Point1000],
    Field(
        min_length=1,
        description=(
            "Centers of one or more visible red markers as [x, y] points in "
            "the 0-to-1000 normalized coordinate system."
        ),
    ),
]
PositiveScale = Annotated[
    float,
    Field(
        gt=0,
        description=(
            "Resize factor for the cropped image: 1 keeps its size, values above "
            "1 enlarge it, and values between 0 and 1 shrink it."
        ),
    ),
]
EditOperation = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "A concrete hypothetical edit to describe, such as 'remove the red "
            "legend' or 'change the blue bar to green'."
        ),
    ),
]
InjectedModelName = Annotated[
    str, Field(description="Host-injected model name. Never provide this argument.")
]
InjectedApiKey = Annotated[
    str, Field(description="Host-injected API key. Never provide this argument.")
]
InjectedHostUrl = Annotated[
    str, Field(description="Host-injected model endpoint. Never provide this argument.")
]

# Coordinates arrive from a language model, so 1000 is occasionally emitted
# with tiny floating-point noise; absorb it instead of rejecting a valid box.
_COORDINATE_SCALE = 1000.0
_COORD_TOLERANCE = 1e-6

# Cache AsyncOpenAI clients keyed by (host_url, api_key) so repeated tool
# calls reuse the same underlying HTTP connection pool instead of leaking a
# new one (and its sockets/buffers) on every invocation.
_openai_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _get_openai_client(api_key: str, host_url: str) -> AsyncOpenAI:
    key = (host_url, api_key)
    client = _openai_clients.get(key)
    if client is None:
        client = AsyncOpenAI(api_key=api_key, base_url=host_url)
        _openai_clients[key] = client
    return client


def _decode_image(image: str, mode: str | None = None) -> Image.Image:
    """
    Decode the host-injected base64 payload into a fully loaded PIL image.

    A malformed payload is a failure the tool anticipates, so it is reported as
    a ToolError (message reaches the model) instead of a crash (message is
    replaced by a generic "Error executing tool ..." the model cannot act on).
    """
    try:
        pil_image = Image.open(io.BytesIO(base64.b64decode(image)))
        pil_image.load()
    except Exception as exc:
        raise ToolError(f"could not decode the `image` payload: {exc}") from exc
    return pil_image.convert(mode) if mode else pil_image


def _coordinates_1000(
    values: Sequence[float], expected_len: int, param_name: str, image: Image.Image
) -> list[float]:
    """
    Validate model-native coordinates normalized to the 0-to-1000 range.

    These coordinates are resolution-independent: x=500 is the horizontal
    midpoint and y=500 is the vertical midpoint for every image size.
    """
    width, height = image.size
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != expected_len:
        raise ToolError(
            f"`{param_name}` must be a list of {expected_len} numbers, got {values!r}."
        )

    coords: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ToolError(
                f"`{param_name}` must contain {expected_len} finite numbers, got {values!r}."
            )
        coords.append(float(value))

    if any(
        coord < -_COORD_TOLERANCE
        or coord > _COORDINATE_SCALE + _COORD_TOLERANCE
        for coord in coords
    ):
        raise ToolError(
            f"`{param_name}`={list(values)!r} is outside the required 0-to-1000 "
            f"normalized coordinate range. Do not use pixel coordinates. "
            f"For this {width}x{height} image, convert a pixel point with "
            f"`x / {width} * 1000` and `y / {height} * 1000` "
            f"(the image center is always [500, 500]) and call the tool again."
        )

    return [
        min(_COORDINATE_SCALE, max(0.0, coord)) / _COORDINATE_SCALE
        for coord in coords
    ]


def _denormalize_bbox(
    image: Image.Image, bbox: Sequence[float], param_name: str = "crop_box"
) -> list[int]:
    """
    Convert a 0-to-1000 normalized bbox to absolute pixel coordinates.

    Parameters:
    - image: PIL Image object
    - bbox: [x1, y1, x2, y2], where (x1, y1) is top-left and
      (x2, y2) is bottom-right; all values are in [0, 1000].
    - param_name: Name of the tool parameter the bbox came from, used in error messages.

    Returns:
    - Absolute pixel coordinates [x1, y1, x2, y2], following PIL's half-open
      convention: x2 and y2 are exclusive. The box is always at least 1x1
      pixel, so the crop it describes can never be empty.
    """
    left_n, upper_n, right_n, lower_n = _coordinates_1000(
        bbox, 4, param_name, image
    )
    if right_n < left_n:
        left_n, right_n = right_n, left_n
    if lower_n < upper_n:
        upper_n, lower_n = lower_n, upper_n

    width, height = image.size
    # Round the box outwards so a thin but legitimate region (e.g. a single
    # text line) keeps at least one pixel instead of collapsing to nothing.
    left = max(0, min(int(math.floor(left_n * width)), width - 1))
    upper = max(0, min(int(math.floor(upper_n * height)), height - 1))
    right = min(width, max(int(math.ceil(right_n * width)), left + 1))
    lower = min(height, max(int(math.ceil(lower_n * height)), upper + 1))
    return [left, upper, right, lower]

def _denormalize_point(
    image: Image.Image, point: Sequence[float], param_name: str = "points"
) -> list[int]:
    """
    Convert a 0-to-1000 normalized point to absolute pixel coordinates.
    
    Parameters:
    - image: PIL Image object
    - point: An [x, y] point with values in [0, 1000].
    - param_name: Name of the tool parameter the point came from, used in error messages.
    
    Returns:
    - A list defining the absolute pixel coordinates of the point [x, y].
    """
    x_n, y_n = _coordinates_1000(point, 2, param_name, image)
    width, height = image.size
    x = min(int(x_n * width), width - 1)
    y = min(int(y_n * height), height - 1)
    return [x, y]


def _applied_coords(image: Image.Image, pixels: Sequence[int]) -> Dict[str, Any]:
    """
    Describe the coordinates a tool actually operated on, for the output trace.

    The model's own arguments are recorded by the host, but they are not
    necessarily what was applied: `_denormalize_bbox` reorders a reversed box
    and rounds it outwards to keep it at least 1x1. Reporting the applied
    values back makes every produced image reproducible from the trace alone.
    """
    width, height = image.size
    axis = [width, height] * (len(pixels) // 2)
    return {
        "pixels": [int(value) for value in pixels],
        "normalized_1000": [
            round(value / length * _COORDINATE_SCALE, 3)
            for value, length in zip(pixels, axis)
        ],
        "coordinate_scale": [0, 1000],
        "image_size": [width, height],
    }


def _image_content(
    image_bytes: bytes, meta: Dict[str, Any] | None = None
) -> mcp_types.ImageContent:
    """Wrap encoded PNG bytes as image content, carrying `meta` in `_meta`."""
    return mcp_types.ImageContent(
        type="image",
        data=base64.b64encode(image_bytes).decode("utf-8"),
        mimeType="image/png",
        _meta=meta or None,
    )

@mcp.tool()
def crop_zoomin(
    image: InjectedImage,
    crop_box: Box1000,
    expansion_factor: PositiveScale,
) -> mcp_types.ImageContent:
    """
    Crop one image region and return it as a new image.

    USE to inspect a known region more closely. Select the source with
    `image_index` (0 is the original; later indices are previously generated
    images). `crop_box` is [x1, y1, x2, y2], where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom-right corner. Coordinates use the
    0-to-1000 normalized system: x increases left-to-right and y increases
    top-to-bottom. Never pass pixels. [0, 0, 1000, 1000] selects the full image;
    [250, 250, 750, 750] selects its centered half. Use `expansion_factor=1`
    to preserve crop size or a value above 1 to enlarge it. The returned image
    is appended to image history.
    """
    pil_image = _decode_image(image)

    if isinstance(expansion_factor, bool) or not isinstance(expansion_factor, (int, float)) \
            or not math.isfinite(float(expansion_factor)) or expansion_factor <= 0:
        raise ToolError(
            f"`expansion_factor` must be a number greater than 0, got {expansion_factor!r}."
        )

    crop_box = _denormalize_bbox(pil_image, crop_box, "crop_box")
    cropped = pil_image.crop(crop_box)
    if expansion_factor != 1.0:
        width, height = cropped.size
        # Never round a side down to 0: PIL cannot encode an empty image.
        cropped = cropped.resize(
            (max(1, int(width * expansion_factor)), max(1, int(height * expansion_factor))),
            Image.LANCZOS,
        )

    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")
    return _image_content(
        buffered.getvalue(),
        {
            "applied_crop_box": _applied_coords(pil_image, crop_box),
            "applied_expansion_factor": float(expansion_factor),
            "output_size": list(cropped.size),
        },
    )

@mcp.tool()
def mark_dots(
    image: InjectedImage, points: Points1000
) -> mcp_types.ImageContent:
    """
    Mark one or more points with red circles and return a new image.

    USE when counting objects: a dot on each object helps you count them
    without missing or repeating one. Select the source with `image_index`.
    Each point is [x, y] in the 0-to-1000 normalized coordinate system: x
    increases left-to-right and y increases top-to-bottom. Never pass pixels.
    [0, 0] is top-left, [500, 500] is center, and [1000, 1000] is bottom-right.
    You choose the points; the tool only draws them. The returned image is
    appended to image history.
    """
    pil_image = _decode_image(image, "RGB")
    draw = ImageDraw.Draw(pil_image)
    marker_radius = max(3, min(12, round(min(pil_image.size) * 0.006)))

    applied: list[Dict[str, Any]] = []
    for index, point in enumerate(points or []):
        x, y = _denormalize_point(pil_image, point, f"points[{index}]")
        draw.ellipse(
            [
                x - marker_radius,
                y - marker_radius,
                x + marker_radius,
                y + marker_radius,
            ],
            fill=(255, 0, 0),
        )
        applied.append(_applied_coords(pil_image, [x, y]))

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return _image_content(buffered.getvalue(), {"applied_points": applied})

@mcp.tool()
def draw_bbox(
    image: InjectedImage, bboxes: Boxes1000
) -> mcp_types.ImageContent:
    """
    Draw red rectangles around one or more regions and return a new image.

    USE when locating objects or checking how objects are placed relative to
    each other; several boxes can be drawn at once to see them together.
    Select the source with `image_index`. Every box is [x1, y1, x2, y2]:
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    Coordinates use the 0-to-1000 normalized system, with x increasing
    left-to-right and y increasing top-to-bottom; never pass pixels.
    [0, 0, 1000, 1000] outlines the full image. You choose the boxes; the tool
    only draws them. The returned image is appended to image history.
    """
    pil_image = _decode_image(image, "RGB")
    draw = ImageDraw.Draw(pil_image)

    applied: list[Dict[str, Any]] = []
    for index, bbox in enumerate(bboxes or []):
        left, upper, right, lower = _denormalize_bbox(pil_image, bbox, f"bboxes[{index}]")
        # `_denormalize_bbox` returns PIL's half-open crop box, while
        # `rectangle` takes inclusive corners.
        draw.rectangle([left, upper, right - 1, lower - 1], outline=(255, 0, 0), width=2)
        applied.append(_applied_coords(pil_image, [left, upper, right, lower]))

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return _image_content(buffered.getvalue(), {"applied_bboxes": applied})


@mcp.tool()
async def crop_ocr_image(
    image: InjectedImage,
    crop_box: Box1000,
    model_name: InjectedModelName,
    api_key: InjectedApiKey,
    host_url: InjectedHostUrl,
):
    """
    Extract text only from a selected rectangular image region.

    USE when text in a known region is too small or dense to read reliably.
    Select the source with `image_index` and crop tightly around the target
    text. `crop_box` is [x1, y1, x2, y2], where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom-right corner. Coordinates use the
    0-to-1000 normalized system, with x increasing left-to-right and y
    increasing top-to-bottom; never pass pixels. [0, 0, 1000, 1000] selects
    the full image. Returns OCR text, not an image. For visual inspection
    without OCR, use crop_zoomin instead.
    """
    # This is a placeholder implementation. In a real implementation, you would integrate with an OCR library or service.
    ocr_client = _get_openai_client(api_key, host_url)
    pil_image = _decode_image(image)

    crop_box = _denormalize_bbox(pil_image, crop_box, "crop_box")
    cropped = pil_image.crop(crop_box)
    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")
    cropped_b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    ocr_agent_message = [
        {
            "role": "system",
            "content": OCR_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{cropped_b64_string}"
                    }
                }
            ]
        }
    ]
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": ocr_agent_message,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": False,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
    ocr_result = await ocr_client.chat.completions.create(**payload)
    ocr_text = ocr_result.choices[0].message.content
    return mcp_types.TextContent(
        type="text",
        text=ocr_text or "",
        _meta={"applied_crop_box": _applied_coords(pil_image, crop_box)},
    )

@mcp.tool()
async def imaginate_editing(
    image: InjectedImage,
    edit_operation: EditOperation,
    model_name: InjectedModelName,
    api_key: InjectedApiKey,
    host_url: InjectedHostUrl,
):
    """
    Describe the hypothetical visual result of an edit without modifying the image.

    USE only when the question asks what a specified edit would look like.
    Select the source with `image_index` and state one concrete operation in
    `edit_operation`. This tool returns a textual simulation: it does NOT edit
    the image, return a new image, perform OCR, or add anything to image
    history. Use crop/drawing tools for actual image outputs.
    """
    imagination_message = [
        {
            "role": "system",
            "content": IMAGINATION_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Edit Operation: {edit_operation}\n\nInput Image:\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                }
            ]
        }
    ]
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": imagination_message,
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    }
    imagination_client = _get_openai_client(api_key, host_url)
    imagination_result = await imagination_client.chat.completions.create(**payload)
    imagination_text = imagination_result.choices[0].message.content
    return imagination_text

if __name__ == "__main__":
    mcp.run(transport="stdio")