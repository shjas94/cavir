import json
from PIL import Image, ImageDraw
import io
import ast
import base64
import mcp.server.fastmcp as mcp_server
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
import os
from typing import Dict, Any

"""
Tool server for cropping and zooming an image based on a specified crop box and expansion factor.
"""

IMAGINATION_PROMPT = """
You are an image edit simulation model that imagines the result of editing an image based on a given edit operation. 
You do not perform actual image editing, but instead provide a detailed description of what the edited image would look like after applying the specified edit operation to the input image.
Given the Edit Operation and Input Image, describe the detailed description of the expected result image after applying the edit operation. 
"""

OCR_PROMPT = """
You are and OCR model that extracts text from images.
"""


mcp = FastMCP("image_processing")

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


def _denormalize_bbox(image: Image.Image, bbox: list[float, float, float, float]) -> list[int, int, int, int]:
    """
    Convert normalized bbox coordinates to absolute pixel coordinates.
    
    Parameters:
    - image: PIL Image object
    - bbox: A list defining the normalized bounding box (left, upper, right, lower) with values in [0, 1].
    
    Returns:
    - A list defining the absolute pixel coordinates of the bounding box (left, upper, right, lower).
    """
    width, height = image.size
    left = int(bbox[0] * width)
    upper = int(bbox[1] * height)
    right = int(bbox[2] * width)
    lower = int(bbox[3] * height)
    # Clamp to valid pixel range and enforce ordering
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1))
    upper = max(0, min(upper, height - 1))
    lower = max(0, min(lower, height - 1))
    if right < left:
        left, right = right, left
    if lower < upper:
        upper, lower = lower, upper
    return [left, upper, right, lower]

def _denormalize_point(image: Image.Image, point: list[float, float]) -> list[int, int]:
    """
    Convert normalized point coordinates to absolute pixel coordinates.
    
    Parameters:
    - image: PIL Image object
    - point: A list defining the normalized point (x, y) with values in [0, 1].
    
    Returns:
    - A list defining the absolute pixel coordinates of the point [x, y].
    """
    width, height = image.size
    x = max(0, min(int(point[0] * width), width - 1))
    y = max(0, min(int(point[1] * height), height - 1))
    return [x, y]

@mcp.tool()
def crop_zoomin(image: str, crop_box: list[float, float, float, float], expansion_factor: float) -> mcp_server.Image:
    """
    Crops the image using the provided crop box and applies zoom based on the expansion factor.
    Use this tool when you want to crop a specific region from the image and optionally zoom in on that region for a closer view. The crop box is defined in normalized coordinates, and the expansion factor allows you to specify how much to zoom in on the cropped area.
    
    Parameters:
    - image_index: Index of the image to operate on, NOT the raw image data itself. 0 refers to the original input image; every successful call to crop_zoomin, mark_dots, or draw_bbox appends a new image to the history (1 = first such image, 2 = second, etc.). Pick the index of whichever image (original or a previously produced one) you want to crop from. Never pass base64 image bytes yourself.
    - image: base64-encoded string of the image resolved from `image_index`. (Injected automatically by the host based on `image_index`. Do not pass this yourself.)
    - crop_box: A list defining the crop box (left, upper, right, lower) in normalized coordinates of the original image (origin at the top-left corner, x increases rightward, y increases downward).
    - expansion_factor: A float indicating the zoom factor to apply to the cropped region.

    Returns:
    - An Image object representing the cropped and zoomed image.
    """
    image_data = base64.b64decode(image)
    pil_image = Image.open(io.BytesIO(image_data))

    crop_box = _denormalize_bbox(pil_image, crop_box)
    cropped = pil_image.crop(crop_box)
    if expansion_factor != 1.0:
        width, height = cropped.size
        cropped = cropped.resize((int(width * expansion_factor), int(height * expansion_factor)), Image.LANCZOS)

    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")
    return mcp_server.Image(data=buffered.getvalue(), format="png")

@mcp.tool()
def mark_dots(image: str, points: list[list[float, float]]) -> mcp_server.Image:
    """
    Draw dots on the image at the specified points.
    Use this tool when you want to highlight specific points of interest on the image. The points are defined in normalized coordinates, and the tool will mark these points with red dots on the image.
    
    Parameters:
    - image_index: Index of the image to operate on, NOT the raw image data itself. 0 refers to the original input image; every successful call to crop_zoomin, mark_dots, or draw_bbox appends a new image to the history (1 = first such image, 2 = second, etc.). Pick the index of whichever image (original or a previously produced one) you want to mark. Never pass base64 image bytes yourself.
    - image: base64-encoded string of the image resolved from `image_index`. (Injected automatically by the host based on `image_index`. Do not pass this yourself.)
    - points: A list of lists, where each inner list contains the (x, y) coordinates of a point to be marked, given in normalized coordinates of the original image (origin at the top-left corner, x increases rightward, y increases downward).
    
    Returns:
    - An Image object representing the image with marked dots.
    """
    image_data = base64.b64decode(image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    for point in points:
        x, y = _denormalize_point(pil_image, point)
        pil_image.putpixel((x, y), (255, 0, 0))  # Mark the point with a red dot

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return mcp_server.Image(data=buffered.getvalue(), format="png")

@mcp.tool()
def draw_bbox(image: str, bboxes: list[list[float, float, float, float]]) -> mcp_server.Image:
    """
    Draw bounding boxes on the image.
    Use this tool when you want to highlight specific regions of interest on the image. The bounding boxes are defined in normalized coordinates, and the tool will draw red rectangles around these regions.
    
    Parameters:
    - image_index: Index of the image to operate on, NOT the raw image data itself. 0 refers to the original input image; every successful call to crop_zoomin, mark_dots, or draw_bbox appends a new image to the history (1 = first such image, 2 = second, etc.). Pick the index of whichever image (original or a previously produced one) you want to annotate. Never pass base64 image bytes yourself.
    - image: base64-encoded string of the image resolved from `image_index`. (Injected automatically by the host based on `image_index`. Do not pass this yourself.)
    - bboxes: A list of lists, where each inner list defines a bounding box (left, upper, right, lower) in normalized coordinates of the original image (origin at the top-left corner, x increases rightward, y increases downward).
    
    Returns:
    - An Image object representing the image with drawn bounding boxes.
    """
    image_data = base64.b64decode(image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for bbox in bboxes:
        left, upper, right, lower = _denormalize_bbox(pil_image, bbox)
        draw.rectangle([left, upper, right, lower], outline=(255, 0, 0), width=2)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return mcp_server.Image(data=buffered.getvalue(), format="png")


@mcp.tool()
async def crop_ocr_image(image: str, crop_box: list[float, float, float, float], model_name: str, api_key: str, host_url: str):
    """
    Crop the input image based on the specified crop box and return the extracted text.
    Use this tool when you want to extract text from a specific region of the image. The crop box is defined in normalized coordinates, and the tool will crop the image accordingly and then use an OCR model to extract and return the text from that region.
    
    Parameters:
    - image_index: Index of the image to run OCR on, NOT the raw image data itself. 0 refers to the original input image; every successful call to crop_zoomin, mark_dots, or draw_bbox appends a new image to the history (1 = first such image, 2 = second, etc.). Never pass base64 image bytes yourself.
    - image: base64-encoded string of the image resolved from `image_index`. (Injected automatically by the host based on `image_index`. Do not pass this yourself.)
    - crop_box: A list defining the bounding box (left, upper, right, lower) in normalized coordinates of the original image (origin at the top-left corner, x increases rightward, y increases downward).
    - model_name: The name of the OCR model to be used. (Injected automatically by the host. Do not pass this yourself.)
    - api_key: The API key for accessing the OCR model. (Injected automatically by the host. Do not pass this yourself.)
    - host_url: The URL of the OCR model. (Injected automatically by the host. Do not pass this yourself.)
    
    Returns:
    - A string containing the text extracted from the image.
    """
    # This is a placeholder implementation. In a real implementation, you would integrate with an OCR library or service.
    ocr_client = _get_openai_client(api_key, host_url)
    image_data = base64.b64decode(image)
    pil_image = Image.open(io.BytesIO(image_data))

    crop_box = _denormalize_bbox(pil_image, crop_box)
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
    return ocr_text

@mcp.tool()
async def imaginate_editing(image: str, edit_operation: str, model_name: str, api_key: str, host_url: str):
    """
    Perform imaginate editing on the input image based on the specified edit operation.
    Use this tool when you want to simulate modify or enhance specific aspects of an image. The edit operation is defined as a string describing the desired changes, and the tool will return description of the expected result image.
    
    Parameters:
    - image_index: Index of the image to imagine editing on, NOT the raw image data itself. 0 refers to the original input image; every successful call to crop_zoomin, mark_dots, or draw_bbox appends a new image to the history (1 = first such image, 2 = second, etc.). Never pass base64 image bytes yourself.
    - image: base64-encoded string of the image resolved from `image_index`. (Injected automatically by the host based on `image_index`. Do not pass this yourself.)
    - edit_operation: A string describing the desired edit operation to be performed on the image.
    - model_name: The name of the image editing model to be used. (Injected automatically by the host. Do not pass this yourself.)
    - api_key: The API key for accessing the image editing model. (Injected automatically by the host. Do not pass this yourself.)
    - host_url: The URL of the image editing model. (Injected automatically by the host. Do not pass this yourself.)
    
    Returns:
    - Description of the edited image
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