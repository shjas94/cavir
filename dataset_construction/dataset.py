import json
import os
import re
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import base64
import math
from typing import Tuple, Optional

IMAGE_MIN_TOKEN_NUM = 4
# Matches the SFT image budget (LLaMA Factory image_max_pixels=262144 = 334 * 28**2),
# so the teacher perceives images at the resolution the student is trained on.
# Fine detail is then only recoverable through tools, which read the original file.
IMAGE_MAX_TOKEN_NUM = 334
MAX_RATIO = 200
IMAGES_DIR = {
    "docvqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/docvqa/images/documents",
    "chartqa_1": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_1",
    "chartqa_2": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_2",
    "infographics_vqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-VLM-Dataset-v2/snapshots/51f4f4d219315c3283950994d4eb3d7fc30aa87b/infographicsvqa_cot/images",
    "plotqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/mulberry_1",
    "gqa_1": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/gqa_1/images",
    "gqa_2": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/gqa_1/images",
    "geometry3k": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/geometry3k/train",
    "geomverse_cot": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-VLM-Dataset-v2/snapshots/51f4f4d219315c3283950994d4eb3d7fc30aa87b/geomverse_cot",
    # v3.5 ships no images: fetch_v35_images.py writes them here under the names the annotations use
    "hopchain_fintabnet": "/root/.cache/huggingface/v35_images/hopchain_fintabnet",
    "chartnet_qa_01_difficulty": "/root/.cache/huggingface/v35_images/chartnet_qa_01_difficulty",
    "coco_count_spatial_qa": "/root/.cache/huggingface/v35_images/coco_count_spatial_qa",
}


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _preprocess_image(img_path: str) -> str:
    """
    Preprocess an image file as base64 string
    Args:
        img_path: Path to the image file
    Returns:
        Base64 encoded string of the image
    """
    # Use context managers so the opened image file handle and the in-memory
    # PNG buffer are released deterministically instead of lingering until GC,
    # which would leak file descriptors/memory across many samples.
    with Image.open(img_path) as pil_image:
        resized_height, resized_width = smart_resize(pil_image.height, pil_image.width, factor=14*2)
        resized_image = pil_image.resize((resized_width, resized_height))

    # Convert PIL Image to base64 string
    with BytesIO() as buffer:
        resized_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64

class HFDataset(Dataset):
    def __init__(self, dataset_name: str, jsonl_path: str):
        self.dataset_name = dataset_name
        self.jsonl_path = jsonl_path
        self._offsets = []
        if self.dataset_name not in IMAGES_DIR.keys():
            raise KeyError(f"dataset_name must be one of {list(IMAGES_DIR.keys())}, but got {self.dataset_name}")
        with open(jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self._offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self._offsets)

    def item_ids(self) -> list[str]:
        """Ids in index order, by the same rule as __getitem__, without decoding any image."""
        with open(self.jsonl_path, 'rb') as f:
            items = [json.loads(line) for line in f if line.strip()]
        return [item.get('id', self.dataset_name + "_" + str(idx)) for idx, item in enumerate(items)]

    def __getitem__(self, idx):
        with open(self.jsonl_path, 'rb') as f:
            f.seek(self._offsets[idx])
            item = json.loads(f.readline())
        if 'id' in item.keys():
            item_id = item['id']
        else:
            item_id = self.dataset_name + "_" + str(idx)
        if self.dataset_name == "docvqa":
            question = item['messages'][0]['content'][0]
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][1]['image'])
            gt = item['messages'][1]['content'][0]
            
        elif self.dataset_name == "chartqa_1" or self.dataset_name == "chartqa_2":
            question = item['messages'][0]['content'][1]
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]
        
        elif self.dataset_name == "infographics_vqa":
            question = item['messages'][0]['content'][1]['text'].strip('\n')
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]['text']
            
        elif self.dataset_name == "plotqa":
            question = item['messages'][0]['content'][1].strip('\n').split('\n')[0]
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]
        
        elif self.dataset_name == "gqa_1" or self.dataset_name == "gqa_2":
            question = item['messages'][0]['content'][1].split('\n')[1]
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]
        
        elif self.dataset_name == "geometry3k":
            geometry3k_postfix = '\nDo not explain your answer but write only the answer label (A, B, ...) in this format:\n\nAnswer: \\boxed{...}.'
            question = item['messages'][0]['content'][1].strip('\n').replace(geometry3k_postfix, '')
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]
        
        # TODO
        elif self.dataset_name == "geomverse_cot":
            question, image_name = "", ""
            for message in item['messages'][0]['content']:
                if message["type"] == "text":
                    question = message["text"]
                elif message["type"] == "image":
                    image_name = message["image"]
            question = "\n".join(message["text"].strip('\n').split("\n")[:-1])
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], image_name)
            gt = item['messages'][1]['content'][0]['text']
            
        elif self.dataset_name in ("hopchain_fintabnet", "chartnet_qa_01_difficulty", "coco_count_spatial_qa"):
            question = item['messages'][0]['content'][1].strip('\n')
            image_path = os.path.join(IMAGES_DIR[self.dataset_name], item['messages'][0]['content'][0]['image'])
            gt = item['messages'][1]['content'][0]
            if self.dataset_name == "coco_count_spatial_qa":
                # "(B)" -> "B": the option letter, as geometry3k's ground truths are written (the teacher boxes the letter)
                gt = re.sub(r"\(([A-Z])\)\s*$", r"\1", gt)

        else:
            raise NotImplementedError(f"dataset_name {self.dataset_name} is not supported yet.")  
        
        image_base64 = _preprocess_image(image_path)
        return {
            'id': item_id,
            'question': question,
            'image': image_base64,
            'image_path': image_path,
            'gt': gt
        }
        
        
def collate_fn(batch):
    collated = []
    for item in batch:
        if item is None:
            continue
        collated.append({
            'id': item['id'],
            'question': item['question'],
            'image': item['image'],
            'image_path': item['image_path'],
            'gt': item['gt']
        })
    return collated