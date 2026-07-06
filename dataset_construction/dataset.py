import json
import os
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import base64
import math
from typing import Tuple, Optional

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 1024
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGES_DIR = {
    "docvqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/docvqa/images/documents",
    "chartqa_1": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_1",
    "chartqa_2": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3/snapshots/7656391d4d4cb11ec3722b34f10d499435de0460/chartqa_2",
    "infographics_vqa": "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-VLM-Dataset-v2/snapshots/51f4f4d219315c3283950994d4eb3d7fc30aa87b/infographicsvqa_cot/images"
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
    pil_image = Image.open(img_path)
    resized_height, resized_width = smart_resize(pil_image.height, pil_image.width, factor=256*2)
    pil_image = pil_image.resize((resized_width, resized_height))
    
    # Convert PIL Image to base64 string
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
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

    def _load_data(self):
        pass

    def __getitem__(self, idx):
        with open(self.jsonl_path, 'rb') as f:
            f.seek(self._offsets[idx])
            item = json.loads(f.readline())
        item_id = item['id']
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
            gt = item['messages'][1]['content'][0]['text'].strip('\n')
            
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