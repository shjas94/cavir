#!/usr/bin/env python3
"""Fetch the images of the chosen Nemotron-Image-Training-v3.5 problems from their original sources.

v3.5 ships annotations only; each subset's README names the source of its images. Only the images of the listed problems
are fetched (the target and held-out ids under rollout_ids/w13/), never a whole source:
- coco_count_spatial_qa: COCO train2017 images one by one over HTTP (the HTTPS host's certificate does not match its name).
- chartnet_qa_01_difficulty: ibm-granite/ChartNet core_permissive parquet at the README's revision. Image keys are random
  UUIDs, so the id column of every file is read first, then only the row groups (100 rows, about 14 MiB) that hold a
  needed image; the embedded bytes are written unchanged as <id>.png.
- hopchain_fintabnet: the FinTabNet PDF of each page from the Kaggle mirror the README names (token in
  /root/.kaggle/access_token), first page rendered as an RGB PNG with a 1,000-pixel longest side.
  The README's renderer uses PyMuPDF; this uses poppler's pdftoppm, already installed, with the same size rule.

    /root/miniconda3/envs/research/bin/python dataset_construction/fetch_v35_images.py DATASET

Images go to IMAGE_ROOT/DATASET under the names the annotations use (dataset.py's IMAGES_DIR points there); a rerun skips
images already written. Prints the counts and any failures; exits 1 if an image is missing at the end.
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
V35 = Path("/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Image-Training-v3.5/snapshots/"
           "70b27e021e64bd97641597dc1f127a208b58be0e")
IMAGE_ROOT = Path("/root/.cache/huggingface/v35_images")
IDS = HERE / "rollout_ids" / "w13"
CHARTNET = "datasets/ibm-granite/ChartNet@ff6cf062b8bb28c9c29ca315f18b1b890ac39712/core_permissive"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/jiongjiong/fintabnet/"
KAGGLE_TRIES = 20
THREADS = 4   # network-bound, but pdftoppm and parquet decoding use CPU (at most half the cores)
LOCATE_THREADS = 16   # ChartNet id lookup: ~25,000 small range reads that only wait on the network


def needed(dataset: str) -> list[str]:
    """Image names (as the annotations write them) of the target and held-out problems."""
    wanted = set((IDS / f"{dataset}.txt").read_text().split()) | set((IDS / f"eval_heldout_{dataset}.txt").read_text().split())
    names = set()
    for line in open(V35 / dataset / f"{dataset}.jsonl"):
        record = json.loads(line)
        if record["id"] in wanted:
            names.add(next(c["image"] for c in record["messages"][0]["content"] if isinstance(c, dict)))
    return sorted(names)


def coco(names: list[str], out: Path) -> None:
    def one(name):
        with urllib.request.urlopen(f"http://images.cocodataset.org/{name}", timeout=60) as response:
            data = response.read()
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    with ThreadPoolExecutor(THREADS) as pool:
        list(pool.map(one, names))


def chartnet(names: list[str], out: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{CHARTNET}/*.parquet"))
    wanted = {Path(name).stem for name in names}

    def locate(path):
        """(path, row group, ids wanted there) from the file's id column."""
        with fs.open(path, block_size=64 << 10) as handle:   # small reads: only the id chunk of each row group
            parquet = pq.ParquetFile(handle)
            ids = parquet.read(columns=["id"]).column("id").to_pylist()
            starts = [0]
            for g in range(parquet.metadata.num_row_groups):
                starts.append(starts[-1] + parquet.metadata.row_group(g).num_rows)
        groups = {}
        for row, image_id in enumerate(ids):
            if image_id in wanted:
                groups.setdefault(bisect.bisect_right(starts, row) - 1, []).append(image_id)
        return [(path, g, found) for g, found in groups.items()]

    located = out.parent / f"{out.name}_located.json"   # the lookup takes a while; a rerun reuses it
    if located.exists():
        work = [tuple(w) for w in json.loads(located.read_text())]
    else:
        with ThreadPoolExecutor(LOCATE_THREADS) as pool:
            work = [w for found in pool.map(locate, files) for w in found]
        located.write_text(json.dumps(work))
    # the cache lists every row group of the first run; a rerun fetches only those still holding a missing image
    work = [(path, group, kept) for path, group, found in work if (kept := [i for i in found if i in wanted])]
    print(f"chartnet: {sum(len(w[2]) for w in work)} images in {len(work)} row groups of {len(files)} files", flush=True)

    def fetch(job):
        path, group, found = job
        with fs.open(path, block_size=16 << 20) as handle:
            table = pq.ParquetFile(handle).read_row_group(group, columns=["id", "image"])
        for image_id, image in zip(table.column("id").to_pylist(), table.column("image").to_pylist()):
            if image_id in found:
                (out / f"{image_id}.png").write_bytes(image["bytes"])
    with ThreadPoolExecutor(THREADS) as pool:
        list(pool.map(fetch, work))


def fintabnet(names: list[str], out: Path) -> None:
    token = Path("/root/.kaggle/access_token").read_text().strip()
    pdf_dir = out.parent / f"{out.name}_pdf"
    pdf_dir.mkdir(exist_ok=True)

    def one(name):
        company, year, _, page = Path(name).stem.rsplit("_", 3)   # BMY_2006_page_111.png -> BMY/2006/page_111.pdf
        pdf = pdf_dir / f"{Path(name).stem}.pdf"
        if not pdf.exists():
            source = f"fintabnet/pdf/{company}/{year}/page_{page}.pdf"
            request = urllib.request.Request(KAGGLE + urllib.parse.quote(source, safe=""),
                                             headers={"Authorization": f"Bearer {token}"})
            for _ in range(KAGGLE_TRIES):
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        pdf.write_bytes(response.read())
                    break
                except urllib.error.HTTPError as exc:   # Kaggle rate-limits (429): wait as told, then retry
                    if exc.code != 429:
                        raise
                    time.sleep(int(exc.headers.get("Retry-After") or 30))
            else:
                raise RuntimeError(f"{source}: still rate-limited after {KAGGLE_TRIES} tries")
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-f", "1", "-l", "1", "-scale-to", "1000", str(pdf),
                        str(out / Path(name).stem)], check=True, capture_output=True)
    for name in names:   # one request at a time: Kaggle rate-limits concurrent downloads (429)
        one(name)


SOURCES = {"coco_count_spatial_qa": coco, "chartnet_qa_01_difficulty": chartnet, "hopchain_fintabnet": fintabnet}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the chosen v3.5 problems' images from their original sources.")
    parser.add_argument("dataset", choices=sorted(SOURCES))
    args = parser.parse_args()
    out = IMAGE_ROOT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    names = needed(args.dataset)
    todo = [n for n in names if not (out / n).exists()]
    print(f"{args.dataset}: {len(names)} images, {len(todo)} to fetch", flush=True)
    if todo:
        SOURCES[args.dataset](todo, out)
    bad = []
    for name in names:
        try:
            with Image.open(out / name) as image:
                image.verify()
        except Exception as exc:
            bad.append(f"{name}: {type(exc).__name__}")
    print(f"{args.dataset}: {len(names) - len(bad)} of {len(names)} images readable in {out}")
    if bad:
        sys.exit("missing or unreadable: " + ", ".join(bad[:10]))


if __name__ == "__main__":
    main()
