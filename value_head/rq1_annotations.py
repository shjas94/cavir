#!/usr/bin/env python3
"""Link branch-rollout problems to their source annotations and find the answer region (faithfulness analysis).

One output row per problem in the given points files:

    {id, dataset, image, image_size, question, answer, source, linked, link_note, regions, region_note}

``regions`` are candidate answer regions as [x0, y0, x1, y1] on the 0–1000 scale of the image the tools saw (the
same scale as the tools' ``normalized_1000``), each with ``kind`` and the annotation ``text`` it matched. A tool call
counts as faithful when it covers any candidate (``rq1_faithfulness.py``). ``region_note`` is ``matched``,
``no_match`` (e.g. a computed answer: send to the LLM judge), or why there is nothing to match.

Annotation sources:
- chartqa_1/2: ``ahmed-masry/ChartQA`` ``ChartQA Dataset.zip`` (train/val/test annotations), by image file name
- gqa_1/2: Stanford GQA ``sceneGraphs.zip`` and ``questions1.2.zip`` in /root/.cache/gqa, by image id and question
- docvqa: ``pixparse/docvqa-single-page-questions`` train OCR (official DocVQA OCR, word boxes), by image file name
- plotqa: ``achang/plot_qa`` train (PlotQA images with their ``models``: values, element boxes, series names; axis
  labels are dropped, so only value answers and series names can match), by pixels, since Nemotron renamed the images
- infographics_vqa: none (official OCR needs an RRC login); geometry: none (LLM judge)

Matching rules (normalisation shared with the crop_ocr_image string rule):
- text: lowercase, whitespace, commas and currency signs removed, numbers written without trailing zeros;
- numbers: an answer that is a number, optionally followed by a unit or %, equals a value with the same float. With
  a magnitude word (thousand, million, billion, trillion) the answer is rounded, so the value also matches when it
  rounds to the answer at the answer's decimals, either as is or divided by the magnitude.
- docvqa: the shortest run of consecutive OCR words (reading order) whose joined normalised text equals the
  normalised answer; every occurrence is a candidate;
- plotqa: answers round the raw data (2 decimals, or 3 significant digits in e-notation), so a value also matches
  when it rounds to an answer that has decimals or is in e-notation;
- line charts annotate segments (one box fewer than points); point i is the shared corner of segments i-1 and i.

    python value_head/rq1_annotations.py --points R1/selected_points.jsonl --points R2/selected_points.jsonl --out DIR
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HUB = Path("/root/.cache/huggingface/hub")
GQA_DIR = Path("/root/.cache/gqa")
CURRENCY = "$€£¥₩"


def norm_text(s: str) -> str:
    s = str(s).lower().strip().rstrip(".")
    s = re.sub(rf"[\s,{CURRENCY}]", "", s)
    return re.sub(r"\d+\.\d+", lambda m: m.group().rstrip("0").rstrip("."), s)


MAGNITUDE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def parse_number(s) -> tuple[float, int, str] | None:
    """(number, decimals, unit) of a value or an answer like "3.9 billion", "45%", "$1,200", "3.66e+07."."""
    m = re.fullmatch(rf"\s*[{CURRENCY}]?\s*(-?[\d.,]*\d(?:e[+-]?\d+)?)\s*(%|[a-z][a-z .]*)?\s*", str(s).lower().rstrip("."))
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    try:
        number = float(digits)
    except ValueError:
        return None
    decimals = len(digits.split(".")[1]) if "." in digits and "e" not in digits else 0
    return number, decimals, (m.group(2) or "").strip()


def rounds_to(value: float, answer: str) -> bool:
    """Whether ``value`` rounded like ``answer`` (its decimals, or its significant digits in e-notation) equals it."""
    a = parse_number(answer)
    m = re.match(r"\s*-?(\d)(?:\.(\d+))?e", answer.lower())
    if m:
        digits = 1 + len(m.group(2) or "")
        return float(f"{value:.{digits - 1}e}") == a[0]
    # An integer answer ("0", "24") would take every value within 0.5; PlotQA stores those exactly.
    return a[1] > 0 and round(value, a[1]) == a[0]


def same_value(answer: str, value, rounded: bool = False) -> bool:
    """Equal numbers or equal normalised text; ``rounded``: the value may carry more precision than the answer."""
    a, v = parse_number(answer), parse_number(str(value).replace(" ", ""))
    if a is None or v is None:
        return norm_text(answer) == norm_text(value) != ""
    if abs(a[0] - v[0]) <= 1e-6 * max(1.0, abs(a[0])) or (rounded and rounds_to(v[0], answer)):
        return True
    scale = next((f for word, f in MAGNITUDE.items() if a[2].startswith(word)), None)
    return scale is not None and any(round(v[0] / s, a[1]) == a[0] for s in (1, scale))


def to_1000(box, size) -> list[float]:
    """Pixel box (x0, y0, x1, y1) in an image of ``size`` (w, h) to the 0–1000 scale, corners ordered."""
    (x0, y0, x1, y1), (w, h) = box, size
    xs, ys = sorted((x0, x1)), sorted((y0, y1))
    return [round(xs[0] * 1000 / w, 3), round(ys[0] * 1000 / h, 3), round(xs[1] * 1000 / w, 3), round(ys[1] * 1000 / h, 3)]


def xywh(b: dict) -> tuple[float, float, float, float]:
    return b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]


# ---- chartqa ----

def chartqa_regions(ann: dict, answer: str, size) -> list[dict]:
    regions = []
    add = lambda box, kind, text: regions.append({"box": to_1000(xywh(box), size), "kind": kind, "text": str(text)})
    info = ann.get("general_figure_info", {})
    for axis in ("x_axis", "y_axis"):
        labels = info.get(axis, {}).get("major_labels", {})
        for value, box in zip(labels.get("values", []), labels.get("bboxes", [])):
            if same_value(answer, value):
                add(box, "axis_label", value)
    for item in info.get("legend", {}).get("items", []):
        text = item.get("label", {}).get("text") or item.get("model")
        if text and same_value(answer, text) and "bbox" in item.get("label", {}):
            add(item["label"]["bbox"], "legend", text)
    for model in ann.get("models", []):
        if "bboxes" in model:  # bar charts: one box per element; line charts: one per segment
            xs, ys, labels, boxes = model.get("x", []), model.get("y", []), model.get("labels", []), model["bboxes"]
            if boxes and len(boxes) == len(ys) - 1:
                boxes = [line_point(boxes, i) for i in range(len(ys))]
            for i, box in enumerate(boxes):
                if i < len(ys) and same_value(answer, ys[i]):
                    add(box, "element_value", ys[i])
                label = labels[i] if i < len(labels) else xs[i] if i < len(xs) else None
                if label is not None and same_value(answer, label):
                    add(box, "element_label", label)
        elif "bbox" in model:  # pie slice
            if "value" in model and same_value(answer, model["value"]):
                add(model["bbox"], "element_value", model["value"])
            if model.get("text_label") and same_value(answer, model["text_label"]):
                add(model["bbox"], "element_label", model["text_label"])
                if "text_bbox" in model:
                    add(model["text_bbox"], "element_text", model["text_label"])
    return regions


def line_point(segments: list[dict], i: int) -> dict:
    """Zero-width box at point i of a line: its x, and the y range shared by the adjacent segments."""
    near = [segments[j] for j in (i - 1, i) if 0 <= j < len(segments)]
    x = segments[i]["x"] if i < len(segments) else segments[i - 1]["x"] + segments[i - 1]["w"]
    y0, y1 = max(s["y"] for s in near), min(s["y"] + s["h"] for s in near)
    if y0 > y1:  # rounding in the annotation
        y0, y1 = min(s["y"] for s in near), max(s["y"] + s["h"] for s in near)
    return {"x": x, "y": y0, "w": 0, "h": y1 - y0}


def link_chartqa(problems: list[dict]) -> None:
    from PIL import Image

    z = zipfile.ZipFile(glob.glob(str(HUB / "datasets--ahmed-masry--ChartQA/snapshots/*/ChartQA Dataset.zip"))[0])
    names = set(z.namelist())
    for p in problems:
        stem = Path(p["image"]).stem
        split = next((s for s in ("train", "val", "test") if f"ChartQA Dataset/{s}/annotations/{stem}.json" in names), None)
        if split is None:
            p["link_note"] = "no annotation for this file name"
            continue
        ann = json.loads(z.read(f"ChartQA Dataset/{split}/annotations/{stem}.json"))
        size = Image.open(io.BytesIO(z.read(f"ChartQA Dataset/{split}/png/{stem}.png"))).size
        p.update(linked=True, link_note=f"file name, {split}", image_size=list(size),
                 source={"file": f"ChartQA Dataset/{split}/annotations/{stem}.json", "chart_type": ann.get("type")},
                 regions=chartqa_regions(ann, p["answer"], size))


# ---- gqa ----

def link_gqa(problems: list[dict]) -> None:
    graphs = {}
    with zipfile.ZipFile(GQA_DIR / "sceneGraphs.zip") as z:
        for name in ("train_sceneGraphs.json", "val_sceneGraphs.json"):
            graphs.update(json.loads(z.read(name)))
    wanted = defaultdict(list)
    for p in problems:
        wanted[Path(p["image"]).stem].append(p)
    found = {}
    with zipfile.ZipFile(GQA_DIR / "questions1.2.zip") as z:
        files = [n for n in z.namelist() if n.endswith(".json") and n.startswith(("train_", "val_"))]  # splits with scene graphs
        # Balanced files first: they are small and hold most of our questions.
        for name in sorted(files, key=lambda n: ("all_questions" in n, n)):
            if len(found) == len(problems):
                break
            for qid, q in json.loads(z.read(name)).items():
                for p in wanted.get(q["imageId"], []):
                    if id(p) not in found and norm_text(q["question"]) == norm_text(p["question"]):
                        found[id(p)] = (name, qid, q)
    for p in problems:
        image_id = Path(p["image"]).stem
        if id(p) not in found or image_id not in graphs:
            p["link_note"] = "question not found" if image_id in graphs else "no scene graph"
            continue
        name, qid, q = found[id(p)]
        graph = graphs[image_id]
        size = (graph["width"], graph["height"])
        refs = set()
        for part in ("question", "answer", "fullAnswer"):
            for ids in q.get("annotations", {}).get(part, {}).values():
                refs.update(str(ids).split(","))
        for step in q.get("semantic", []):
            for ids in re.findall(r"\(([\d,]+)\)", step.get("argument", "")):
                refs.update(ids.split(","))
        objects = [(o, graph["objects"][o]) for o in sorted(refs) if o in graph["objects"]]
        regions = []
        if objects:
            boxes = [xywh(obj) for _, obj in objects]
            union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
            regions.append({"box": to_1000(union, size), "kind": "objects_union", "text": ", ".join(obj["name"] for _, obj in objects)})
        p.update(linked=True, link_note=f"image id + question, {name}", image_size=list(size), regions=regions,
                 source={"file": name, "question_id": qid, "objects": [{"id": o, "name": obj["name"], "box": to_1000(xywh(obj), size)}
                                                                      for o, obj in objects]})
        if not objects:
            p["region_note"] = "no referenced objects"


# ---- docvqa ----

def polygon_box(points: list[int]) -> tuple[int, int, int, int]:
    xs, ys = points[0::2], points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def docvqa_regions(ocr: dict, answer: str) -> list[dict]:
    words = [w for line in ocr["lines"] for w in line["words"]]
    target, regions = norm_text(answer), []
    for i in range(len(words)):
        if not norm_text(words[i]["text"]):  # leader dots and the like would only widen the box
            continue
        joined = ""
        for j in range(i, len(words)):
            joined += norm_text(words[j]["text"])
            if joined == target:
                boxes = [polygon_box(w["bounding_box"]) for w in words[i:j + 1]]
                union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
                regions.append({"box": to_1000(union, (ocr["width"], ocr["height"])), "kind": "ocr_words",
                                "text": " ".join(w["text"] for w in words[i:j + 1])})
            if not target.startswith(joined) or len(joined) >= len(target):
                break
    return regions


def link_docvqa(problems: list[dict]) -> None:
    import pyarrow.parquet as pq

    wanted = {Path(p["image"]).name: p for p in problems}
    files = sorted(glob.glob(str(HUB / "datasets--pixparse--docvqa-single-page-questions/snapshots/*/data/train-*.parquet")))
    for f in files:
        for row in pq.read_table(f, columns=["question_id", "question", "other_metadata", "ocr_results"]).to_pylist():
            p = wanted.get(row["other_metadata"]["image"])
            if p is None or p["linked"]:
                continue
            ocr = row["ocr_results"]
            p.update(linked=True, link_note="file name", image_size=[ocr["width"], ocr["height"]],
                     source={"file": Path(f).name, "doc_id": row["other_metadata"]["doc_id"], "image": row["other_metadata"]["image"],
                             "same_question": norm_text(row["question"]) == norm_text(p["question"])},
                     regions=docvqa_regions(ocr, p["answer"]))
    for p in problems:
        if not p["linked"]:
            p["link_note"] = "file name not in pixparse train"


# ---- plotqa ----

def plotqa_models(text: str) -> list[dict]:
    """``achang/plot_qa`` donut-token string to [{name, y, boxes}], boxes as pixel (x0, y0, x1, y1)."""
    models = []
    for chunk in text.split("</s_bboxes>")[:-1]:
        head, boxes = chunk.split("<s_bboxes>")
        y = re.search(r"<s_y>(.*?)</s_y>", head)
        name = re.search(r"<s_name>(.*?)</s_name>", head)
        models.append({"name": name.group(1) if name else None, "y": y.group(1).split("<sep/>") if y else [],
                       "boxes": [(int(x), int(y0), int(x) + int(w), int(y0) + int(h)) for y0, x, w, h in
                                 re.findall(r"<s_y>(\d+)</s_y><s_x>(\d+)</s_x><s_w>(\d+)</s_w><s_h>(\d+)</s_h>", boxes)]})
    return models


def grey(image):
    import numpy as np

    return np.asarray(image.convert("L")).astype(np.int16)


def link_plotqa(problems: list[dict]) -> None:
    """Find each image among the PlotQA train images: same size and the same grey pixels (mean difference < 0.5).

    PlotQA charts share templates, so near-duplicates are common and a small thumbnail can match the wrong chart; at
    full resolution the true image differs by 0.0 and the next closest by much more.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    ours = defaultdict(list)
    for p in problems:
        image = Image.open(p["image"])
        ours[image.size].append((p, grey(image)))
    files = sorted(glob.glob(str(HUB / "datasets--achang--plot_qa/snapshots/*/data/train-*.parquet")))
    for f in files:
        row_index = 0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=512, columns=["image", "text"]):
            for row in batch.to_pylist():
                image = Image.open(io.BytesIO(row["image"]["bytes"]))
                candidates = [(p, pixels) for p, pixels in ours.get(image.size, []) if not p["linked"]]
                pixels_here = grey(image) if candidates else None
                for p, pixels in candidates:
                    if np.abs(pixels_here - pixels).mean() < 0.5:
                        regions = []
                        for model in plotqa_models(row["text"]):
                            if model["name"] and same_value(p["answer"], model["name"]):
                                regions += [{"box": to_1000(b, image.size), "kind": "series", "text": model["name"]} for b in model["boxes"]]
                            for value, box in zip(model["y"], model["boxes"]):
                                if same_value(p["answer"], value, rounded=True):
                                    regions.append({"box": to_1000(box, image.size), "kind": "element_value", "text": value})
                        p.update(linked=True, link_note="pixels (same size, mean grey difference < 0.5)", image_size=list(image.size), regions=regions,
                                 source={"file": Path(f).name, "row": row_index})
                row_index += 1
        if all(p["linked"] for p in problems):
            break
    for p in problems:
        if not p["linked"]:
            p["link_note"] = "image not found in achang/plot_qa train"


LINKERS = {"chartqa_1": link_chartqa, "chartqa_2": link_chartqa, "gqa_1": link_gqa, "gqa_2": link_gqa,
           "docvqa": link_docvqa, "plotqa": link_plotqa}
NO_SOURCE = {"infographics_vqa": "official OCR needs an RRC login",
             "geometry3k": "no region annotation (LLM judge)", "geomverse_cot": "no region annotation (LLM judge)"}


def load_problems(points: list[Path]) -> list[dict]:
    problems = {}
    for path in points:
        for line in open(path):
            d = json.loads(line)
            if d["id"] not in problems:
                problems[d["id"]] = {"id": d["id"], "dataset": d["dataset"], "image": d["images"][0],
                                     "question": d["turns"][0].replace("<image>", "").strip(), "answer": d["answer"],
                                     "image_size": None, "source": None, "linked": False, "link_note": None, "regions": []}
    return list(problems.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Link problems to annotations and find answer regions.")
    parser.add_argument("--points", action="append", type=Path, required=True, help="selected_points.jsonl of each run")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", help="only these datasets (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    problems = [p for p in load_problems(args.points) if not args.datasets or p["dataset"] in args.datasets]
    by_dataset = defaultdict(list)
    for p in problems:
        by_dataset[p["dataset"]].append(p)
    linkers = defaultdict(list)
    for dataset, rows in by_dataset.items():
        if dataset in NO_SOURCE:
            for p in rows:
                p["link_note"] = NO_SOURCE[dataset]
        else:
            linkers[LINKERS[dataset]].extend(rows)
    for linker, rows in linkers.items():
        linker(rows)
    for p in problems:
        if p["linked"] and "region_note" not in p:
            p["region_note"] = "matched" if p["regions"] else "no_match"
        p.setdefault("region_note", None)

    with open(args.out / "annotations.jsonl", "w") as handle:
        for p in problems:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary = {d: {"problems": len(rows), "linked": sum(p["linked"] for p in rows),
                   "region_note": dict(Counter(p["region_note"] for p in rows)),
                   "link_note": dict(Counter(p["link_note"] for p in rows if not p["linked"])),
                   "region_kinds": dict(Counter(r["kind"] for p in rows for r in p["regions"])),
                   "multiple_regions": sum(len(p["regions"]) > 1 for p in rows)}
               for d, rows in sorted(by_dataset.items())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
