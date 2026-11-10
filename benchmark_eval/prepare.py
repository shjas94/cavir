#!/usr/bin/env python3
"""Turn VLMEvalKit's benchmark files into the problem list ``rollout.py`` reads.

Run with the vlmeval env (VLMEvalKit from /workspace/VLMEvalKit on PYTHONPATH, see README), after the benchmark TSVs
are in $LMUData:

    python benchmark_eval/prepare.py --out benchmark_eval/outputs/problems

Writes OUT/<dataset>.jsonl for VStarBench, DocVQA_VAL and ChartQA_TEST, one line per benchmark problem (none left out):
{index "<dataset>/<VLMEvalKit index>", dataset, source_index, prompt, image, answers, ...}. ``prompt`` and ``image``
are what VLMEvalKit's ``build_prompt`` gives the model: the official question text and the image VLMEvalKit writes
out of the TSV, at the resolution stored there (full resolution). ``answers`` are the accepted answers (DocVQA keeps several), used by the
faithfulness judgment; scoring reads the answers from VLMEvalKit itself.
VStarBench rows also get ``bboxes``, the answer objects' boxes ([x, y, w, h] pixels) from the original V* release
(craigwu/vstar_bench): its test_questions.jsonl ids equal the TSV index, which is checked against the question text
and the answer letter; a problem that fails the check gets no boxes (faithfulness None).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from vlmeval.dataset import build_dataset

DATASETS = ("VStarBench", "DocVQA_VAL", "ChartQA_TEST")
VSTAR = "craigwu/vstar_bench"


def vstar_boxes() -> dict[str, dict]:
    """question_id -> {question, label, bboxes} from the original V* annotations."""
    root = Path(snapshot_download(VSTAR, repo_type="dataset", allow_patterns=["*.json", "*.jsonl"]))
    boxes = {}
    for q in map(json.loads, open(root / "test_questions.jsonl")):
        annotation = json.loads((root / q["image"]).with_suffix(".json").read_text())
        boxes[q["question_id"]] = {"question": q["text"].split("\n")[0], "label": q["label"], "bboxes": annotation["bbox"]}
    return boxes


def answers(dataset: str, raw: str) -> list[str]:
    return [str(a) for a in ast.literal_eval(raw)] if dataset == "DocVQA_VAL" else [str(raw)]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLMEvalKit benchmark files to rollout.py problem lists.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    boxes = vstar_boxes()
    for name in DATASETS:
        path = args.out / f"{name}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists")
        dataset = build_dataset(name)
        rows, unmatched = [], 0
        for i in range(len(dataset.data)):
            line = dataset.data.iloc[i]
            msgs = dataset.build_prompt(line)
            images = [m["value"] for m in msgs if m["type"] == "image"]
            texts = [m["value"] for m in msgs if m["type"] == "text"]
            if len(images) != 1 or len(texts) != 1:
                raise SystemExit(f"{name} {line['index']}: expected one image and one text, got {[m['type'] for m in msgs]}")
            row = {"index": f"{name}/{line['index']}", "dataset": name, "source_index": int(line["index"]),
                   "prompt": texts[0], "image": images[0], "answers": answers(name, line["answer"])}
            if name == "VStarBench":
                b = boxes.get(str(line["index"]))
                ok = b is not None and b["question"].strip() == str(line["question"]).strip() and b["label"] == line["answer"]
                row["bboxes"] = b["bboxes"] if ok else None
                unmatched += not ok
            rows.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name}: {len(rows)} problems -> {path}" + (f" ({unmatched} without matched boxes)" if name == "VStarBench" else ""))


if __name__ == "__main__":
    main()
