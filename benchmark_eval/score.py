#!/usr/bin/env python3
"""Score every prediction file of one model and benchmark with VLMEvalKit's official evaluation.

Run with the vlmeval env (see ``prepare.py``) after ``rollout.py predictions``:

    python benchmark_eval/score.py --dataset DocVQA_VAL --out OUT
    python benchmark_eval/score.py --dataset VStarBench --out OUT --judge-base-url URL --judge-model NAME

For each OUT/predictions/<name>.jsonl (natural, execute_r<k>, block_r<k>) it writes the benchmark table with a
``prediction`` column to OUT/vlmeval/<name>.tsv and calls the dataset's own ``evaluate``: ANLS for DocVQA,
relaxed accuracy for ChartQA, option matching for VStarBench. The official overall scores go to OUT/scores/official.json
({name: VLMEvalKit's Overall as it reports it: % for the VQA sets, a fraction for VStarBench}); per-problem scores to OUT/scores/<name>.jsonl ({index, score} in [0, 1]), taken
from VLMEvalKit's own per-item files: hit_calculate on each row's eval_match for the VQA sets, the ``hit`` column for
VStarBench.
VStarBench goes through VLMEvalKit's judge path, as officially: the rule maps a prediction to an option, and the judge
is asked only when the rule fails. VLMEvalKit's default judge (gpt-4o-mini) is an external API, so a served model
stands in for it through VLMEvalKit's own local-judge setting (LOCAL_LLM), with temperature 0, thinking off and 8,192 tokens. The judge must answer a test request before scoring
starts (VLMEvalKit would otherwise fall back to the rule alone). official.json reports per file ``unmatched``, the
predictions the rule could not map (sent to the judge), and ``judge_failed``, those the judge could not map either
(VLMEvalKit then picks an option at random).
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
from pathlib import Path

import vlmeval.dataset.image_vqa as image_vqa
from dotenv import dotenv_values
from vlmeval.dataset import build_dataset
from vlmeval.dataset.utils.judge_util import build_judge
from vlmeval.dataset.utils.vqa_eval import hit_calculate
from vlmeval.smp import dump, load

POOL = 8   # VLMEvalKit's VQA scoring opens Pool(16); at most half of the 20 cores
_pool = multiprocessing.Pool   # image_vqa.mp is the multiprocessing module itself
image_vqa.mp.Pool = lambda processes=None, *a, **k: _pool(min(processes or POOL, POOL), *a, **k)
JUDGE = "local_judge"   # the name VLMEvalKit files the judge results under; LOCAL_LLM holds the served model's name
API_KEY = dotenv_values(Path(__file__).resolve().parents[1] / "dataset_construction" / ".env").get("DIRECT_API_KEY", "EMPTY")


def judge_kwargs(base_url: str) -> dict:
    return {"model": JUDGE, "api_base": f"{base_url.rstrip('/')}/chat/completions", "key": API_KEY, "temperature": 0,
            "max_tokens": 8192, "timeout": 600, "chat_template_kwargs": {"enable_thinking": False}, "nproc": POOL}


def score_file(dataset, name: str, rows: list[dict], out: Path, judge: dict | None) -> tuple[float, list[dict], dict]:
    predictions = {int(r["index"].split("/", 1)[1]): r["prediction"] for r in rows}
    table = dataset.data.drop(columns=["image"]).copy()
    if set(predictions) != set(table["index"].astype(int)):
        raise SystemExit(f"{name}: predictions do not cover the benchmark exactly")
    table["prediction"] = [predictions[int(i)] for i in table["index"]]
    eval_file = out / "vlmeval" / f"{name}.tsv"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    dump(table, str(eval_file))
    result = dataset.evaluate(str(eval_file), **(dict(judge) if dataset.TYPE == "MCQ" else {}))
    overall = float(result.loc[0, "Overall"] if "Overall" in result.columns else result.set_index(result.columns[0]).loc["Overall"].iloc[0])
    stem = eval_file.with_suffix("")
    if dataset.TYPE == "MCQ":
        per_item = load(str(stem.parent / f"{stem.name}_{JUDGE}_result.xlsx"))
        scores = [{"index": f"{dataset.dataset_name}/{int(i)}", "score": float(h)} for i, h in zip(per_item["index"], per_item["hit"])]
        # The rule's answer is logged as the prediction itself; anything else in the log came from the judge.
        notes = {"unmatched": int(sum(str(log) != str(pred) for log, pred in zip(per_item["log"], per_item["prediction"]))),
                 "judge_failed": int(sum("Failed to predict" in str(log) for log in per_item["log"]))}
    else:
        per_item = load(str(stem.parent / f"{stem.name}_results.xlsx"))
        matches = [m if isinstance(m, list) else ast.literal_eval(str(m)) for m in per_item["eval_match"]]
        hits = hit_calculate([{"match": m} for m in matches], dataset.dataset_name)
        scores = [{"index": f"{dataset.dataset_name}/{int(i)}", "score": float(h)} for i, h in zip(per_item["index"], hits)]
        notes = {}
    return overall, scores, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Official VLMEvalKit scores of the prediction files.")
    parser.add_argument("--dataset", required=True, choices=["VStarBench", "DocVQA_VAL", "ChartQA_TEST"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge-base-url", help="VStarBench: OpenAI-compatible server of the judge model")
    parser.add_argument("--judge-model", help="VStarBench: the judge's served model name")
    args = parser.parse_args()
    judge = None
    if args.dataset == "VStarBench":
        if not (args.judge_base_url and args.judge_model):
            raise SystemExit("VStarBench needs --judge-base-url and --judge-model")
        os.environ["LOCAL_LLM"] = args.judge_model
        judge = judge_kwargs(args.judge_base_url)
        if not build_judge(**dict(judge)).working():
            raise SystemExit(f"judge {args.judge_model} at {args.judge_base_url} does not answer")
    folder = args.out / "scores"
    if folder.exists():
        raise SystemExit(f"{folder} exists")
    folder.mkdir(parents=True)
    dataset = build_dataset(args.dataset)
    official = {}
    for path in sorted((args.out / "predictions").glob("*.jsonl")):
        rows = [r for r in map(json.loads, open(path)) if r["index"].startswith(f"{args.dataset}/")]
        overall, scores, notes = score_file(dataset, path.stem, rows, args.out, judge)
        official[path.stem] = {"overall": overall, "problems": len(scores)} | notes
        (folder / f"{path.stem}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
        print(path.stem, official[path.stem], flush=True)
    (folder / "official.json").write_text(json.dumps(official, indent=1) + "\n")


if __name__ == "__main__":
    main()
