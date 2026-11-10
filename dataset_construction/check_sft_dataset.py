"""Encode a merged cold-start v2 file with LlamaFactory's SFT pipeline and check every record.

Run with the train_cluster env from /workspace/cavir/sft/LlamaFactory:
    /root/miniconda3/envs/train_cluster/bin/python /workspace/cavir/dataset_construction/check_sft_dataset.py \
        <scratch dir> [--dataset cavir_coldstart_v2] [--dataset-dir <cavir>/sft/LlamaFactory/data]
Settings match examples/train_full/qwen35_full_cavir_v2.yaml. Checks that no record is dropped, reports
token lengths, and that every <think>, <evaluating>, <tool_call>, <answer> and image survives into the
encoded (and trained) tokens. Exits with status 1 if any check fails.
"""
import argparse, json, os, sys
from collections import Counter
from llamafactory.hparams import get_train_args
from llamafactory.data import get_template_and_fix_tokenizer, get_dataset
from llamafactory.model import load_tokenizer

parser = argparse.ArgumentParser(description="Check a cold-start dataset with LlamaFactory's SFT preprocessing.")
parser.add_argument("scratch_dir", help="Where LlamaFactory may write its output_dir (nothing is trained).")
parser.add_argument("--dataset", default="cavir_coldstart_v2", help="Name in dataset_info.json.")
parser.add_argument("--dataset-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft", "LlamaFactory", "data"),
                    help="Dir holding dataset_info.json.")
cli = parser.parse_args()
out = cli.scratch_dir
source = os.path.join(cli.dataset_dir, json.load(open(os.path.join(cli.dataset_dir, "dataset_info.json")))[cli.dataset]["file_name"])
args = dict(stage="sft", do_train=True, model_name_or_path="Qwen/Qwen3.5-2B", finetuning_type="full",
            dataset=cli.dataset, dataset_dir=cli.dataset_dir, template="qwen3_5",
            enable_thinking=True, cutoff_len=32000, image_max_pixels=262144, video_max_pixels=16384,
            trust_remote_code=True, output_dir=os.path.join(out, "out"), overwrite_cache=True,
            preprocessing_num_workers=8, report_to="none", per_device_train_batch_size=1)
model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
tok_mod = load_tokenizer(model_args)
template = get_template_and_fix_tokenizer(tok_mod["tokenizer"], data_args)
print("template:", type(template).__name__, "enable_thinking:", template.enable_thinking,
      "preserve_thinking:", getattr(template, "preserve_thinking", None), "mask_history:", data_args.mask_history, flush=True)
ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_mod)["train_dataset"]
tok = tok_mod["tokenizer"]
src = [json.loads(l) for l in open(source)]
print(f"source {len(src)} records, encoded {len(ds)}", flush=True)
if len(ds) != len(src):
    sys.exit(f"{len(src) - len(ds)} records were dropped")

lens = sorted(len(x) for x in ds["input_ids"])
q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))]
print(f"length min {lens[0]} p50 {q(.5)} p90 {q(.9)} p99 {q(.99)} max {lens[-1]}; "
      f">4096: {sum(l > 4096 for l in lens)}, >8192: {sum(l > 8192 for l in lens)}, >16384: {sum(l > 16384 for l in lens)}")

bad = Counter(); examples = {}
for i, rec in enumerate(src):
    ids, labels = ds[i]["input_ids"], ds[i]["labels"]
    text = tok.decode(ids)
    trained = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    gpt = [m["value"] for m in rec["conversations"] if m["from"] == "gpt"]
    n_think = sum(v.count("<think>") for v in gpt)
    n_eval = sum(v.count("<evaluating>") for v in gpt)
    n_call = sum(v.count("<tool_call>") for v in gpt)
    n_ans = sum(v.count("<answer>") for v in gpt)
    checks = {
        "think_in_input": text.count("<think>") >= n_think,
        "think_trained": trained.count("<think>") >= n_think,
        "eval_trained": trained.count("<evaluating>") == n_eval,
        "tool_call_trained": trained.count("<tool_call>") == n_call,
        "answer_trained": trained.count("<answer>") == n_ans,
        "images": text.count("<|vision_start|>") == len(rec["images"]),
        "tools_in_system": "<tools>" in text,
    }
    for k, ok in checks.items():
        if not ok:
            bad[k] += 1
            examples.setdefault(k, (rec["id"], rec["meta"]["variant"]))
print("failed checks:", dict(bad) or "none", examples, flush=True)
sys.exit(1 if bad else 0)
