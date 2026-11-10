"""Stage-2 scorer: frozen policy + scorer LoRA + M heads predicting q(a) per action.

The input is a decision point's prefix (everything before ``<evaluating>``), encoded with LLaMA Factory's
training path exactly as ``sft/selection_following.py`` does: turns through ``encode_multiturn`` with the
``qwen3_5`` template and ``image_max_pixels`` 262,144, then the unfinished assistant text appended. Each head
maps the last prefix token's final hidden state to one logit per action; sigmoid gives q(a), and
Δ(o) = q(o) − q(reference), the reference being answer_now, think_more or the better of the two
(``analyze_branches.REFERENCES``). Heads differ only in initialization; their spread is the uncertainty.
Labels are binomial counts (s successes of n continuations) from ``analyze_branches.py``.

Two variants start from the (dataset, action) mean table (``mean_table``): each head's
output is logit(table q) + a correction whose last layer starts at zero, so an untrained scorer predicts the table.
Method 1 trains LoRA and heads; method 2 (probe) keeps the policy frozen and trains the heads on hidden states cached
once by ``cache_features.py``. ``fold_split`` gives the 5-fold cross-validation splits.

Runs in the ``train_cluster`` env with cwd ``sft/LlamaFactory``.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

Image.MAX_IMAGE_PIXELS = None  # as in policy.py: visualize_data plots in prefixes can be huge

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "sft/LlamaFactory/saves/qwen35-2b/full/cavir_coldstart_v2_noarith_fixtools"
ACTIONS = ("answer_now", "crop_zoomin", "crop_ocr_image", "draw_bbox", "mark_dots", "imaginate_editing",
           "visualize_data", "math_compute", "think_more")  # think_more last: older 8-action checkpoints keep their indices
# Language-model linear layers of both layer kinds (Gated DeltaNet and full attention) and the MLPs. The vision
# tower and the 16-wide in_proj_a/in_proj_b gates stay frozen without LoRA.
LORA_TARGETS = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


def stratum(prescreen: dict | None) -> str:
    """Pre-screen stratum from answer_now samples: mixed, all_correct, all_wrong (or none without a pre-screen)."""
    if not prescreen:
        return "none"
    if prescreen["s"] == 0:
        return "all_wrong"
    return "all_correct" if prescreen["s"] == prescreen["n"] else "mixed"


def load_examples(pairs: list[tuple[Path, Path]]) -> list[dict]:
    """One example per labeled decision point from (points file, labels file) pairs; rows with n = 0 are dropped."""
    examples = []
    for points_path, labels_path in pairs:
        labels = defaultdict(dict)
        for line in open(labels_path):
            row = json.loads(line)
            if row["n"] > 0:
                labels[row["dp_id"]][row["action"]] = (row["s"], row["n"])
        for line in open(points_path):
            point = json.loads(line)
            if point["dp_id"] not in labels:
                continue
            unknown = set(labels[point["dp_id"]]) - set(ACTIONS)
            if unknown:
                raise ValueError(f"{point['dp_id']}: actions outside ACTIONS: {unknown}")
            examples.append({**{k: point[k] for k in ("dp_id", "id", "dataset", "system", "tools", "turns", "images", "partial")},
                             "stratum": stratum(point.get("prescreen")), "labels": labels.pop(point["dp_id"]),
                             "source": str(points_path)})
        if labels:
            raise ValueError(f"{labels_path}: {len(labels)} labeled points missing from {points_path}")
    dp_ids = [e["dp_id"] for e in examples]
    if len(dp_ids) != len(set(dp_ids)):
        raise ValueError("duplicate dp_id across inputs")
    return examples


def split_ids(examples: list[dict], seed: int, ratios: tuple[float, float] = (0.8, 0.1)) -> dict[str, str]:
    """Problem id -> train/val/test, stratified by dataset, so no problem straddles two splits."""
    by_dataset = defaultdict(set)
    for e in examples:
        by_dataset[e["dataset"]].add(e["id"])
    assignment = {}
    for dataset in sorted(by_dataset):
        ids = sorted(by_dataset[dataset])
        random.Random(f"{seed}:{dataset}").shuffle(ids)
        n_train, n_val = round(len(ids) * ratios[0]), round(len(ids) * ratios[1])
        for i, pid in enumerate(ids):
            assignment[pid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


class Encoder:
    """Token ids and image tensors for a decision-point prefix, via LLaMA Factory's SFT encoding."""

    def __init__(self, model_path: str):
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_tokenizer

        model_args, data_args, _, _ = get_infer_args(dict(
            model_name_or_path=model_path, template="qwen3_5", enable_thinking=True, image_max_pixels=262144,
            trust_remote_code=True, infer_backend="huggingface"))
        tok_mod = load_tokenizer(model_args)
        self.tokenizer, self.processor = tok_mod["tokenizer"], tok_mod["processor"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)

    def __call__(self, example: dict) -> dict:
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(example["turns"])]
        messages.append({"role": "assistant", "content": ""})
        plugin = self.template.mm_plugin
        messages = plugin.process_messages(messages, example["images"], [], [], self.processor)
        pairs = self.template.encode_multiturn(self.tokenizer, messages, example["system"], example["tools"])
        ids = [t for src, tgt in pairs[:-1] for t in src + tgt] + pairs[-1][0]
        ids += self.tokenizer.encode(example["partial"], add_special_tokens=False)
        mm = plugin.get_mm_inputs(example["images"], [], [], [len(example["images"])], [0], [0], [ids], self.processor)
        input_ids = torch.tensor([ids])
        # M-RoPE needs the image positions; marked as LLaMA Factory's SFT collator does (image = 1).
        mm_token_type_ids = (input_ids == self.tokenizer.convert_tokens_to_ids("<|image_pad|>")).long()
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "mm_token_type_ids": mm_token_type_ids,
                **{k: v for k, v in mm.items() if torch.is_tensor(v)}}


def last_hidden(backbone: nn.Module, inputs: dict) -> torch.Tensor:
    """Final (normed) hidden state of the last prefix token, float32, from the inner model without the 248k lm_head."""
    inner = backbone.get_base_model().model if hasattr(backbone, "get_base_model") else backbone.model
    return inner(**inputs, use_cache=False).last_hidden_state[0, -1].float()


class Scorer(nn.Module):
    """Backbone (with LoRA) + ``heads`` MLPs; forward returns logits of shape [heads, len(ACTIONS)].

    ``backbone`` may be None for a probe on cached hidden states (give ``width``; call ``heads_only``).
    ``zero_last`` starts every head's last layer at zero, so the logits start at the added offset."""

    def __init__(self, backbone: nn.Module | None, heads: int, hidden: int = 512, seed: int = 0, actions: int = len(ACTIONS),
                 width: int | None = None, zero_last: bool = False):
        super().__init__()
        self.backbone = backbone
        width = width or backbone.config.text_config.hidden_size
        self.heads = nn.ModuleList()
        for i in range(heads):
            torch.manual_seed(seed * 1000 + i)
            self.heads.append(nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, actions)))
            if zero_last:
                nn.init.zeros_(self.heads[-1][2].weight)
                nn.init.zeros_(self.heads[-1][2].bias)

    def heads_only(self, hidden: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
        logits = torch.stack([head(hidden) for head in self.heads])
        return logits if offset is None else logits + offset

    def forward(self, inputs: dict, offset: torch.Tensor | None = None) -> torch.Tensor:
        return self.heads_only(last_hidden(self.backbone, inputs), offset)


def load_backbone(model_path: str, lora: dict | None = None, adapter: str | None = None) -> nn.Module:
    """Policy in bf16, frozen; adds a fresh LoRA (``lora`` = peft kwargs) or loads a saved ``adapter``."""
    import peft.tuners.lora.torchao as peft_torchao
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText

    # peft 0.18.1 imports a name torchao 0.18.0 (train_cluster) no longer has; the weights are plain bf16.
    peft_torchao.is_torchao_available = lambda: False

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).cuda()
    model.requires_grad_(False)
    if adapter:
        return PeftModel.from_pretrained(model, adapter)
    if lora:
        return get_peft_model(model, LoraConfig(target_modules=LORA_TARGETS, bias="none", **lora))
    return model


def binomial_nll(logits: torch.Tensor, labels: dict[str, tuple[int, int]]) -> torch.Tensor:
    """Binomial NLL summed over labeled actions and averaged over heads; unlabeled actions are masked out."""
    index = torch.tensor([ACTIONS.index(a) for a in labels], device=logits.device)
    s = torch.tensor([labels[a][0] for a in labels], dtype=torch.float32, device=logits.device)
    n = torch.tensor([labels[a][1] for a in labels], dtype=torch.float32, device=logits.device)
    z = logits[:, index]
    return -(s * F.logsigmoid(z) + (n - s) * F.logsigmoid(-z)).sum(-1).mean()


def save_scorer(scorer: Scorer, out: Path) -> None:
    """Heads, plus the LoRA adapter when the backbone has one (a probe has none)."""
    if hasattr(scorer.backbone, "save_pretrained") and hasattr(scorer.backbone, "get_base_model"):
        scorer.backbone.save_pretrained(out / "adapter")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(scorer.heads.state_dict(), out / "heads.pt")


def load_scorer(ckpt: Path, model_path: str | None, heads: int, hidden: int, actions: int = len(ACTIONS),
                width: int | None = None) -> Scorer:
    """``model_path`` None loads only the heads (a probe evaluated on cached hidden states)."""
    adapter = ckpt / "adapter"
    backbone = load_backbone(model_path, adapter=str(adapter) if adapter.exists() else None) if model_path else None
    scorer = Scorer(backbone, heads, hidden, actions=actions, width=width)
    scorer.heads.load_state_dict(torch.load(ckpt / "heads.pt", map_location="cpu"))
    return scorer.cuda().eval()


TABLE_CLAMP = (0.01, 0.99)  # keeps logit(table q) finite where every train continuation succeeded or failed


def mean_table(examples: list[dict]) -> dict[str, dict[str, float]]:
    """Mean q = s/n per (dataset, action) over ``examples`` (each decision point weighs 1)."""
    acc = defaultdict(list)
    for e in examples:
        for action, (s, n) in e["labels"].items():
            acc[(e["dataset"], action)].append(s / n)
    table = defaultdict(dict)
    for (dataset, action), values in sorted(acc.items()):
        table[dataset][action] = sum(values) / len(values)
    return dict(table)


def table_offset(table: dict, dataset: str, actions: tuple[str, ...] = ACTIONS) -> torch.Tensor:
    """logit of the clamped table q per action; 0 for an action the table lacks (no labels, so no loss either)."""
    lo, hi = TABLE_CLAMP
    q = [min(max(table.get(dataset, {}).get(a, 0.5), lo), hi) for a in actions]
    return torch.tensor([math.log(p / (1 - p)) for p in q])


def fold_split(examples: list[dict], folds: int, fold: int, seed: int, val_share: float = 0.125) -> dict[str, str]:
    """Problem id -> train/val/test for one fold of cross-validation.

    Folds are dealt round-robin per dataset over sorted problem ids, so the mean table
    and the scorers share folds. Fold ``fold`` is
    the test split; ``val_share`` of the other folds' problems per dataset (shuffled with ``seed``) is val."""
    by_dataset = defaultdict(set)
    for e in examples:
        by_dataset[e["dataset"]].add(e["id"])
    assignment = {}
    for dataset in sorted(by_dataset):
        ids = sorted(by_dataset[dataset])
        rest = [pid for i, pid in enumerate(ids) if i % folds != fold]
        random.Random(f"{seed}:{fold}:{dataset}").shuffle(rest)
        n_val = round(len(rest) * val_share)
        assignment |= {pid: "test" for i, pid in enumerate(ids) if i % folds == fold}
        assignment |= {pid: "val" if i < n_val else "train" for i, pid in enumerate(rest)}
    return assignment


def dist_setup() -> tuple[int, int]:
    """(rank, world size): under torchrun across both nodes (one GPU each) the NCCL process group is started;
    a plain run is (0, 1). Launch with ``run_ddp.sh``."""
    import os
    from datetime import timedelta

    import torch.distributed as dist

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, world


def gather_to_all(items: list, world: int) -> list:
    """Concatenation of every rank's ``items`` (rank order); ``items`` itself when not distributed."""
    if world == 1:
        return items
    import torch.distributed as dist

    parts = [None] * world
    dist.all_gather_object(parts, items)
    return [x for part in parts for x in part]


MIN_AVAILABLE_GB = 16  # node peak must stay ≤ 120 GB of ~130.7 GB; stop well before the system freezes


def memory_guard() -> None:
    """Exit when MemAvailable drops below MIN_AVAILABLE_GB (unified memory: a full node freezes instead of OOM)."""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024 / 1e9
            if available < MIN_AVAILABLE_GB:
                raise SystemExit(f"memory guard: MemAvailable {available:.1f} GB < {MIN_AVAILABLE_GB} GB, stopping")
            return
