#!/usr/bin/env python3
"""Train the stage-2 scorer (LoRA + heads on the frozen policy) on branch-rollout labels.

Run with the train_cluster env from sft/LlamaFactory, one GPU:

    cd /workspace/cavir/sft/LlamaFactory
    /root/miniconda3/envs/train_cluster/bin/python ../../value_head/train_scorer.py \
        --data R1/selected_points.jsonl R1/analysis_labels.jsonl --data R2/selected_points.jsonl R2/analysis_labels.jsonl \
        --out ../../value_head/outputs/scorer_<name>

``--data POINTS LABELS`` may repeat; all inputs are pooled and then split by problem id (80/10/10 per dataset), so
later decision points of a problem land in the same split as its first. One decision point per forward pass,
gradients accumulated over ``--grad-accum`` points. Loss is the binomial NLL per continuation; after each epoch (or
every ``--val-every`` optimizer steps) the val NLL (of the head-mean q) decides early stopping and the best state goes
to OUT/best. OUT also holds config.json, split.json (problem id -> split) and log.jsonl.

``--state-baseline`` trains the per-state baseline V(s): same backbone,
LoRA, heads and split, but only the labels of the no-tool actions (answer_now, and think_more when labeled), so those
outputs give V̂(s) and the tool outputs are untrained. ``evaluate_scorer.py`` reads it as Δ̂(s, o) = −V̂(s) and
q̂(s, a) = V̂(s) for every action, with V̂ the baseline's value of the chosen reference. This is CARL's
V(s) adapted to a before-execution comparison: CARL's Δ̂ = V(state after the tool) − V(now) sees the tool output.

Cross-validation, table offset and probe:
- ``--folds 5 --fold k``: 5-fold cross-validation split (``scorer.fold_split``) instead of the seed's 80/10/10.
- ``--table-offset``: start from the (dataset, action) mean table of the fold's train and val problems (OUT/table.json,
  the same table the baseline predicts with): logits = logit(table q) + heads, last head layers zero at the start; val is also
  measured at step 0 (epoch -1), so OUT/best can be the untrained scorer, which predicts the table.
- ``--frozen --features FEATURES.pt``: probe, no LoRA; the heads train on hidden states cached by
  ``cache_features.py``, so the policy is not loaded. Everything else (heads, loss, lr, early stopping) is shared.

``--thin-ties SHARE`` thins the train points whose best tool q equals the best
of answer_now/think_more (q = s/n), per (dataset, sub-type: all actions equal / only the maxima equal), shuffled with
``--thin-seed``: ``--thin-ties SHARE`` keeps SHARE of them in every stratum; ``--thin-tie-ratio R``
keeps as many as make their share of the train points R times the original, spread over the strata by size; val and test are untouched and the table offset comes from the thinned train + val.
``--thin-random`` is the size-matched control: the same number of points per dataset, drawn from all train points.
The counts go to config.json under "thin".

Two nodes (data parallel; launch with ``run_ddp.sh``): every rank shuffles the same order, and within each
optimizer step's ``--grad-accum`` points rank r takes the positions with position % world == r. Each point's loss is
divided by trials × grad_accum as before and the gradients are summed across ranks, so every step sees the same points
with the same weights as a single run. LoRA dropout is seeded per point (seed, epoch, position), so a point gets the
same mask on either rank. Val NLL is summed across ranks. Every rank writes the checkpoint (config, split, table, best/)
to OUT on its own node, so a two-node ``evaluate_scorer.py`` finds it on both; only rank 0 writes log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branches import BASES  # noqa: E402
from scorer import (ACTIONS, POLICY, Encoder, Scorer, binomial_nll, dist_setup, fold_split, load_backbone,  # noqa: E402
                    load_examples, mean_table, memory_guard, save_scorer, split_ids, table_offset)


NO_TOOL = ("answer_now", "think_more")
HEAD_HIDDEN = 512        # hidden width of each head MLP
LORA_ALPHA, LORA_DROPOUT = 32, 0.05


def tie_kind(labels: dict) -> str | None:
    """'all_equal' or 'max_equal' when the best tool q equals the best answer_now/think_more q; None otherwise
    (also when the point has no tool label or no answer_now/think_more label)."""
    q = {a: s / n for a, (s, n) in labels.items() if n}
    tools, bases = [v for a, v in q.items() if a not in NO_TOOL], [v for a, v in q.items() if a in NO_TOOL]
    if not tools or not bases or max(tools) != max(bases):
        return None
    return "all_equal" if len(set(q.values())) == 1 else "max_equal"


def thin_train(train: list[dict], share: float, seed: int, random_control: bool,
               ratio: float | None = None) -> tuple[list[dict], dict]:
    """Train points after thinning the ties (or, as a control, as many random points per dataset), and the counts.

    ``ratio`` None (``--thin-ties``): every (dataset, sub-type) stratum keeps floor(n × ``share``) ties.
    ``ratio`` R (``--thin-tie-ratio``, e.g. 0.5 halves the ties' share of the train points): with tie
    share p0 = t/(t+u) the target is R·p0, which needs K = round(target·u/(1 − target)) ties; K is spread over the strata
    in proportion to their sizes (largest remainder, ties broken by stratum name), so the sub-type mix stays the same."""
    strata = defaultdict(list)
    for e in train:
        if kind := tie_kind(e["labels"]):
            strata[(e["dataset"], kind)].append(e)
    ties = sum(len(rows) for rows in strata.values())
    others = len(train) - ties
    out = {"seed": seed, "random_control": random_control, "train_before": len(train), "tie_share_before": round(ties / len(train), 4)}
    if ratio is None:
        keep_n = {key: math.floor(len(rows) * share) for key, rows in strata.items()}
        out["share"] = share
    else:
        target = ratio * ties / len(train)
        k = round(target * others / (1 - target))
        quota = {key: k * len(rows) / ties for key, rows in strata.items()}
        keep_n = {key: math.floor(q) for key, q in quota.items()}
        for key in sorted(quota, key=lambda key: (keep_n[key] - quota[key], key))[: k - sum(keep_n.values())]:
            keep_n[key] += 1
        out |= {"ratio": ratio, "tie_share_target": round(target, 4), "ties_to_keep": k}
    rng, removed, report = random.Random(f"thin:{seed}"), [], {}
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda e: e["dp_id"])
        rng.shuffle(rows)
        removed += rows[keep_n[key]:]
        report[f"{key[0]} {key[1]}"] = {"total": len(rows), "kept": keep_n[key], "removed": len(rows) - keep_n[key]}
    out |= {"tie_strata": report, "removed": len(removed)}
    if random_control:
        per_dataset, rng, removed = Counter(e["dataset"] for e in removed), random.Random(f"thin-random:{seed}"), []
        for dataset in sorted(per_dataset):
            rows = sorted((e for e in train if e["dataset"] == dataset), key=lambda e: e["dp_id"])
            rng.shuffle(rows)
            removed += rows[:per_dataset[dataset]]
        out["random_removed_per_dataset"] = dict(sorted(per_dataset.items()))
        out["random_removed_ties"] = sum(tie_kind(e["labels"]) is not None for e in removed)
    gone = {e["dp_id"] for e in removed}
    kept = [e for e in train if e["dp_id"] not in gone]
    out["train_after"] = len(kept)
    out["tie_share_after"] = round(sum(tie_kind(e["labels"]) is not None for e in kept) / len(kept), 4)
    return kept, out


def make_logits(scorer: Scorer, encoder: Encoder | None, features: dict | None, offsets: dict | None):
    """Function example -> logits [heads, actions]: through the policy, or from cached features; plus the table offset."""
    def logits(e: dict) -> torch.Tensor:
        offset = offsets[e["dp_id"]] if offsets else None
        if features is not None:
            return scorer.heads_only(features[e["dp_id"]], offset)
        return scorer({k: v.cuda() for k, v in encoder(e).items()}, offset)
    return logits


def evaluate_nll(scorer: Scorer, logits, examples: list[dict], rank: int = 0, world: int = 1) -> float:
    """Per-continuation NLL of the head-mean q over ``examples`` (split across ranks and summed)."""
    scorer.eval()
    total, trials = 0.0, 0
    with torch.inference_mode():
        for e in examples[rank::world]:
            q = torch.sigmoid(logits(e)).mean(0).clamp(1e-6, 1 - 1e-6)
            for action, (s, n) in e["labels"].items():
                p = q[ACTIONS.index(action)].item()
                total -= s * math.log(p) + (n - s) * math.log(1 - p)
                trials += n
    scorer.train()
    if world > 1:
        import torch.distributed as dist

        sums = torch.tensor([total, trials], dtype=torch.float64, device="cuda")
        dist.all_reduce(sums)
        total, trials = sums.tolist()
    return total / trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stage-2 scorer.")
    parser.add_argument("--data", nargs=2, action="append", type=Path, required=True, metavar=("POINTS", "LABELS"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=str(POLICY))
    parser.add_argument("--seed", type=int, default=0, help="split, head init, data order")
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2, help="val runs without val NLL improvement before stopping")
    parser.add_argument("--val-every", type=int, default=0, help="validate every N optimizer steps (0 = after each epoch)")
    parser.add_argument("--state-baseline", action="store_true", help="train V(s) on the no-tool labels only")
    parser.add_argument("--folds", type=int, default=0, help="cross-validation folds (0 = the seed's 80/10/10 split)")
    parser.add_argument("--fold", type=int, default=0, help="which fold is the test split")
    parser.add_argument("--table-offset", action="store_true", help="start from the (dataset, action) mean table")
    parser.add_argument("--frozen", action="store_true", help="probe: no LoRA, heads on --features")
    parser.add_argument("--features", type=Path, help="cache_features.py output (required with --frozen)")
    parser.add_argument("--limit", type=int, default=0, help="smoke: use only this many train points (0 = all)")
    parser.add_argument("--thin-ties", type=float, default=0.0, help="keep this share of train ties (0 = off)")
    parser.add_argument("--thin-seed", type=int, default=0)
    parser.add_argument("--thin-tie-ratio", type=float, default=0.0,
                        help="multiply the ties' share of train by this (0 = off)")
    parser.add_argument("--thin-random", action="store_true", help="control: remove as many random train points per dataset")
    args = parser.parse_args()
    if args.frozen != bool(args.features):
        raise SystemExit("--frozen and --features go together")
    if args.folds and not 0 <= args.fold < args.folds:
        raise SystemExit("--fold must be in [0, --folds)")
    rank, world = dist_setup()
    lead = rank == 0  # only rank 0 logs; every rank writes the checkpoint files to OUT on its own node
    args.out.mkdir(parents=True, exist_ok=False)

    examples = load_examples([tuple(pair) for pair in args.data])
    assignment = fold_split(examples, args.folds, args.fold, args.seed) if args.folds else split_ids(examples, args.seed)
    splits = {name: [e for e in examples if assignment[e["id"]] == name] for name in ("train", "val", "test")}
    if args.state_baseline:
        # After the split, so the baseline sees exactly the scorer's problems in each split.
        splits = {name: [e | {"labels": {a: v for a, v in e["labels"].items() if a in BASES}} for e in rows
                         if "answer_now" in e["labels"]] for name, rows in splits.items()}
    thin = None
    if args.thin_ties or args.thin_tie_ratio:
        if args.state_baseline or (args.thin_ties and args.thin_tie_ratio):
            raise SystemExit("--thin-ties / --thin-tie-ratio: one of them, and not with --state-baseline")
        splits["train"], thin = thin_train(splits["train"], args.thin_ties, args.thin_seed, args.thin_random,
                                           args.thin_tie_ratio or None)
    table = mean_table(splits["train"] + splits["val"]) if args.table_offset else None  # before --limit: same table
    if args.limit:
        splits["train"] = splits["train"][:args.limit]
    counts = {name: {"points": len(rows), "problems": len({e["id"] for e in rows})} for name, rows in splits.items()}
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "data": [[str(p), str(l)] for p, l in args.data], "actions": ACTIONS, "splits": counts, "world_size": world,
              "thin": thin, "head_hidden": HEAD_HIDDEN, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "split.json").write_text(json.dumps(assignment, indent=0) + "\n")
    if lead:
        print(json.dumps(counts), flush=True)
    offsets = None
    if table:
        (args.out / "table.json").write_text(json.dumps(table, indent=1) + "\n")
        offsets = {e["dp_id"]: table_offset(table, e["dataset"]).cuda() for rows in splits.values() for e in rows}

    if args.frozen:
        cache = torch.load(args.features)
        if cache["model"] != args.model:
            raise SystemExit(f"--features were cached from {cache['model']}, not {args.model}")
        missing = [e["dp_id"] for rows in splits.values() for e in rows if e["dp_id"] not in cache["features"]]
        if missing:
            raise SystemExit(f"{len(missing)} points have no cached features, e.g. {missing[:3]}")
        encoder, features = None, {k: v.cuda() for k, v in cache["features"].items()}
        scorer = Scorer(None, args.heads, HEAD_HIDDEN, args.seed, width=cache["width"], zero_last=bool(table)).cuda().train()
        lora_params = []
    else:
        encoder, features = Encoder(args.model), None
        torch.manual_seed(args.seed)  # LoRA A is drawn at random: same start on every rank and in a single run
        backbone = load_backbone(args.model, lora=dict(r=args.lora_r, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT))
        backbone.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        scorer = Scorer(backbone, args.heads, HEAD_HIDDEN, args.seed, zero_last=bool(table)).cuda().train()
        lora_params = [p for p in backbone.parameters() if p.requires_grad]
    logits = make_logits(scorer, encoder, features, offsets)
    groups = [{"params": lora_params, "lr": args.lr}] if lora_params else []
    optimizer = torch.optim.AdamW(groups + [{"params": scorer.heads.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    trainable = lora_params + list(scorer.heads.parameters())
    if world > 1:  # every rank starts from rank 0's LoRA and heads
        import torch.distributed as dist

        for p in trainable:
            dist.broadcast(p.data, 0)
    if lead:
        print(f"trainable: lora {sum(p.numel() for p in lora_params):,} heads {sum(p.numel() for p in scorer.heads.parameters()):,}"
              f"; world size {world}", flush=True)

    log = open(args.out / "log.jsonl", "w") if lead else None
    best, stale, step = math.inf, 0, 0
    rng = random.Random(args.seed)
    start = time.time()

    def validate(epoch: int) -> bool:
        """Logs val NLL, saves OUT/best on improvement; True when patience is used up."""
        nonlocal best, stale
        val = evaluate_nll(scorer, logits, splits["val"], rank, world)
        improved = val < best
        row = {"epoch": epoch, "step": step, "val_nll": round(val, 4), "best": improved, "seconds": round(time.time() - start)}
        if lead:
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        if improved:
            best, stale = val, 0
            save_scorer(scorer, args.out / "best")  # on every rank: the weights are identical after the summed gradients
        else:
            stale += 1
        return stale >= args.patience

    if table:  # step 0 is the table itself; early stopping may keep it when training never beats it on val
        validate(-1)
    stop = False
    for epoch in range(args.epochs):
        order = splits["train"][:]
        rng.shuffle(order)
        running, running_trials = 0.0, 0
        for i, e in enumerate(order):
            if (i % args.grad_accum) % world == rank:  # this rank's share of the step's points
                trials = sum(n for _, n in e["labels"].values())
                # LoRA dropout masks seeded per point, so a point gets the same mask on whichever rank runs it
                torch.manual_seed(args.seed * 1_000_003 + epoch * 100_003 + i)
                loss = binomial_nll(logits(e), e["labels"])
                (loss / trials / args.grad_accum).backward()
                running += loss.item()
                running_trials += trials
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(order):
                if world > 1:  # sum the step's gradients over ranks (a rank may have had no point in a short last step)
                    for p in trainable:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    memory_guard()
                    if world > 1:
                        sums = torch.tensor([running, running_trials], dtype=torch.float64, device="cuda")
                        dist.all_reduce(sums)
                        running, running_trials = sums.tolist()
                    row = {"epoch": epoch, "step": step, "train_nll": round(running / running_trials, 4),
                           "seconds": round(time.time() - start)}
                    if lead:
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(json.dumps(row), flush=True)
                    running, running_trials = 0.0, 0
                if args.val_every and step % args.val_every == 0:
                    stop = validate(epoch)
                    if stop:
                        break
        if stop or (not args.val_every and validate(epoch)):
            break
    if lead:
        log.close()


if __name__ == "__main__":
    main()
