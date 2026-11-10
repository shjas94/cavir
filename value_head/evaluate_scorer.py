#!/usr/bin/env python3
"""Evaluation of a trained scorer.

Run with the train_cluster env from sft/LlamaFactory:

    cd /workspace/cavir/sft/LlamaFactory
    /root/miniconda3/envs/train_cluster/bin/python ../../value_head/evaluate_scorer.py \
        --ckpt ../../value_head/outputs/scorer_<name> --branches R1/branches/branches.jsonl --branches R2/... \
        --out ../../value_head/outputs/scorer_<name>/eval_test

Predicts q(a) for every decision point of ``--split`` (default test) with the checkpoint's own data and split, and
compares predicted Δ = q(tool) − q(reference) (head mean) with the empirical Δ from ``--branches``, per
(decision point, tool) pair. ``--reference`` is answer_now (default), think_more, best (the better of the two, the
empirical side cross-fitted) or best_raw (``analyze_branches.deltas``); the predicted side of best and
best_raw is q̂(tool) − max(q̂(answer_now), q̂(think_more)). Metrics:
- main: Spearman over all pairs and per dataset, with the noise ceiling (square root of the split-half
  reliability, K=16, ``analyze_branches.reliability``) recomputed on the same pairs from ``--branches``;
- secondary: sign AUROC (empirical Δ > 0 is positive, scored by predicted Δ), overall, per dataset and per
  pre-screen stratum (mixed / all_correct / all_wrong);
- argmax agreement within a decision point (predicted best action is among the empirically best, ties included;
  predicted ties count by the share of tied actions that hit), with the rate a uniform random pick would reach;
- within-decision-point tool ranking: mean per-point Spearman of predicted vs
  empirical Δ over points with ≥ 2 tools and unequal Δ, and the pooled order agreement over tool pairs with unequal Δ
  (a predicted tie counts 1/2);
- q̂ vs q Spearman over all (decision point, action) cells, and on cells centred on their decision point's mean.
A predictor that gives every tool of a point the same value has no within-point order: its per-point and centred
Spearman count as 0. The headline metrics get 95% bootstrap intervals over problem ids. Writes OUT/predictions.jsonl
and OUT/metrics.json.

Main metric: the order agreement over
all action pairs of a decision point (answer_now, think_more and the tools) on the empirical q = s/n; pairs with
equal q are skipped, a predicted tie counts 1/2, pooled over pairs; chance 0.5 (``action_order``). Secondary: routing
regret at λ = 0, the mean of max q − q of the predicted best action (``routing_regret``), with uniform random and
always-answer_now for comparison. ``--table-baseline`` predicts with the checkpoint's table.json (the mean table of
``train_scorer.py --table-offset``) instead of the model. A ``--frozen`` checkpoint runs its heads on the cached
features. ``evaluate_folds.py`` pools the folds of a cross-validation.

Reported metrics and verdict (``main_block``, first key of metrics.json and the printed
output): judgement = action_order agreement, outcome = routing regret at λ = 0, diagnostic = q̂ vs q Spearman over all
cells. With the mean table as the baseline, the verdict passes when the agreement's scorer − table CI lower bound is
> 0 and the routing regret's is ≤ 0. The other metrics above are still computed, as auxiliary.

Two nodes (``run_ddp.sh``): the model's predictions are split across ranks and gathered on rank 0, which does
the rest (branches, metrics, writing OUT); other ranks need only the checkpoint, data and images.

A ``--state-baseline`` checkpoint (``train_scorer.py``) is read as Δ̂(s, o) = −V̂(s) and q̂(s, a) = V̂(s), with
V̂ the baseline's q̂ of the reference (the max of answer_now and think_more for best). ``--predictions`` recomputes the metrics from a saved predictions.jsonl without the model;
``--baseline`` (a baseline's predictions.jsonl on the same points) adds the baseline's metrics and the
scorer − baseline difference with a paired bootstrap CI over problem ids.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branches import BASES, EXCLUDED, FUNCTION_RE, REFERENCES, _ranks, deltas, reliability, spearman  # noqa: E402


def auroc(scores: list[float], positive: list[bool]) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties; None when one class is empty."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = _ranks(scores)
    return (sum(r for r, p in zip(ranks, positive) if p) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def safe_spearman(xs: list[float], ys: list[float]) -> float | None:
    return spearman(xs, ys) if len(xs) >= 5 and len(set(xs)) > 1 and len(set(ys)) > 1 else None


def rho0(xs: list[float], ys: list[float]) -> float | None:
    """Spearman for a within-point comparison: 0 when the prediction has no order, None when the truth has none."""
    if len(set(ys)) < 2:
        return None
    return spearman(xs, ys) if len(set(xs)) > 1 else 0.0


def reference_value(p: dict) -> float:
    """q̂ of the reference set by ``main`` in ``p["reference"]``."""
    bases = (p["reference"],) if p["reference"] in BASES else BASES
    return max(p["q_mean"][b] for b in bases)


def pred_q(p: dict) -> dict[str, float]:
    """Head-mean q̂ per action; a state baseline gives V̂(s) to every action."""
    if p.get("kind") == "state_baseline":
        return {a: reference_value(p) for a in p["q_mean"]}
    return p["q_mean"]


def pred_delta(p: dict, tool: str) -> float:
    if p.get("kind") == "state_baseline":
        return -reference_value(p)
    return p["q_mean"][tool] - reference_value(p)


def make_pairs(predictions: list[dict]) -> list[dict]:
    """(decision point, tool) pairs with the empirical Δ ``main`` put in ``p["delta"]``."""
    return [{"dp_id": p["dp_id"], "id": p["id"], "dataset": p["dataset"], "stratum": p["stratum"], "tool": tool,
             "delta": delta, "pred": pred_delta(p, tool)} for p in predictions for tool, delta in p["delta"].items()]


def pair_metrics(pairs: list[dict]) -> dict:
    preds, deltas = [p["pred"] for p in pairs], [p["delta"] for p in pairs]
    rho = safe_spearman(preds, deltas)
    auc = auroc(preds, [d > 0 for d in deltas])
    return {"pairs": len(pairs), "positive": sum(d > 0 for d in deltas), "zero": sum(d == 0 for d in deltas),
            "spearman": None if rho is None else round(rho, 4), "sign_auroc": None if auc is None else round(auc, 4)}


def within_point(predictions: list[dict]) -> dict:
    """Tool ranking inside each decision point (predicted vs empirical Δ)."""
    by_point = defaultdict(list)
    for pair in make_pairs(predictions):
        by_point[pair["dp_id"]].append(pair)
    rhos, agree, ordered = [], 0.0, 0
    for rows in by_point.values():
        if len(rows) >= 2 and (rho := rho0([r["pred"] for r in rows], [r["delta"] for r in rows])) is not None:
            rhos.append(rho)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["delta"] != b["delta"]:
                    ordered += 1
                    agree += 0.5 if a["pred"] == b["pred"] else (a["pred"] - b["pred"]) * (a["delta"] - b["delta"]) > 0
    return {"points": len(rhos), "spearman_mean": round(sum(rhos) / len(rhos), 4) if rhos else None,
            "tool_pairs": ordered, "order_agreement": round(agree / ordered, 4) if ordered else None}


def q_cells(predictions: list[dict]) -> dict:
    """q̂ vs q over (decision point, action) cells, raw and centred on each point's mean."""
    emp, pred, emp_c, pred_c = [], [], [], []
    for p in predictions:
        q_hat = pred_q(p)
        acts = [a for a, (s, n) in p["labels"].items() if n]
        e = [p["labels"][a][0] / p["labels"][a][1] for a in acts]
        h = [q_hat[a] for a in acts]
        emp += e
        pred += h
        emp_c += [x - sum(e) / len(e) for x in e]
        pred_c += [x - sum(h) / len(h) for x in h]
    rho, rho_c = safe_spearman(pred, emp), rho0(pred_c, emp_c) if len(emp_c) >= 5 else None
    return {"cells": len(emp), "spearman": None if rho is None else round(rho, 4),
            "spearman_centred": None if rho_c is None else round(rho_c, 4)}


def action_order(predictions: list[dict]) -> dict:
    """Main metric: order agreement over all action pairs of a point on the empirical q (unequal q only; tie = 1/2)."""
    agree, ordered = 0.0, 0
    for p in predictions:
        q_hat = pred_q(p)
        q = {a: s / n for a, (s, n) in p["labels"].items() if n}
        acts = sorted(q)
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                if q[a] != q[b]:
                    ordered += 1
                    agree += 0.5 if q_hat[a] == q_hat[b] else (q_hat[a] - q_hat[b]) * (q[a] - q[b]) > 0
    return {"action_pairs": ordered, "agreement": round(agree / ordered, 4) if ordered else None}


def routing_regret(predictions: list[dict]) -> dict:
    """Mean of max q − q(predicted best action) at λ = 0 (predicted ties averaged), with two reference policies."""
    regret, uniform, answer_now, n = 0.0, 0.0, 0.0, 0
    for p in predictions:
        q = {a: s / k for a, (s, k) in p["labels"].items() if k}
        if not q:
            continue
        q_hat = pred_q(p)
        top = [a for a in q if q_hat[a] == max(q_hat[b] for b in q)]
        best = max(q.values())
        regret += best - sum(q[a] for a in top) / len(top)
        uniform += best - sum(q.values()) / len(q)
        answer_now += best - q["answer_now"] if "answer_now" in q else 0.0
        n += 1
    r = lambda v: round(v / n, 4) if n else None
    return {"points": n, "regret": r(regret), "uniform_random": r(uniform), "always_answer_now": r(answer_now)}


def headline(predictions: list[dict]) -> dict[str, float | None]:
    m, w, q = pair_metrics(make_pairs(predictions)), within_point(predictions), q_cells(predictions)
    return {"action_order_agreement": action_order(predictions)["agreement"], "routing_regret": routing_regret(predictions)["regret"],
            "spearman": m["spearman"], "sign_auroc": m["sign_auroc"], "within_spearman_mean": w["spearman_mean"],
            "within_order_agreement": w["order_agreement"], "q_spearman": q["spearman"], "q_spearman_centred": q["spearman_centred"]}


def bootstrap(predictions: list[dict], reps: int, seed: int, baseline: list[dict] | None = None) -> dict:
    """95% intervals of the headline metrics over problem ids; with ``baseline``, also of scorer − baseline (paired)."""
    by_problem, base_by_problem = defaultdict(list), defaultdict(list)
    for p in predictions:
        by_problem[p["id"]].append(p)
    for p in baseline or []:
        base_by_problem[p["id"]].append(p)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    values, diffs = defaultdict(list), defaultdict(list)
    for _ in range(reps):
        sample = rng.choices(problems, k=len(problems))
        m = headline([p for pid in sample for p in by_problem[pid]])
        b = headline([p for pid in sample for p in base_by_problem[pid]]) if baseline else {}
        for key, v in m.items():
            if v is not None:
                values[key].append(v)
                if b.get(key) is not None:
                    diffs[key].append(v - b[key])
    ci = lambda v: [round(sorted(v)[int(0.025 * len(v))], 4), round(sorted(v)[int(0.975 * len(v)) - 1], 4)] if v else None
    out = {"reps": reps, "ci95": {key: ci(values[key]) for key in values}}
    if baseline:
        out["diff_ci95"] = {key: ci(diffs[key]) for key in diffs}
    return out


def argmax_agreement(predictions: list[dict]) -> dict:
    hits, chance, n = 0.0, 0.0, 0
    for p in predictions:
        actions = list(p["labels"])
        if len(actions) < 2:
            continue
        rate = {a: s / k for a, (s, k) in p["labels"].items()}
        best = {a for a in actions if rate[a] == max(rate.values())}
        q_hat = pred_q(p)
        top = [a for a in actions if q_hat[a] == max(q_hat[b] for b in actions)]
        hits += sum(a in best for a in top) / len(top)
        chance += len(best) / len(actions)
        n += 1
    return {"points": n, "agreement": round(hits / n, 4) if n else None, "uniform_random": round(chance / n, 4) if n else None}


def report(predictions: list[dict], ceiling: dict, reps: int, baseline: list[dict] | None = None) -> dict:
    pairs = make_pairs(predictions)
    datasets = sorted({p["dataset"] for p in pairs})
    strata = sorted({p["stratum"] for p in pairs})
    return {
        "all": pair_metrics(pairs) | {"ceiling": ceiling.get("all")},
        "by_dataset": {d: pair_metrics([p for p in pairs if p["dataset"] == d]) | {"ceiling": ceiling.get(d)} for d in datasets},
        "by_stratum": {s: pair_metrics([p for p in pairs if p["stratum"] == s]) for s in strata},
        "action_order": action_order(predictions),
        "routing_regret": routing_regret(predictions),
        "argmax": argmax_agreement(predictions),
        "within_point": within_point(predictions),
        "q_cells": q_cells(predictions),
        "bootstrap": bootstrap(predictions, reps, 0, baseline),
    }


def main_block(m: dict, base: dict | None, base_kind: str | None) -> dict:
    """The three reported metrics and, against the mean table, the verdict.

    Judgement: action_order agreement; outcome: routing regret (λ = 0); diagnostic: q̂ vs q Spearman over all cells.
    The scorer passes when the scorer − table difference of the agreement has a 95% CI lower bound > 0
    and that of the routing regret has a lower bound ≤ 0 (not clearly worse than the table). Everything else in
    ``report`` is auxiliary and stays in metrics.json under its old key.
    """
    values = lambda r: {"action_order_agreement": r["action_order"]["agreement"], "routing_regret": r["routing_regret"]["regret"],
                        "q_spearman": r["q_cells"]["spearman"]}
    ci, diff = m["bootstrap"]["ci95"], m["bootstrap"].get("diff_ci95", {})
    out = {}
    for role, key in (("judgement", "action_order_agreement"), ("outcome", "routing_regret"), ("diagnostic", "q_spearman")):
        out[role] = {"metric": key, "value": values(m)[key], "ci95": ci.get(key)}
        if base:
            out[role] |= {"baseline_" + str(base_kind): values(base)[key], "diff_ci95": diff.get(key)}
    out["outcome"] |= {k: m["routing_regret"][k] for k in ("uniform_random", "always_answer_now")}
    if base_kind == "table" and diff.get("action_order_agreement") and diff.get("routing_regret"):
        better, not_worse = diff["action_order_agreement"][0] > 0, diff["routing_regret"][0] <= 0
        out["verdict"] = {"action_order_above_table": better, "routing_regret_not_worse_than_table": not_worse,
                          "pass": better and not_worse}
    else:
        out["verdict"] = "none: needs the mean table's predictions as the baseline"
    out["auxiliary"] = ["all", "by_dataset", "by_stratum", "argmax", "within_point", "q_cells.spearman_centred"]
    return out


def load_branches(paths: list[Path], dp_ids: set[str]) -> dict:
    """Continuations of ``dp_ids`` grouped by (dp_id, action), with analyze_branches' reclassification of cut-off calls."""
    groups = defaultdict(list)
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            if row["dp_id"] not in dp_ids:
                continue
            if row["first_action"] == "tool_call_unparsed" and (m := FUNCTION_RE.search(row["text"])) and m.group(1).strip() == row["action"]:
                row["finish"], row["correct"] = "length" if row["tokens"] >= 4096 else "no_answer", False
            groups[(row["dp_id"], row["action"])].append({k: row[k] for k in ("k", "finish", "correct")})
    return groups


def predict(args: argparse.Namespace, config: dict, examples: list[dict], rank: int = 0, world: int = 1) -> list[dict]:
    """q̂ per example; with several ranks each predicts every world-th example and all ranks get the full list."""
    import torch

    from scorer import ACTIONS, Encoder, gather_to_all, load_scorer, table_offset

    actions = tuple(config.get("actions", ACTIONS))  # older checkpoints have 8 outputs, without think_more
    table = json.loads((args.ckpt / "table.json").read_text()) if config.get("table_offset") else None
    if config.get("frozen"):
        cache = torch.load(config["features"])
        encoder, features = None, cache["features"]
        scorer = load_scorer(args.ckpt / "best", None, config["heads"], config["head_hidden"], len(actions), width=cache["width"])
    else:
        encoder, features = Encoder(args.model), None
        scorer = load_scorer(args.ckpt / "best", args.model, config["heads"], config["head_hidden"], len(actions))
    out = []
    with torch.inference_mode():
        for i, e in list(enumerate(examples))[rank::world]:
            offset = table_offset(table, e["dataset"], actions).cuda() if table else None
            if features is not None:
                logits = scorer.heads_only(features[e["dp_id"]].cuda(), offset)
            else:
                logits = scorer({k: v.cuda() for k, v in encoder(e).items()}, offset)
            q = torch.sigmoid(logits).cpu().tolist()
            q_heads = [{a: round(row[j], 5) for j, a in enumerate(actions)} for row in q]
            q_mean = {a: round(sum(h[a] for h in q_heads) / len(q_heads), 5) for a in actions}
            out.append((i, {k: e[k] for k in ("dp_id", "id", "dataset", "stratum", "labels")}
                        | {"kind": "state_baseline" if config.get("state_baseline") else "scorer", "q_mean": q_mean, "q_heads": q_heads}))
            if len(out) % 50 == 0:
                print(f"[rank {rank}] predicted {len(out)}/{len(examples[rank::world])}", flush=True)
    return [p for _, p in sorted(gather_to_all(out, world), key=lambda x: x[0])]


def table_predictions(ckpt: Path, examples: list[dict]) -> list[dict]:
    """The (dataset, action) mean table of ``ckpt`` (table.json) as predictions; an action the table lacks gets 0.5."""
    table = json.loads((ckpt / "table.json").read_text())
    return [{k: e[k] for k in ("dp_id", "id", "dataset", "stratum", "labels")}
            | {"kind": "table", "q_mean": {a: table.get(e["dataset"], {}).get(a, 0.5) for a in set(e["labels"]) | set(BASES)}}
            for e in examples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation of a trained scorer.")
    parser.add_argument("--ckpt", required=True, type=Path, help="train_scorer.py output directory")
    parser.add_argument("--branches", action="append", type=Path, required=True, help="branches.jsonl of every input run")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--model", help="defaults to the checkpoint's")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--predictions", type=Path, help="reuse this predictions.jsonl instead of running the model")
    parser.add_argument("--baseline", type=Path, help="predictions.jsonl of a --state-baseline checkpoint on the same points")
    parser.add_argument("--reference", default="answer_now", choices=REFERENCES, help="what a tool's Δ is measured against")
    parser.add_argument("--table-baseline", action="store_true", help="predict with the checkpoint's table.json, no model")
    args = parser.parse_args()

    from scorer import dist_setup, load_examples

    # Two nodes (run_ddp.sh): only the model's predictions are split; rank 0 does everything else.
    rank, world = dist_setup()
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
    elif args.predictions or args.table_baseline:
        return

    config = json.loads((args.ckpt / "config.json").read_text())
    args.model = args.model or config["model"]
    assignment = json.loads((args.ckpt / "split.json").read_text())
    examples = [e for e in load_examples([(Path(p), Path(l)) for p, l in config["data"]]) if assignment[e["id"]] == args.split]
    model_predictions = None
    if world > 1 and not (args.predictions or args.table_baseline):
        model_predictions = predict(args, config, examples, rank, world)
        if rank != 0:
            return

    # Labels must be what the branches give; the ceiling is computed from the same continuations.
    groups = load_branches(args.branches, {e["dp_id"] for e in examples})
    for e in examples:
        for action, (s, n) in e["labels"].items():
            valid = [r for r in groups[(e["dp_id"], action)] if r["finish"] not in EXCLUDED]
            if (sum(bool(r["correct"]) for r in valid), len(valid)) != (s, n):
                raise SystemExit(f"labels of {e['dp_id']} {action} differ from --branches")

    if args.predictions:
        predictions = [json.loads(line) for line in open(args.predictions)]
    elif args.table_baseline:
        predictions = table_predictions(args.ckpt, examples)
    else:
        predictions = model_predictions if model_predictions is not None else predict(args, config, examples)
    with open(args.out / "predictions.jsonl", "w") as handle:
        for p in predictions:
            handle.write(json.dumps(p) + "\n")
    expected = {e["dp_id"]: e["labels"] for e in examples}
    baseline = [json.loads(line) for line in open(args.baseline)] if args.baseline else None
    for name, rows in (("predictions", predictions), ("baseline", baseline or [])):
        if rows and {p["dp_id"]: {a: tuple(v) for a, v in p["labels"].items()} for p in rows} != expected:
            raise SystemExit(f"{name}: decision points or labels differ from the checkpoint's {args.split} split")

    empirical = defaultdict(dict)
    for (dp, tool), delta in deltas(groups, reference=args.reference).items():
        empirical[dp][tool] = delta
    needed = {args.reference} if args.reference in BASES else set(BASES)
    for p in predictions + (baseline or []):
        if not needed <= set(p["q_mean"]):
            raise SystemExit(f"--reference {args.reference} needs q̂ of {sorted(needed)}; {p['dp_id']} has {sorted(p['q_mean'])}")
        p["reference"], p["delta"] = args.reference, empirical[p["dp_id"]]

    dataset_of = {e["dp_id"]: e["dataset"] for e in examples}
    k_total = max(r["k"] for rows in groups.values() for r in rows) + 1
    ceiling = reliability(groups, k_total, dataset_of, args.reference).get(f"K={k_total}", {})
    metrics = {"ckpt": str(args.ckpt), "split": args.split, "points": len(predictions), "problems": len({e["id"] for e in examples}),
               "k": k_total, "reference": args.reference, "kind": predictions[0].get("kind", "scorer"),
               "predictions_from": str(args.predictions or ("table.json" if args.table_baseline else "model"))}
    metrics |= report(predictions, ceiling, args.bootstrap, baseline)
    if baseline:
        metrics["baseline"] = {"predictions": str(args.baseline)} | report(baseline, ceiling, args.bootstrap)
    metrics = {"main": main_block(metrics, metrics.get("baseline"), baseline[0].get("kind", "scorer") if baseline else None)} | metrics
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("points", "problems", "main")}, indent=1))


if __name__ == "__main__":
    main()
