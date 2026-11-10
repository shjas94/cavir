#!/usr/bin/env python3
"""Drop cavir_formatted.jsonl records that have a quality defect.

For each dataset dir, re-runs the deterministic audit rules
(``audit_cavir_coldstart.rule_issues``) on ``cavir_formatted.jsonl`` (never modified)
and writes the records without a dropping issue to ``<output-name>.jsonl``, plus
``cavir_coldstart/<output-name>_report.json`` with counts per issue type, variant, Case and
selection. Existing outputs are never overwritten: pick a new ``--output-name`` when the
rules or the data change. Rules run directly instead of reading ``audit.jsonl``, so the
filter cannot go stale relative to the data. Defects only an LLM can see are not filtered here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_cavir_coldstart import EVAL_RE, rule_issues
from build_cavir_coldstart import MAX_THINK_MORE

# Every audit rule drops the record except this one: it is a weak signal with low
# precision (a correct statement often cites only the inputs).
KEEP_TYPES = {"statement_answer_number_mismatch"}


def selections(record: dict) -> list[str]:
    return [line.split(": ", 1)[1] for message in record["conversations"] if message["from"] == "gpt"
            for block in EVAL_RE.findall(message["value"]) for line in block.splitlines()
            if line.startswith("selection: ")]


def output_paths(dataset_dir: Path, output_name: str) -> tuple[Path, Path]:
    return dataset_dir / f"{output_name}.jsonl", dataset_dir / "cavir_coldstart" / f"{output_name}_report.json"


def filter_dataset(dataset_dir: Path, tau: float, output_name: str) -> dict:
    output, report_path = output_paths(dataset_dir, output_name)
    records = [json.loads(line) for line in open(dataset_dir / "cavir_formatted.jsonl")]
    kept, dropped_types, only_reason = [], Counter(), Counter()
    dropped_variants, kept_variants, kept_cases, kept_selections = Counter(), Counter(), Counter(), Counter()
    for record in records:
        reasons = {i["type"] for i in rule_issues(record, tau, MAX_THINK_MORE)} - KEEP_TYPES
        if reasons:
            dropped_types.update(reasons)
            dropped_variants[record["meta"]["variant"]] += 1
            if len(reasons) == 1:
                only_reason.update(reasons)
            continue
        kept.append(record)
        kept_variants[record["meta"]["variant"]] += 1
        kept_cases[record["meta"]["case"]] += 1
        kept_selections.update(selections(record))

    with open(output, "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "dataset": dataset_dir.name, "tau": tau, "input": len(records), "kept": len(kept),
        "dropped": len(records) - len(kept), "kept_issue_types": sorted(KEEP_TYPES),
        "dropped_records_by_type": dict(dropped_types.most_common()),
        "dropped_records_by_only_type": dict(only_reason.most_common()),
        "dropped_by_variant": dict(dropped_variants.most_common()),
        "kept_by_variant": dict(kept_variants.most_common()),
        "kept_by_case": {str(k): v for k, v in sorted(kept_cases.items())},
        "kept_selections": {"tool": sum(n for s, n in kept_selections.items() if s not in ("answer_now", "think_more")),
                            "answer_now": kept_selections["answer_now"], "think_more": kept_selections["think_more"]},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop cavir_formatted.jsonl records with audit-rule defects.")
    parser.add_argument("dataset_dirs", nargs="+", type=Path, help="Dirs holding cavir_formatted.jsonl.")
    parser.add_argument("--tau", type=float, default=0.7, help="answer_now threshold τ the data was built with.")
    parser.add_argument("--output-name", default="cavir_filtered", help="Output file stem; must not exist yet.")
    args = parser.parse_args()

    existing = [p for d in args.dataset_dirs for p in output_paths(d.resolve(), args.output_name) if p.exists()]
    if existing:
        raise SystemExit(f"{existing[0]} exists; choose another --output-name")
    total_in = total_kept = 0
    for dataset_dir in args.dataset_dirs:
        report = filter_dataset(dataset_dir.resolve(), args.tau, args.output_name)
        total_in, total_kept = total_in + report["input"], total_kept + report["kept"]
        print(f"{report['dataset']}: kept {report['kept']} / {report['input']}, dropped {report['dropped']} "
              f"{report['dropped_records_by_type']}")
    print(f"total: kept {total_kept} / {total_in} ({total_kept / total_in:.1%})")


if __name__ == "__main__":
    main()
