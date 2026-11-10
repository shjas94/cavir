#!/usr/bin/env python3
r"""Filter incorrect or tool-failed CaViR rollout records.

For every discovered ``results.jsonl`` this script writes:

* ``filtered_results.jsonl``: records whose parsed prediction matches the
  parsed ground truth and whose tool calls all succeeded.
* ``filtered_out.jsonl``: rejected records with their filter diagnostics.

Text comparison is deliberately case-sensitive. Numeric-only answers are
compared by value so formatting such as ``1,000`` versus ``1000`` does not
discard a correct result. Only structural wrappers
(``<answer>``, ``\boxed{}``, CoT ``</think>``, outer LaTeX/Markdown wrappers,
and insignificant whitespace) are removed.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

RESULTS_FILENAME = "results.jsonl"
FILTERED_RESULTS_FILENAME = "filtered_results.jsonl"
REJECTED_FILENAME = "filtered_out.jsonl"
SUMMARY_FILENAME = "filter_summary.json"

THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# GeomVerse CoT ground truths always close with the result in one of two
# sentence forms: ``Therefore the final answer is 86.88.`` or
# ``The answer is \(95^\circ\).``
GEOMVERSE_ANSWER_RE = re.compile(
    r"(?:therefore\s+)?the\s+(?:final\s+)?answer\s+is\b", re.IGNORECASE
)
INLINE_MATH_RE = re.compile(r"^(?:\\\((.*?)\\\)|\\\[(.*?)\\\])$", re.DOTALL)
DEGREE_SUFFIX_RE = re.compile(r"(?:\^\s*(?:\\circ|\{\s*\\circ\s*\})|°)\s*$")
TRAILING_PERIOD_RE = re.compile(r"(?<=[\d)}])\s*\.\s*$")
LATEX_ESCAPE_RE = re.compile(r"\\([%$&#_{}])")
WHITESPACE_RE = re.compile(r"\s+")

TOOL_FAILURE_REASONS = {
    "error",
    "tool result error",
}
TOOL_ERROR_TEXT_RE = re.compile(
    r"^(?:error executing tool\b|\[tool (?:argument parse|resolution) error\]|❌)",
    re.IGNORECASE,
)
LATEX_WRAPPER_COMMANDS = (
    r"\text",
    r"\textbf",
    r"\textit",
    r"\textrm",
    r"\mathrm",
    r"\mathbf",
    r"\mathit",
    r"\operatorname",
)
NUMERIC_ANSWER_RE = re.compile(
    r"^\s*[€£¥]?\s*([+-]?)\s*(\d[\d,]*)(?:\.(\d*))?\s*%?\s*$"
)


def _balanced_braced(text: str, open_brace: int) -> tuple[str, int] | None:
    """Return the content/end index of a balanced braced expression."""
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index], index + 1
    return None


def extract_last_boxed(text: str) -> str | None:
    r"""Extract the final balanced ``\boxed{...}`` expression."""
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(positions):
        open_brace = text.find("{", start)
        balanced = _balanced_braced(text, open_brace)
        if balanced is not None:
            return balanced[0]
    return None


def _unwrap_latex_command(text: str) -> str:
    """Remove an outer presentational LaTeX command while preserving content."""
    stripped = text.strip()
    for command in LATEX_WRAPPER_COMMANDS:
        match = re.fullmatch(re.escape(command) + r"\s*\{(.*)\}", stripped, re.DOTALL)
        if not match:
            continue
        open_brace = stripped.find("{")
        balanced = _balanced_braced(stripped, open_brace)
        if balanced is not None and balanced[1] == len(stripped):
            return balanced[0]
    return stripped


def normalize_answer(text: str) -> str:
    """Normalize answer wrappers and whitespace without changing letter case."""
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()

    previous = None
    while text != previous:
        previous = text
        text = _unwrap_latex_command(text)
        for left, right in (("**", "**"), ("__", "__"), ("`", "`"), ('"', '"'), ("'", "'")):
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                break

    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", " ").replace(r"\ ", " ").replace("~", " ")
    text = LATEX_ESCAPE_RE.sub(r"\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_numeric_answer(text: str) -> Decimal | None:
    """Parse formatting-only numeric variants without interpreting text units."""
    match = NUMERIC_ANSWER_RE.fullmatch(text)
    if not match:
        return None
    sign, integer, fraction = match.groups()
    number = (sign or "") + integer.replace(",", "")
    if fraction is not None and fraction:
        number += "." + fraction
    try:
        return Decimal(number)
    except InvalidOperation:
        return None


def answers_match(prediction: str, ground_truth: str) -> tuple[bool, str]:
    """
    Compare answers exactly for text and numerically for numeric-only answers.

    Case, spelling, punctuation and units remain significant for text answers.
    Numeric-only answers may differ in thousands separators, trailing zeros,
    currency symbols, or a trailing percent sign.
    """
    if prediction == ground_truth:
        return True, "exact"
    prediction_number = parse_numeric_answer(prediction)
    ground_truth_number = parse_numeric_answer(ground_truth)
    if (
        prediction_number is not None
        and ground_truth_number is not None
        and prediction_number == ground_truth_number
    ):
        return True, "numeric"
    return False, "different"


def extract_prediction(record: dict[str, Any]) -> tuple[str | None, str]:
    """Parse the final ``<answer>\boxed{...}</answer>`` prediction."""
    response = None
    for message in reversed(record.get("result") or []):
        content = message.get("content") if isinstance(message, dict) else None
        candidate = content.get("llm_content") if isinstance(content, dict) else None
        if isinstance(candidate, str) and candidate.strip():
            response = candidate
            break

    if response is None:
        return None, "missing_llm_content"

    tags = ANSWER_TAG_RE.findall(response)
    if not tags:
        return None, "missing_answer_tag"

    boxed = extract_last_boxed(tags[-1])
    if boxed is None:
        return None, "missing_boxed_answer"

    answer = normalize_answer(boxed)
    return (answer, "answer_boxed") if answer else (None, "empty_prediction")


def _extract_geomverse_answer(text: str) -> tuple[str | None, str]:
    r"""Extract the short answer closing a GeomVerse CoT ground truth.

    GeomVerse tails never use ``\boxed{}`` or an ``Answer:`` prefix. They end
    with a single result sentence whose number is written either plainly
    (``Therefore the final answer is 86.88.``) or inside inline math, sometimes
    with a degree symbol (``The answer is \(95^\circ\).``). Only the number is
    kept: the sentence period, the math delimiters and ``^\circ`` are dropped.
    """
    markers = list(GEOMVERSE_ANSWER_RE.finditer(text))
    if not markers:
        return None, "geomverse_missing_answer_sentence"

    answer = normalize_answer(text[markers[-1].end() :])
    previous = None
    while answer != previous:
        previous = answer
        answer = TRAILING_PERIOD_RE.sub("", answer).strip()
        math = INLINE_MATH_RE.fullmatch(answer)
        if math:
            answer = (math.group(1) or math.group(2) or "").strip()
        answer = DEGREE_SUFFIX_RE.sub("", answer).strip()
    return (answer, "geomverse_answer_sentence") if answer else (None, "empty_ground_truth")


def _extract_verbose_gt_answer(text: str) -> tuple[str, str]:
    """Conservatively extract a marked answer from a verbose GT tail."""
    bold = [normalize_answer(value) for value in BOLD_RE.findall(text)]
    bold = [value for value in bold if value]
    if len(bold) == 1:
        return bold[0], "single_bold"

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        first_bold = [normalize_answer(value) for value in BOLD_RE.findall(paragraphs[0])]
        first_bold = [value for value in first_bold if value]
        if len(first_bold) == 1:
            return first_bold[0], "first_paragraph_bold"

    return normalize_answer(text), "plain"


def extract_ground_truth(record: dict[str, Any], source: str) -> tuple[str | None, str]:
    """Parse the answer portion of a CoT ground-truth string."""
    is_geomverse = "geomverse" in source
    
    gt = record.get("gt")
    if not isinstance(gt, str) or not gt.strip():
        return None, "missing_ground_truth"

    tail = THINK_CLOSE_RE.split(gt)[-1].strip()
    if not tail:
        return None, "empty_ground_truth"

    tags = ANSWER_TAG_RE.findall(tail)
    if tags:
        tail = tags[-1].strip()

    if is_geomverse:
        answer, method = _extract_geomverse_answer(tail)
        if answer:
            return answer, method
        # Without the result sentence fall back to the shared rules below.

    boxed = extract_last_boxed(tail)
    if boxed is not None:
        answer = normalize_answer(boxed)
        return (answer, "gt_boxed") if answer else (None, "empty_ground_truth")

    final_answer = FINAL_ANSWER_RE.fullmatch(tail)
    if final_answer:
        answer = normalize_answer(final_answer.group(1))
        # PlotQA ground truths use prose punctuation: ``Final Answer:12.``.
        if re.fullmatch(r"[+-]?(?:[$€£¥]\s*)?\d[\d,]*(?:\.\d+)?%?\.", answer):
            answer = answer[:-1].rstrip()
        return (answer, "gt_final_answer") if answer else (None, "empty_ground_truth")

    answer, method = _extract_verbose_gt_answer(tail)
    return (answer, method) if answer else (None, "empty_ground_truth")


def find_tool_errors(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return diagnostics for every failed attempted tool call."""
    failures: list[dict[str, str]] = []
    for message in record.get("result") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, dict) or "tool_call" not in content:
            continue

        finish_reason = str(content.get("finish_reason") or "").strip()
        execution_error = str(content.get("code_execution_error") or "").strip()
        result = str(content.get("tool_call_result") or "").strip()
        failed = finish_reason.casefold() in TOOL_FAILURE_REASONS
        failed = failed or bool(execution_error)
        failed = failed or bool(TOOL_ERROR_TEXT_RE.match(result))
        if not failed:
            continue

        failures.append(
            {
                "tool_name": str(content.get("tool_name") or "unknown"),
                "finish_reason": finish_reason,
                "error": execution_error or result or finish_reason or "tool call failed",
            }
        )
    record_reason = str(record.get("finish_reason") or "").strip()
    if (
        not failures
        and record_reason.casefold() in TOOL_FAILURE_REASONS
    ):
        failures.append(
            {
                "tool_name": "unknown",
                "finish_reason": record_reason,
                "error": record_reason,
            }
        )
    return failures


def evaluate_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    """Return parsed answers and rejection reasons for one rollout."""
    strfied_source = str(source)
    prediction, prediction_method = extract_prediction(record)
    ground_truth, gt_method = extract_ground_truth(record, strfied_source)
    tool_errors = find_tool_errors(record)
    answers_equal = False
    comparison_method = "unavailable"
    if prediction is not None and ground_truth is not None:
        answers_equal, comparison_method = answers_match(prediction, ground_truth)

    reasons: list[str] = []
    if prediction is None:
        reasons.append(prediction_method)
    if ground_truth is None:
        reasons.append(gt_method)
    if prediction is not None and ground_truth is not None and not answers_equal:
        reasons.append("answer_mismatch")
    if tool_errors:
        reasons.append("tool_error")

    return {
        "keep": not reasons,
        "reasons": reasons,
        "parsed_prediction": prediction,
        "parsed_ground_truth": ground_truth,
        "prediction_parse_method": prediction_method,
        "ground_truth_parse_method": gt_method,
        "comparison_method": comparison_method,
        "tool_errors": tool_errors,
    }


def discover_result_files(input_dir: Path, output_dir: Path) -> list[Path]:
    """Find a direct or recursively nested set of ``results.jsonl`` files."""
    files = []
    output_is_nested = output_dir != input_dir and output_dir.is_relative_to(input_dir)
    for path in sorted(input_dir.rglob(RESULTS_FILENAME)):
        if output_is_nested and path.is_relative_to(output_dir):
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No {RESULTS_FILENAME} found under {input_dir}")
    return files


def _write_jsonl_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def filter_file(source: Path, destination: Path) -> dict[str, Any]:
    """Filter one JSONL file and write kept/rejected outputs."""
    destination.mkdir(parents=True, exist_ok=True)
    kept_path = destination / FILTERED_RESULTS_FILENAME
    rejected_path = destination / REJECTED_FILENAME
    reason_counts: Counter[str] = Counter()
    parse_counts: Counter[str] = Counter()
    comparison_counts: Counter[str] = Counter()
    total = kept = rejected = 0

    with (
        source.open("r", encoding="utf-8") as input_handle,
        kept_path.open("w", encoding="utf-8") as kept_handle,
        rejected_path.open("w", encoding="utf-8") as rejected_handle,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: record must be a JSON object")

            total += 1
            evaluation = evaluate_record(record, source)
            parse_counts[f"prediction:{evaluation['prediction_parse_method']}"] += 1
            parse_counts[f"ground_truth:{evaluation['ground_truth_parse_method']}"] += 1
            comparison_counts[evaluation["comparison_method"]] += 1

            if evaluation["keep"]:
                _write_jsonl_line(kept_handle, record)
                kept += 1
                continue

            rejected += 1
            reason_counts.update(evaluation["reasons"])
            _write_jsonl_line(
                rejected_handle,
                {
                    "filter": {key: value for key, value in evaluation.items() if key != "keep"},
                    "record": record,
                },
            )

    return {
        "input_file": str(source),
        "output_file": str(kept_path),
        "filtered_out_file": str(rejected_path),
        "total": total,
        "kept": kept,
        "filtered_out": rejected,
        "reason_counts": dict(sorted(reason_counts.items())),
        "parse_counts": dict(sorted(parse_counts.items())),
        "comparison_counts": dict(sorted(comparison_counts.items())),
    }


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Filter all result files below ``input_dir``."""
    input_dir = input_dir.resolve() # 절대 경로 변환
    output_dir = output_dir.resolve() # 절대 경로 변환
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if input_dir != output_dir and input_dir.is_relative_to(output_dir):
        raise ValueError("Output directory must not be an ancestor of the input directory.")

    sources = discover_result_files(input_dir, output_dir) # source 파일 (.../results.jsonl)이 절대경로로 주어짐 
    summaries = []
    for source in sources:
        relative_parent = source.parent.relative_to(input_dir)
        destination = output_dir / relative_parent
        summaries.append(filter_file(source, destination))

    total = sum(item["total"] for item in summaries)
    kept = sum(item["kept"] for item in summaries)
    filtered_out = sum(item["filtered_out"] for item in summaries)
    combined_reasons: Counter[str] = Counter()
    for item in summaries:
        combined_reasons.update(item["reason_counts"])

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": summaries,
        "total": total,
        "kept": kept,
        "filtered_out": filtered_out,
        "reason_counts": dict(sorted(combined_reasons.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep rollout records whose boxed answer matches the parsed ground "
            "truth and whose tool calls completed without errors. Text matching "
            "is case-sensitive; numeric-only answers compare by value."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.input_dir, args.output_dir)
    print(
        f"Filtered {summary['total']} records: kept {summary['kept']}, "
        f"removed {summary['filtered_out']}."
    )
    if summary["reason_counts"]:
        print("Removal reasons:")
        for reason, count in summary["reason_counts"].items():
            print(f"  {reason}: {count}")
    print(f"Summary: {Path(summary['output_dir']) / SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()