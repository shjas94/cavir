#!/usr/bin/env bash
# Run the CaViR dataset construction host.
# Usage: ./run.sh [DATASET_NAME] [SAMPLE_COUNT] [OUTPUT_NAME]
# OUTPUT_NAME names the directory under outputs/ (default: DATASET_NAME).
# OUTPUT_DIR (env) overrides the whole path instead. The tool set is chosen from
# DATASET_NAME (see below); env: TOOLS, CONCURRENCY, PYTHON_BIN, IDS_FILE (one dataset id per
# line: roll out only those samples), ROLLOUTS (generations per sample, default 1). host.py refuses to start if the results file already exists.
set -euo pipefail

# ---- Resolve script directory so relative paths to ./server work ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Config (override via env or positional args) ----
DATASET_NAME="${1:-${DATASET_NAME:-docvqa}}"
CONCURRENCY="${CONCURRENCY:-48}"
SAMPLE="${2:-${SAMPLE:--1}}"

OUTPUT_NAME="${3:-${OUTPUT_NAME:-${DATASET_NAME}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${OUTPUT_NAME}}"
OUTPUT_JSON="${OUTPUT_JSON:-results.jsonl}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/research/bin/python}"

# Tool set: reformat_dataset.DATASET_TOOL_NAMES[DATASET_NAME], the same set that
# reformat and build expect for this dataset. Override with TOOLS="crop_zoomin draw_bbox ...".
if [[ -n "${TOOLS:-}" ]]; then
  read -r -a TOOLS <<< "$TOOLS"
else
  read -r -a TOOLS <<< "$("$PYTHON_BIN" -c \
    "import sys; from reformat_dataset import DATASET_TOOL_NAMES as T; print(' '.join(T[sys.argv[1]]))" "$DATASET_NAME")"
fi
if [[ ${#TOOLS[@]} -eq 0 ]]; then
  echo "[run.sh] no tool set for dataset '$DATASET_NAME'; set TOOLS or add it to DATASET_TOOL_NAMES" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
echo "[run.sh] dataset=$DATASET_NAME"
echo "[run.sh] out=$OUTPUT_DIR/$OUTPUT_JSON  concurrency=$CONCURRENCY  sample=$SAMPLE"
echo "[run.sh] tools=${TOOLS[*]}"
echo "[run.sh] ids_file=${IDS_FILE:-<all>}"

# ---- Load .env if present (for BASE_URL / API_KEY consumed by host.py) ----
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

exec "$PYTHON_BIN" -u host.py \
  --dataset-name    "$DATASET_NAME" \
  --concurrency     "$CONCURRENCY" \
  --tools           "${TOOLS[@]}" \
  --server-dir      "${SCRIPT_DIR}/server" \
  --output-json     "$OUTPUT_JSON" \
  --output-dir      "$OUTPUT_DIR" \
  --sample          "${SAMPLE:- -1}" \
  --rollouts        "${ROLLOUTS:-1}" \
  ${IDS_FILE:+--ids-file "$IDS_FILE"}