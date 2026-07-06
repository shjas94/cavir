#!/usr/bin/env bash
# Run the CaViR dataset construction host.
# Usage: ./run.sh [DATASET_NAME] [JSONL_PATH]
set -euo pipefail

# ---- Resolve script directory so relative paths to ./server work ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Config (override via env or positional args) ----
DATASET_NAME="${1:-${DATASET_NAME:-docvqa}}"
CONCURRENCY="${CONCURRENCY:-32}"
SAMPLE="${2:-${SAMPLE:--1}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${DATASET_NAME}_${TIMESTAMP}}"
OUTPUT_JSON="${OUTPUT_JSON:-results.jsonl}"

SERVERS=(
  "${SCRIPT_DIR}/server/code_execution_tool.py"
  "${SCRIPT_DIR}/server/image_processing_tool.py"
)

mkdir -p "$OUTPUT_DIR"
echo "[run.sh] dataset=$DATASET_NAME"
echo "[run.sh] out=$OUTPUT_DIR/$OUTPUT_JSON  concurrency=$CONCURRENCY  sample=$SAMPLE"

# ---- Load .env if present (for BASE_URL / API_KEY consumed by host.py) ----
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

exec python -u host.py \
  --dataset-name    "$DATASET_NAME" \
  --concurrency     "$CONCURRENCY" \
  --servers         "${SERVERS[@]}" \
  --output-json     "$OUTPUT_JSON" \
  --output-dir      "$OUTPUT_DIR" \
  --sample          "${SAMPLE:- -1}"