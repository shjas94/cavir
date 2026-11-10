#!/usr/bin/env bash
# Build balanced ShareGPT SFT files from filtered_results.jsonl files.
# Usage: ./reformat_dataset.sh INPUT_DIR [SEED]
#
# INPUT_DIR may be a single dataset directory (outputs/docvqa) or the parent of
# all of them (outputs). In the parent case every dataset directory found below
# it is converted and each reformatted_results.jsonl is written back into its
# own dataset directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT_DIR [SEED]" >&2
  exit 2
fi

INPUT_DIR="$1"
SEED="${2:-42}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/research/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Thinking}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-32000}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-1024}"
MAX_CLASS_RATIO="${MAX_CLASS_RATIO:-2.0}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/reformat_dataset.py" \
  --input-dir "$INPUT_DIR" \
  --seed "$SEED" \
  --model-name "$MODEL_NAME" \
  --max-context-length "$MAX_CONTEXT_LENGTH" \
  --image-max-pixels "$IMAGE_MAX_PIXELS" \
  --image-min-pixels "$IMAGE_MIN_PIXELS" \
  --max-class-ratio "$MAX_CLASS_RATIO"
