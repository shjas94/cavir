#!/usr/bin/env bash
# Build CaVIR cold-start v2 data (<evaluating> blocks) from filtered, reformatted rollouts.
# Usage: ./build_cavir_coldstart.sh DATASET [DATASET ...]
#
# DATASET is a directory under outputs/ (e.g. docvqa, gqa_1) or a path to one.
# Each must hold reformatted_results.jsonl (its answers are the ground truth); the result
# is written to <dataset>/cavir_formatted.jsonl, auxiliary files and the report
# to <dataset>/cavir_coldstart/.
#
# Env overrides: PILOT=N runs on N random rollouts and writes pilot files only;
# TAU (answer_now threshold, default 0.7), SEED, CONCURRENCY, MAX_TOOL_CALLS, PYTHON_BIN.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [DATASET ...]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/research/bin/python}"
TAU="${TAU:-0.7}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-3}"
PILOT="${PILOT:-0}"

# DIRECT_BASE_URL / DIRECT_API_KEY / MODEL_NAME for the teacher (Qwen3.8-Flash-Next).
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

for DATASET in "$@"; do
  if [[ -d "$DATASET" ]]; then
    DATASET_DIR="$(cd "$DATASET" && pwd)"
  else
    DATASET_DIR="${SCRIPT_DIR}/outputs/${DATASET}"
  fi
  if [[ ! -f "${DATASET_DIR}/reformatted_results.jsonl" ]]; then
    echo "[build_cavir_coldstart] missing ${DATASET_DIR}/reformatted_results.jsonl" >&2
    exit 1
  fi
  echo "[build_cavir_coldstart] dataset=${DATASET_DIR} tau=${TAU} pilot=${PILOT} concurrency=${CONCURRENCY}"
  "$PYTHON_BIN" -u build_cavir_coldstart.py \
    --dataset-dir    "$DATASET_DIR" \
    --tau            "$TAU" \
    --seed           "$SEED" \
    --concurrency    "$CONCURRENCY" \
    --max-tool-calls "$MAX_TOOL_CALLS" \
    --pilot          "$PILOT"
done
