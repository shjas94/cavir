#!/usr/bin/env bash
# Build the cold-start v2 SFT dataset from finished rollouts: every step after run.sh.
# Usage: ./make_coldstart_v2.sh DATASET [DATASET ...]
#
# Per dataset (DATASET must be the rollout directory name, e.g. docvqa):
#   1. filter_out  $ROLLOUT_ROOT/<ds>/results.jsonl         -> $ROLLOUT_ROOT/<ds>/filtered_results.jsonl
#   2. stage       copy of filtered_results.jsonl            -> $WORK_ROOT/<ds>/filtered_results.jsonl
#   3. reformat    ShareGPT, no class balancing              -> $WORK_ROOT/<ds>/reformatted_results.jsonl
#   4. build       <evaluating> conversion (model server)    -> $WORK_ROOT/<ds>/cavir_formatted.jsonl
#   5. filter      drop records failing the audit rules      -> $WORK_ROOT/<ds>/$FILTER_NAME.jsonl
# Then for all given datasets together:
#   6. merge       one shuffled file, registered in $SFT_DIR/dataset_info.json as $SFT_NAME; tool-output
#                  images are copied to $SFT_DIR/$SFT_NAME/images/ so training does not depend on this repo
#                                                            -> $SFT_DIR/$SFT_NAME/$SFT_NAME.jsonl
#   7. check       LlamaFactory SFT preprocessing (train_cluster env)
#                                                            -> $SFT_DIR/$SFT_NAME/${SFT_NAME}_check.log
#
# A step whose output already exists is skipped, so the command can be rerun after an
# interruption and never overwrites anything. To redo a step, use new names (FILTER_NAME,
# SFT_NAME) or a new WORK_ROOT.
#
# Env (defaults): ROLLOUT_ROOT (outputs), WORK_ROOT (outputs_cavir), FILTER_NAME (cavir_filtered_v2),
# SFT_NAME (cavir_coldstart_v2), LLAMAFACTORY_DIR (../sft/LlamaFactory), SFT_DIR ($LLAMAFACTORY_DIR/data),
# PROCESSOR (Qwen/Qwen3-VL-2B-Thinking, measures context length in reformat), TAU, SEED, CONCURRENCY,
# MAX_TOOL_CALLS (build), PYTHON_BIN (research env), TRAIN_PYTHON (train_cluster env; empty skips step 7),
# SKIP_MERGE (non-empty: stop after step 5, so datasets can be built one at a time and merged later).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [DATASET ...]" >&2
  exit 2
fi

ROLLOUT_ROOT="$(realpath -m "${ROLLOUT_ROOT:-outputs}")"
WORK_ROOT="$(realpath -m "${WORK_ROOT:-outputs_cavir}")"
FILTER_NAME="${FILTER_NAME:-cavir_filtered_v2}"
SFT_NAME="${SFT_NAME:-cavir_coldstart_v2}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$(realpath -m "${SCRIPT_DIR}/../sft/LlamaFactory")}"
SFT_DIR="$(realpath -m "${SFT_DIR:-${LLAMAFACTORY_DIR}/data}")"
PROCESSOR="${PROCESSOR:-Qwen/Qwen3-VL-2B-Thinking}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/research/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON-/root/miniconda3/envs/train_cluster/bin/python}"
export PYTHON_BIN

step() { echo "[make_coldstart_v2] $*"; }

WORK_DIRS=()
for DATASET in "$@"; do
  ROLLOUT="${ROLLOUT_ROOT}/${DATASET}"
  WORK="${WORK_ROOT}/${DATASET}"
  WORK_DIRS+=("$WORK")

  if [[ -f "${ROLLOUT}/filtered_results.jsonl" ]]; then
    step "${DATASET}: 1 filter_out skipped (filtered_results.jsonl exists)"
  else
    [[ -f "${ROLLOUT}/results.jsonl" ]] || { echo "missing ${ROLLOUT}/results.jsonl" >&2; exit 1; }
    step "${DATASET}: 1 filter_out"
    ./filter_out.sh "$ROLLOUT" "$ROLLOUT"
  fi

  mkdir -p "$WORK"
  if [[ -f "${WORK}/filtered_results.jsonl" ]]; then
    if ! cmp -s "${ROLLOUT}/filtered_results.jsonl" "${WORK}/filtered_results.jsonl"; then
      echo "${WORK}/filtered_results.jsonl differs from ${ROLLOUT}/filtered_results.jsonl; use a new WORK_ROOT" >&2
      exit 1
    fi
    step "${DATASET}: 2 stage skipped (already staged)"
  else
    step "${DATASET}: 2 stage"
    cp "${ROLLOUT}/filtered_results.jsonl" "${WORK}/filtered_results.jsonl"
  fi

  if [[ -f "${WORK}/reformatted_results.jsonl" ]]; then
    step "${DATASET}: 3 reformat skipped (reformatted_results.jsonl exists)"
  else
    step "${DATASET}: 3 reformat (no class balancing, processor ${PROCESSOR})"
    MODEL_NAME="$PROCESSOR" MAX_CLASS_RATIO=1000 ./reformat_dataset.sh "$WORK"
  fi

  if [[ -f "${WORK}/cavir_formatted.jsonl" ]]; then
    step "${DATASET}: 4 build skipped (cavir_formatted.jsonl exists)"
  else
    step "${DATASET}: 4 build"
    ./build_cavir_coldstart.sh "$WORK"
  fi

  if [[ -f "${WORK}/${FILTER_NAME}.jsonl" ]]; then
    step "${DATASET}: 5 filter skipped (${FILTER_NAME}.jsonl exists)"
  else
    step "${DATASET}: 5 filter"
    "$PYTHON_BIN" filter_cavir_coldstart.py --output-name "$FILTER_NAME" "$WORK"
  fi
done

if [[ -n "${SKIP_MERGE:-}" ]]; then
  step "SKIP_MERGE set: stopping after step 5; run again without it, listing every dataset, to merge"
  exit 0
fi

MERGED="${SFT_DIR}/${SFT_NAME}/${SFT_NAME}.jsonl"
if [[ -f "$MERGED" ]]; then
  step "6 merge skipped (${MERGED} exists)"
else
  step "6 merge -> ${MERGED}"
  "$PYTHON_BIN" merge_cavir_coldstart.py --input-name "$FILTER_NAME" --output "$MERGED" \
    --dataset-info "${SFT_DIR}/dataset_info.json" --image-dir "${SFT_DIR}/${SFT_NAME}/images" "${WORK_DIRS[@]}"
fi

CHECK_LOG="${SFT_DIR}/${SFT_NAME}/${SFT_NAME}_check.log"
if [[ -z "$TRAIN_PYTHON" ]]; then
  step "7 check skipped (TRAIN_PYTHON is empty)"
elif [[ -f "$CHECK_LOG" ]]; then
  step "7 check skipped (${CHECK_LOG} exists)"
else
  step "7 check with LlamaFactory (${TRAIN_PYTHON}, CPU)"
  SCRATCH="$(mktemp -d)"
  # The check only tokenizes, so it runs on CPU: the teacher server used by step 4 usually still
  # holds the GPU memory, and CUDA initialization would fail with out-of-memory.
  if (cd "$LLAMAFACTORY_DIR" && CUDA_VISIBLE_DEVICES="" "$TRAIN_PYTHON" "${SCRIPT_DIR}/check_sft_dataset.py" "$SCRATCH" \
        --dataset "$SFT_NAME" --dataset-dir "$SFT_DIR") > "${CHECK_LOG}.tmp" 2>&1; then
    mv "${CHECK_LOG}.tmp" "$CHECK_LOG"
    grep -E "^(source|length|failed checks)" "$CHECK_LOG"
  else
    mv "${CHECK_LOG}.tmp" "${CHECK_LOG%.log}_failed.log"
    echo "check failed; see ${CHECK_LOG%.log}_failed.log" >&2
    exit 1
  fi
  rm -rf "$SCRATCH"
fi
step "done: LlamaFactory dataset '${SFT_NAME}' in ${SFT_DIR}/dataset_info.json"
