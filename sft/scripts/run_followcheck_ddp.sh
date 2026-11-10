#!/usr/bin/env bash
set -euo pipefail

# Selection-following check over both nodes (torchrun, one process per node, data parallel).
# Run the same command on each node, changing only NODE_RANK (node0 = 0, node1 = 1):
#   NODE_RANK=0 OUT_DIR=<dir> bash run_followcheck_ddp.sh --model <model> --headers-from <train.jsonl> [more args]
# --prompts defaults to the cavir_pilot_v2 set. Rank 0 (node0) writes the combined output to OUT_DIR;
# node1's OUT_DIR holds only its own injections.rank1.jsonl.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_common_fsdp.sh"
cd "$LLAMAFACTORY_DIR"

: "${NODE_RANK:?set NODE_RANK (0 on node0, 1 on node1)}"
: "${OUT_DIR:?set OUT_DIR}"
PORT="${FOLLOWCHECK_PORT:-29501}"  # not the training port, so it can never join a training rendezvous
echo "[node${NODE_RANK}] master=$MASTER_ADDR:$PORT  out=$OUT_DIR"

mkdir -p logs
LOG="logs/followcheck_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

/root/miniconda3/envs/train_cluster/bin/torchrun \
  --nnodes "$NNODES" --nproc_per_node 1 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$PORT" \
  "${SCRIPT_DIR}/../selection_following.py" \
  --prompts "${LLAMAFACTORY_DIR}/data/cavir_pilot_v2/cavir_pilot_v2.jsonl" --out-dir "$OUT_DIR" "$@" 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
