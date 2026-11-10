#!/usr/bin/env bash
set -euo pipefail

# Two-node data-parallel run of a value_head GPU script: torchrun, one process per node, node0 = rank 0.
# Run the same command on each node, changing only NODE_RANK (node0 = 0, node1 = 1; node1 inside its container):
#   NODE_RANK=0 bash value_head/run_ddp.sh train_scorer.py --data ... --out ... [args]
# SCRIPT is train_scorer.py, evaluate_scorer.py or cache_features.py; the arguments are the same as a single run.
# Runs from sft/LlamaFactory as the single-node commands do. Rank 0 writes the outputs on node0.
# The two launches wait up to an hour for each other (rendezvous), so a rank-0-only CPU step may sit between GPU steps.
VH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${VH}/../sft/scripts/env_common_fsdp.sh"
cd "$LLAMAFACTORY_DIR"

: "${NODE_RANK:?set NODE_RANK (0 on node0, 1 on node1)}"
SCRIPT=$1; shift
PORT="${DDP_PORT:-29502}"  # not 29500 (sft training) or 29501 (selection-following check)
echo "[node${NODE_RANK}] master=$MASTER_ADDR:$PORT  $SCRIPT"

exec /root/miniconda3/envs/train_cluster/bin/torchrun \
  --nnodes "$NNODES" --nproc_per_node 1 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$PORT" \
  --rdzv-conf timeout=3600 \
  "${VH}/${SCRIPT}" "$@"
