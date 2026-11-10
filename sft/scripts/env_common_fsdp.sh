#!/usr/bin/env bash

# ── Distributed ──
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/cluster.env"   # private network values, git-ignored
export NNODES=2
export MASTER_ADDR=${NODE0_ADDR:?set NODE0_ADDR in cluster.env}
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=${NET_IFNAME:?set NET_IFNAME in cluster.env}
export TP_SOCKET_IFNAME=$NET_IFNAME
export NCCL_ALGO=Ring

# ── NCCL: RoCE (dual-rail) ──
export NCCL_SOCKET_IFNAME=$NET_IFNAME   # bootstrap/OOB 전용
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

# ── GB10 권장 ──
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1

# ── Reliability / diagnostics ──
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ── Paths ──
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAMAFACTORY_DIR="${SFT_DIR}/LlamaFactory"
export ACCELERATE_CONFIG="${SFT_DIR}/configs/fsdp2_config_qwen35_multiple_nodes.yaml"
