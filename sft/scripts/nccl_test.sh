# node 0 (on node1 use --node_rank 1)
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common_fsdp.sh
torchrun --nnodes 2 --node_rank 0 --nproc_per_node 1 \
  --master_addr "$MASTER_ADDR" --master_port 29500 \
  nccl_test.py