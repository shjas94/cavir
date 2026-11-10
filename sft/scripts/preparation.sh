#!/usr/bin/env bash
source "$(dirname "$0")/env_common_fsdp.sh"

echo "=== 인터페이스 ==="
ip -br addr show "$NCCL_SOCKET_IFNAME"

echo "=== 링크 속도 ==="
ethtool "$NCCL_SOCKET_IFNAME" 2>/dev/null | grep -i speed || echo "확인 불가"

echo "=== RDMA 매핑 ==="
ibdev2netdev 2>/dev/null || echo "ibdev2netdev 없음"

echo "=== master 도달 ==="
ping -c 2 -W 3 "$MASTER_ADDR" && echo OK || echo FAIL

echo "=== 포트 $MASTER_PORT ==="
ss -lntp 2>/dev/null | grep ":$MASTER_PORT" || echo "미사용 (정상)"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== LLaMA Factory ==="
llamafactory-cli version