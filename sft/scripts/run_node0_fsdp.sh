#!/usr/bin/env bash
set -euo pipefail

# Lives in cavir/sft/scripts; LlamaFactory resolves data/, saves/ and logs/ from its own root,
# so the launch runs from cavir/sft/LlamaFactory. Configs come from cavir/sft/configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_common_fsdp.sh"
: "${CONFIG:?set CONFIG to a training yaml under sft/configs (no default, so a forgotten CONFIG cannot overwrite another model)}"
cd "$LLAMAFACTORY_DIR"

echo "[node0] master=$MASTER_ADDR:$MASTER_PORT  config=$CONFIG"

mkdir -p logs
LOG="logs/node0_$(date +%Y%m%d_%H%M%S).log"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --machine_rank 0 \
  --num_machines "$NNODES" \
  --num_processes "$NNODES" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  src/llamafactory/launcher.py "$CONFIG" 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"