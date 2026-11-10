#!/usr/bin/env bash
# Filter CaViR rollout results by answer correctness and tool-call success.
# Usage: ./filter_out.sh INPUT_DIR OUTPUT_DIR
# INPUT_DIR and OUTPUT_DIR may be the same; filtered_results.jsonl is written
# beside each source results.jsonl without modifying the source file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/research/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/filter_out.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"