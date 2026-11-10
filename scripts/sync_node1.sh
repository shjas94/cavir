#!/usr/bin/env bash
# Copy node0 paths to node1 and compare md5 of every copied file.
#
#   scripts/sync_node1.sh [--dry-run|--check] PATH...   # PATH relative to /workspace, e.g. cavir/value_head
#   SRC_ROOT=/root/.cache/huggingface DST_ROOT=<node1 home>/.cache/huggingface \
#       scripts/sync_node1.sh hub/models--cavir--<name>  # model weights for multi-node inference
#
# One way and additive: files deleted on node0 stay on node1. Symlinks (HF cache snapshots) are copied as links,
# and md5 is taken of the file they point to. --dry-run lists the files and stops; --check compares md5 without
# copying (run it before a sync to see which node1 files would be overwritten). Exits 1 on any md5 mismatch.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cluster.env"   # private network values, git-ignored
SRC_ROOT=${SRC_ROOT:-/workspace}
DST_ROOT=${DST_ROOT:-${NODE1_HOME:?set NODE1_HOME in cluster.env}/workspace}
NODE1=${NODE1:-${NODE1_SSH:?set NODE1_SSH in cluster.env}}
EXCLUDES=(__pycache__ cavir/sft/LlamaFactory/saves cavir/sft/LlamaFactory/logs cavir/sft/LlamaFactory/wandb)

mode=sync
if [[ ${1:-} == --dry-run || ${1:-} == --check ]]; then mode=${1#--}; shift; fi
(( $# )) || { echo "usage: $0 [--dry-run|--check] PATH..." >&2; exit 2; }

cd "$SRC_ROOT"
prune=()
for e in "${EXCLUDES[@]}"; do
  [[ $e == */* ]] && prune+=(-path "$e" -prune -o) || prune+=(-name "$e" -prune -o)
done
list=$(mktemp)
trap 'rm -f "$list" "$list.local" "$list.remote"' EXIT
find "$@" "${prune[@]}" \( -type f -o -type l \) -print0 | sort -z > "$list"
echo "$(tr -cd '\0' < "$list" | wc -c) files, $(du -cLh --files0-from="$list" | tail -1 | cut -f1)"
[[ $mode == dry-run ]] && { tr '\0' '\n' < "$list"; exit 0; }

[[ $mode == sync ]] && tar -cf - --null -T "$list" | ssh "$NODE1" "tar -C '$DST_ROOT' -xf - --no-same-owner"
xargs -0 md5sum < "$list" > "$list.local"
# Files missing on node1 show up as diff lines; md5sum's errors go to stderr.
ssh "$NODE1" "cd '$DST_ROOT' && xargs -0 md5sum" < "$list" > "$list.remote" || true
if diff "$list.local" "$list.remote"; then
  echo "md5 match: $(wc -l < "$list.local") files"
else
  echo "md5 MISMATCH (lines above: < node0, > node1)" >&2
  exit 1
fi
