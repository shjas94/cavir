#!/bin/bash
# Warn-only watch of node1, run on node0 over ssh.
# Every 30 s it records node1 host used memory (MemTotal - MemAvailable, kB) and whether a vLLM container
# is up to LOG; above the 120 GB limit (117,187,500 kB = 111.8 GiB; MEMWARN_KB overrides) it appends a WARNING
# line to LOG.warn. Nothing is stopped; end the watch by its PID.
#   bash value_head/mem_warn_node1.sh LOG
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cluster.env"   # private network values, git-ignored
LOG=$1
LIMIT=${MEMWARN_KB:-117187500}
while true; do
    read -r used up < <(ssh -o ConnectTimeout=10 "$NODE1_SSH" \
        "awk '/MemTotal/{t=\$2}/MemAvailable/{a=\$2}END{printf \"%d \", t-a}' /proc/meminfo; docker ps -q -f name=vllm | wc -l" 2>/dev/null)
    echo "$(date '+%F %T') used_kB=${used:-NA} vllm_up=${up:-NA}" >> "$LOG"
    if [ -n "${used:-}" ] && [ "$used" -gt "$LIMIT" ]; then
        echo "$(date '+%F %T') WARNING node1 used $used kB ($(awk -v u=$used 'BEGIN{printf "%.1f", u/1048576}') GiB) > $LIMIT kB (not stopped)" >> "$LOG.warn"
    fi
    sleep 30
done
