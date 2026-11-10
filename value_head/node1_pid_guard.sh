#!/bin/bash
# node1 memory guard for one two-node run, run on node0 over ssh.
# Every 30 s it records node1 host used memory (MemTotal - MemAvailable, kB) to LOG; if it passes LIMIT
# (110 GiB = 115,343,360 kB by default, below the 120 GB limit) it sends SIGTERM to the process tree of the given
# container PID (the node1 launch, found by parent PIDs; no name patterns) and stops.
# It also ends when that PID is gone.
#   bash value_head/node1_pid_guard.sh CONTAINER_PID LOG
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cluster.env"   # private network values, git-ignored
P=$1 LOG=$2
LIMIT=${GUARD_KB:-115343360}
on1() { ssh -o ConnectTimeout=10 "$NODE1_SSH" "$1" 2>/dev/null; }
while on1 "docker exec $NODE1_CONTAINER kill -0 $P" ; do
    used=$(on1 "awk '/MemTotal/{t=\$2}/MemAvailable/{a=\$2}END{print t-a}' /proc/meminfo")
    echo "$(date '+%F %T') used_kB=${used:-NA}" >> "$LOG"
    if [ -n "${used:-}" ] && [ "$used" -gt "$LIMIT" ]; then
        tree=$(on1 "docker exec $NODE1_CONTAINER bash -c 'p=$P; all=\$p; while [ -n \"\$p\" ]; do p=\$(ps -o pid= --ppid \"\$p\" | tr -d \" \" | tr \"\\n\" \" \"); all=\"\$all \$p\"; done; echo \$all'")
        on1 "docker exec $NODE1_CONTAINER kill $tree"
        echo "$(date '+%F %T') STOP node1 used $used kB > $LIMIT kB: SIGTERM to [$tree]" >> "$LOG.stop"
        break
    fi
    sleep 30
done
echo "$(date '+%F %T') guard end (pid $P gone or stopped)" >> "$LOG"
