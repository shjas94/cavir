#!/bin/bash
# Warn-only memory watch around one generation step.
# Runs CMD, records node0 used memory (MemTotal - MemAvailable, kB) every 2 s to LOGDIR/meminfo.log, and above
# the 120 GB limit (117,187,500 kB = 111.8 GiB; MEMWARN_KB overrides) appends a line to LOGDIR/memwarn.log.
# Nothing is stopped. The step's output goes to LOGDIR/NAME.log and its exit code to LOGDIR/progress.log.
#   bash value_head/mem_warn.sh LOGDIR NAME CMD...
set -u
LOGDIR=$1 NAME=$2; shift 2
LIMIT=${MEMWARN_KB:-117187500}
mkdir -p "$LOGDIR"
"$@" > "$LOGDIR/$NAME.log" 2>&1 & PID=$!
while kill -0 $PID 2>/dev/null; do
    used=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t-a}' /proc/meminfo)
    echo "$(date +%s) $NAME $used" >> "$LOGDIR/meminfo.log"
    if [ "$used" -gt "$LIMIT" ]; then
        echo "$(date '+%F %T') WARNING $NAME node0 used $used kB ($(awk -v u=$used 'BEGIN{printf "%.1f", u/1048576}') GiB) > $LIMIT kB (not stopped)" >> "$LOGDIR/memwarn.log"
    fi
    sleep 2
done
wait $PID; rc=$?
echo "$(date '+%F %T') $NAME exit $rc" >> "$LOGDIR/progress.log"
exit $rc
