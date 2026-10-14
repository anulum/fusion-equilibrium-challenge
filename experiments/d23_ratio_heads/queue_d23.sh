#!/bin/bash
# SCPN Reactor Systems — push the five D23 kernels one by one as Kaggle CPU sessions free up (limit: 5 at a time)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WATCH="fec-v29-subframe-s2a fec-v29-subframe-s2b fec-v29-subframe-s2c fec-v29-subframe-s2d fec-v29-subframe-s2e"
for k in 0 1 2 3 4; do
  while :; do
    running=0
    for n in $WATCH; do timeout 60 kaggle kernels status anulum/$n 2>&1 | grep -q "RUNNING\|QUEUED" && running=$((running+1)); done
    [ $running -lt 5 ] && break
    sleep 300
  done
  out=$(timeout 180 kaggle kernels push -p "$HERE/kaggle_d23_f$k" 2>&1 | tail -1)
  echo "$(date -u +%FT%TZ) f$k running_before=$running: $out"
  case "$out" in *successfully*) WATCH="$WATCH fec-d23-ratio-heads-f$k"; sleep 120;; *) sleep 600; out=$(timeout 180 kaggle kernels push -p "$HERE/kaggle_d23_f$k" 2>&1 | tail -1); echo "$(date -u +%FT%TZ) f$k retry: $out"; WATCH="$WATCH fec-d23-ratio-heads-f$k";; esac
done
echo "QUEUE_DONE $(date -u +%FT%TZ)"
