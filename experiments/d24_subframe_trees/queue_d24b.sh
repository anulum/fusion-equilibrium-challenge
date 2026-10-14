#!/bin/bash
# SCPN Reactor Systems — push the five part-b D24 kernels as the part-a kernels finish (Kaggle allows 5 CPU sessions)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
for k in 0 1 2 3 4; do
  while timeout 60 kaggle kernels status anulum/fec-d24-subframe-trees-f${k}a 2>&1 | grep -q "RUNNING\|QUEUED"; do sleep 300; done
  st=$(timeout 60 kaggle kernels status anulum/fec-d24-subframe-trees-f${k}a 2>&1 | tail -1)
  case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) ;; *) sleep 300; continue;; esac
  out=$(timeout 180 kaggle kernels push -p "$HERE/kaggle_d24_f${k}b" 2>&1 | tail -1)
  echo "$(date -u +%FT%TZ) f${k}a: ${st##*status } -> f${k}b: $out"
done
echo "QUEUE_DONE $(date -u +%FT%TZ)"
