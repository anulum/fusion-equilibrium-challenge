#!/bin/bash
# SCPN Reactor Systems — unattended end of D24-0: wait for the part-b kernels, fetch them, verify the frozen hashes, evaluate.
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for k in 0 1 2 3 4; do
  while timeout 60 kaggle kernels status anulum/fec-d24-subframe-trees-f${k}b 2>&1 | grep -q "RUNNING\|QUEUED"; do sleep 600; done
  mkdir -p kaggle_out_f${k}b
  timeout 900 kaggle kernels output anulum/fec-d24-subframe-trees-f${k}b -p kaggle_out_f${k}b 2>&1 | tail -1
  echo "$(date -u +%FT%TZ) f${k}b: $(timeout 60 kaggle kernels status anulum/fec-d24-subframe-trees-f${k}b 2>&1 | tail -1)"
done
echo "protocol files not matching their frozen hashes: $(sha256sum -c protocol_d24_0.sha256 2>&1 | grep -vc OK)"
$HOME/fusion-equilibrium-challenge/fec_harness/.venv/bin/python d24_eval.py > d24_eval.log 2>&1
sha256sum d24_result.json d24_eval.log kaggle_out_f*/gbm_sub_oof_fold*.npz > results_d24_0.sha256
echo "D24_FINISHED $(date -u +%FT%TZ)"; tail -n 16 d24_eval.log
