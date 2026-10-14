#!/usr/bin/env bash
# Download each Kaggle fold's output once its kernel COMPLETEs (folds 1-4).
K=$FEC_WORK/first_agent/kfold
todo=(1 2 3 4)
while [ ${#todo[@]} -gt 0 ]; do
  rest=()
  for k in "${todo[@]}"; do
    s=$(kaggle kernels status anulum/fec-kfold-fold$k 2>&1 | tail -1)
    if echo "$s" | grep -q COMPLETE; then
      mkdir -p $K/kaggle_out_fold$k && (cd $K/kaggle_out_fold$k && kaggle kernels output anulum/fec-kfold-fold$k -p . >/dev/null 2>&1)
      echo "$(date -u +%FT%TZ) fold $k downloaded: $(ls $K/kaggle_out_fold$k/out | grep -c _k$k) fits"
    elif echo "$s" | grep -q -E "ERROR|CANCEL"; then
      echo "$(date -u +%FT%TZ) fold $k status: $s"; rest+=($k)
    else rest+=($k); fi
  done
  todo=("${rest[@]}"); [ ${#todo[@]} -gt 0 ] && sleep 240
done
echo "$(date -u +%FT%TZ) all folds downloaded"
