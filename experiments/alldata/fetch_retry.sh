#!/usr/bin/env bash
A=$FEC_WORK/first_agent/alldata
todo=(27 28)
while [ ${#todo[@]} -gt 0 ]; do rest=()
  for v in "${todo[@]}"; do s=$(kaggle kernels status anulum/fec-alldata-retry$v 2>&1 | tail -1)
    if echo "$s" | grep -q COMPLETE; then mkdir -p $A/kaggle_out_retry$v && (cd $A/kaggle_out_retry$v && kaggle kernels output anulum/fec-alldata-retry$v -p . >/dev/null 2>&1); echo "$(date -u +%FT%TZ) retry$v downloaded: $(ls $A/kaggle_out_retry$v/out | grep -c _retry) fits"
    elif echo "$s" | grep -q -E "ERROR|CANCEL"; then echo "$(date -u +%FT%TZ) retry$v $s"
    else rest+=($v); fi; done
  todo=("${rest[@]}"); [ ${#todo[@]} -gt 0 ] && sleep 120; done
echo "$(date -u +%FT%TZ) done"
