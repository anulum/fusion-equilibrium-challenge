#!/usr/bin/env bash
A=$FEC_WORK/first_agent/alldata
todo=(a b)
while [ ${#todo[@]} -gt 0 ]; do rest=()
  for p in "${todo[@]}"; do s=$(kaggle kernels status anulum/fec-alldata-$p 2>&1 | tail -1)
    if echo "$s" | grep -q COMPLETE; then mkdir -p $A/kaggle_out_$p && (cd $A/kaggle_out_$p && kaggle kernels output anulum/fec-alldata-$p -p . >/dev/null 2>&1); echo "$(date -u +%FT%TZ) $p downloaded: $(ls $A/kaggle_out_$p/out | grep -c _all) fits"
    elif echo "$s" | grep -q -E "ERROR|CANCEL"; then echo "$(date -u +%FT%TZ) $p $s"
    else rest+=($p); fi; done
  todo=("${rest[@]}"); [ ${#todo[@]} -gt 0 ] && sleep 180; done
echo "$(date -u +%FT%TZ) done"
