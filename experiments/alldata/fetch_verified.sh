#!/usr/bin/env bash
# fetch_verified.sh KERNEL OUTDIR EXPECTED_FITS — wait for COMPLETE, download, verify every network.npz loads; retry download up to 3x
A=$FEC_WORK/first_agent/alldata
k=$1; out=$2; want=$3; PY=$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python
while true; do s=$(kaggle kernels status $k 2>&1 | tail -1)
  echo "$s" | grep -q COMPLETE && break
  echo "$s" | grep -q -E "ERROR|CANCEL" && { echo "$(date -u +%FT%TZ) $k $s"; exit 1; }
  sleep 120; done
for try in 1 2 3; do d=$out.try$try; mkdir -p $d && (cd $d && kaggle kernels output $k -p . >/dev/null 2>&1)
  n=$($PY -c "
import glob,numpy as np,sys
ok=0
for p in glob.glob('$d/out/*/network.npz'):
    try: z=np.load(p); [z[x] for x in z.files]; ok+=1
    except Exception: pass
print(ok)")
  echo "$(date -u +%FT%TZ) $k try$try: $n loadable of $want"
  if [ "$n" -eq "$want" ]; then ln -sfn $(basename $d) $out; echo "verified -> $out"; exit 0; fi
  sleep 60; done
exit 2
