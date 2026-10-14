#!/bin/bash
# SCPN Reactor Systems — D57: after both arms, copy the fits to ML350 and run the frozen head report there
set -u
D=$FEC_EXPERIMENTS/D57_temporal_specialists_2026-10-13T200039Z; R=$FEC_RUNS/ml350/runs/d57_temporal_20261013
until ! systemctl --user is-active -q fec-d57-temporal && ! systemctl --user is-active -q fec-d57-control; do sleep 120; done
n=$(ls $D/fits/*/fit_report.json 2>/dev/null | wc -l)
echo "fits with a report: $n" > $D/after_fits.log
[ "$n" -eq 16 ] || { echo "not 16 fits; no report run" >> $D/after_fits.log; exit 11; }
python3 - >> $D/after_fits.log <<PY
import json,glob
r=[json.load(open(f)) for f in sorted(glob.glob('$D/fits/*/fit_report.json'))]
print({a:[x['seed_quality_pass'] for x in r if x['arm']==a] for a in ('temporal','control')})
print('all pass', all(x['seed_quality_pass'] for x in r))
raise SystemExit(0 if all(x['seed_quality_pass'] for x in r) else 12)
PY
[ $? -eq 0 ] || exit 12
ssh -o BatchMode=yes ml350 "mkdir -p $R/src" && rsync -aH $D/fits $D/SHA256SUMS $D/protocol_frozen_v1.json $D/temporal_model.py $D/train_temporal.py $D/head_report.py $D/run_arm.sh ml350:$R/src/ || exit 13
ssh -o BatchMode=yes ml350 "cd $R/src && sha256sum -c SHA256SUMS >/dev/null && systemd-run --user --wait --pipe -q --unit fec-d57-report -p MemoryMax=32G -p MemorySwapMax=0 -p WorkingDirectory=$R/src -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=8 -E OPENBLAS_NUM_THREADS=8 $HOME/fec_mast_v10/venv/bin/python head_report.py $R/src/fits $R/head_report_358.json" > $D/head_report_358.log 2>&1
rsync -aH ml350:$R/head_report_358.json $D/ >> $D/after_fits.log 2>&1
echo "D57 report done" >> $D/after_fits.log
