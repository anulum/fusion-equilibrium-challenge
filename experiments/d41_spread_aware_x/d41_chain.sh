#!/bin/bash
# SCPN Reactor Systems — D41: screen on the 120, qualify and select by the frozen rule, validate once on the 238
set -u
X=$FEC_RUNS/ml350/runs/d39_finisher_20261011
cd $X/src_d41; P=$HOME/fec_mast_v10/venv/bin/python; mkdir -p $X/reports
$P d41_eval.py --variants d41_variants.json --workers 22 --stride 3 --offsets 0 || exit 11
$P d41_report.py --variants d41_variants.json --stride 3 --offsets 0 --seed 4101 --out $X/reports/d41_screen120.json || exit 12
$P d41_select.py; code=$?
[ $code = 3 ] && { echo "D41: no arm passes the screening rule"; exit 0; }
[ $code = 0 ] || exit 13
$P d41_eval.py --variants d41_variants_r2.json --workers 22 --stride 3 --offsets 1 2 || exit 14
$P d41_report.py --variants d41_variants_r2.json --stride 3 --offsets 1 2 --seed 4102 --out $X/reports/d41_confirm238.json || exit 15
echo "D41 round 2 complete"
