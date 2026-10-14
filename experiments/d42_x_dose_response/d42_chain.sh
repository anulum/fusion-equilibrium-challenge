#!/bin/bash
# SCPN Reactor Systems — D42: dose-response diagnostic on the 120 screening shots
set -u
X=$FEC_RUNS/ml350/runs/d39_finisher_20261011
cd $X/src_d42; P=$HOME/fec_mast_v10/venv/bin/python
$P d42_eval.py --variants d42_variants.json --workers 22 --stride 3 --offsets 0 || exit 11
$P d42_report.py --variants d42_variants.json --stride 3 --offsets 0 --seed 4201 --out $X/reports/d42_dose_response_120.json || exit 12
echo "D42 complete"
