#!/bin/bash
# SCPN Reactor Systems — D45: head-target dose-response diagnostic on the 120 screening shots
set -u
X=$FEC_RUNS/ml350/runs/d39_finisher_20261011
cd $X/src_d45; P=$HOME/fec_mast_v10/venv/bin/python
$P d45_eval.py --variants d45_variants.json --workers 22 --stride 3 --offsets 0 || exit 11
$P d45_report.py --variants d45_variants.json --stride 3 --offsets 0 --seed 4501 --out $X/reports/d45_head_dose_response_120.json || exit 12
echo "D45 complete"
