#!/bin/bash
# SCPN Reactor Systems — D39 round 2 (amendment 6): select by the frozen rule, lock, score once on the 238
set -u
X=$FEC_RUNS/ml350/runs/d39_finisher_20261011
cd $X/src; P=$HOME/fec_mast_v10/venv/bin/python; mkdir -p $X/reports
while systemctl --user is-active --quiet fec-d39-round1b; do sleep 60; done
[ "$(systemctl --user show fec-d39-round1b -p Result --value)" = success ] || { echo "round 1 did not end in success"; exit 20; }
$P d39_report_v3.py --variants d39_variants_a2.json --stride 3 --offsets 0 --select d39_amendment_4.json --out $X/reports/screen120_all32_selection.json || exit 21
$P d39_lock.py; code=$?
[ $code = 3 ] && { echo "no group has a passer: D39 ends without a finalist"; exit 0; }
[ $code = 0 ] || exit 22
$P d39_eval_r2.py --variants d39_variants_r2.json --workers 22 --stride 3 --offsets 1 2 --only base combo || exit 23
$P d39_report_v3.py --variants d39_variants_r2.json --stride 3 --offsets 1 2 --seed 3902 --out $X/reports/confirm238_base_combo.json || exit 24
$P - <<'PY' || exit 25
import json
from pathlib import Path
v=json.loads(Path('d39_variants_r2.json').read_text()); v['variants']['base_r2']={}
Path('d39_variants_r2_screen.json').write_text(json.dumps(v,indent=1))
PY
$P d39_eval_r2.py --variants d39_variants_r2_screen.json --workers 22 --stride 3 --offsets 0 --only combo base_r2 || exit 26
echo "round 2 complete"
