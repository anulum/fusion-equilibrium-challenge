#!/bin/bash
# SCPN Reactor Systems — D53-0 gate chain on ML350
set -u
R=$FEC_RUNS/ml350/runs
D=$R/d53_sibling_20261013; P=$HOME/fec_mast_v10/venv/bin/python
cd $D/src && sha256sum -c SHA256SUMS_d53 || exit 10
$P sibling_offsets.py $R/legacy_tree_fec_first_agent_20261005/d9/table/table_training.npz $R/d50_thermal_20261013/nn_oof.npz $D/out || exit 11
for k in 5 1 3 10 20; do $P d9_gate_eval.py --nn $R/d50_thermal_20261013/nn_oof.npz --gbm $D/out/k$k --out $D/d53_result_k$k.json > $D/eval_k$k.log || exit 12; done
echo "D53 complete"
