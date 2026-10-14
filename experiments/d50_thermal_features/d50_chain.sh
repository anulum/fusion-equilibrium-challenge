#!/bin/bash
# SCPN Reactor Systems — D50-0 gate chain on ML350: extract, fit both arms, paired evaluation
set -u
X=$FEC_RUNS/ml350/runs
D=$X/d50_thermal_20261013; P=$HOME/fec_mast_v10/venv/bin/python
cd $D/src && sha256sum -c SHA256SUMS_d50 || exit 10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f $D/thermal_training.npz ] || PYTHONPATH=$X/legacy_tree_fec_first_agent_20261005/deploy/pkg/source/fec_adapter $P thermal_features.py $FEC_DATA/data/diii_d_train stems_training.json $D/thermal_training.npz 22 || exit 11
export OMP_NUM_THREADS=22
PYTHONPATH=$X/d39_finisher_20261011/d32/vendor $P run_d50_gate.py $D/nn_oof.npz $D/thermal_training.npz $D/gate 22 || exit 12
$P d50_paired.py $D/nn_oof.npz $D/gate $D/d50_gate_result.json || exit 13
echo "D50 complete"
