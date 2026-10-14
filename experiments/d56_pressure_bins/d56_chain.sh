#!/bin/bash
# SCPN Reactor Systems — D56-0 gate chain on ML350
set -u
R=$FEC_RUNS/ml350/runs
D=$R/d56_pressure_bins_20261013; P=$HOME/fec_mast_v10/venv/bin/python
cd $D/src && sha256sum -c SHA256SUMS_d56 || exit 10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f $D/bins_training.npz ] || PYTHONPATH=$R/legacy_tree_fec_first_agent_20261005/deploy/pkg/source/fec_adapter $P pressure_bins.py $FEC_DATA/data/diii_d_train stems_training.json $R/d50_thermal_20261013/thermal_training.npz $D/bins_training.npz 12 || exit 11
export OMP_NUM_THREADS=12
PYTHONPATH=$R/d39_finisher_20261011/d32/vendor $P run_d50_gate.py $R/d50_thermal_20261013/nn_oof.npz $D/bins_training.npz $D/gate 12 || exit 12
$P d50_paired.py $R/d50_thermal_20261013/nn_oof.npz $D/gate $D/d56_gate_result.json || exit 13
echo "D56 complete"
