#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — local (ROCm) K-fold folds 3 and 4, protocol_kfold_oof_maps_20260929
K=$FEC_WORK/first_agent/kfold
cd $HOME/fusion-equilibrium-challenge || exit 9
source env.sh gpu
[ "${JAX_PLATFORMS:-}" = cpu ] && { echo "GPUs held, refusing to fit on CPU"; exit 8; }
echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES XLA_FLAGS=$XLA_FLAGS"
for k in 3 4; do
  nice -n 10 fec_adapter/.venv/bin/python $K/kfold_run.py --fold $k \
    --lcfs fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge \
    --work $DISK/_offroot_cache/fec_kfold_work \
    --out $K/out_fold$k || exit $?
done
echo ALL_LOCAL_FOLDS_DONE
