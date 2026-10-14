#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D6 confirmation: D4-recipe specialist fits, folds 2-4, local ROCm (protocol_d6_confirm_folds234.json)
cd $HOME/fusion-equilibrium-challenge || exit 9
export FEC_GPUS=1,2,3,4
source env.sh gpu
unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
[ "${JAX_PLATFORMS:-}" = cpu ] && { echo "GPUs held, refusing to fit on CPU"; exit 8; }
echo "$(date -u +%FT%TZ) ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES XLA_FLAGS=$XLA_FLAGS"
for k in 2 3 4; do
  echo "$(date -u +%FT%TZ) fold $k fits start"
  nice -n 10 fec_adapter/.venv/bin/python $FEC_WORK/first_agent/d6/d4spec/kfold_run.py --fold $k \
    --lcfs fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge \
    --work $DISK/_offroot_cache/fec_d6_work \
    --out $FEC_WORK/first_agent/d6/d4spec/out_fold$k || { echo "fold $k fits failed rc=$?"; exit 1; }
  echo "$(date -u +%FT%TZ) fold $k fits done"
done
echo ALL_D4SPEC_FOLDS_DONE
