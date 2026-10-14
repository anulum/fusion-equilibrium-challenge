#!/usr/bin/env bash
# For folds 1-4: wait for readout receipt (local WS for fold 1, ML350 for 2-4), pull it, build OOF maps, extract features.
K=$FEC_WORK/first_agent/kfold
PY=$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python
export PYTHONPATH=$HOME/fusion-equilibrium-challenge/fec_adapter:$K OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FEC_DATA_ROOT=$FEC_DATA/data
for k in 1 2 3 4; do
  until [ -f $K/readout_fold$k/readout_receipt_fold$k.json ]; do
    if [ $k -ge 2 ]; then rsync -a user@ml350:$HOME/fec_kfold/readout_fold$k $K/ 2>/dev/null; fi
    sleep 60
  done
  echo "$(date -u +%FT%TZ) fold $k readout present"
  [ -f $K/oof_fold$k/receipt.json ] || nice -n 12 $PY $K/kfold_build.py --fold $k --fits $K/kaggle_out_fold$k/out --readouts $K/readout_fold$k --out $K/oof_fold$k --workers 5 > $K/build_fold$k.log 2>&1 || { echo "build fold $k FAILED"; exit 1; }
  echo "$(date -u +%FT%TZ) fold $k built"
  (cd $K && nice -n 12 $PY corrector_features.py --archive $K/oof_fold$k/diii_d_train.npz --stems-from $K/oof_fold$k/receipt.json --out $K/features_fold$k.npz --workers 5) || { echo "features fold $k FAILED"; exit 1; }
  echo "$(date -u +%FT%TZ) fold $k features done"
done
echo "$(date -u +%FT%TZ) ALL FOLDS READY"
