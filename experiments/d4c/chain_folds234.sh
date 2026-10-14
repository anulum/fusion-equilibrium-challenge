#!/bin/bash
# D4c-confirm chain: folds 2,3,4 — fetch hc fits, ledger, ML350 readouts, WS build, score (protocol_d4c_confirm_folds234.json)
C=$FEC_WORK/first_agent
D=$C/d4c; K=$C/kfold; R=$HOME/fusion-equilibrium-challenge; PY=$R/fec_adapter/.venv/bin/python
H=user@ml350
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
for k in 2 3 4; do
  if [ ! -d $D/kaggle_out_fold$k/out ]; then
    until timeout 60 kaggle kernels status anulum/fec-d4c-lcfsw-fold$k 2>&1 | grep -q -E "COMPLETE|ERROR|CANCEL"; do sleep 120; done
    timeout 60 kaggle kernels status anulum/fec-d4c-lcfsw-fold$k 2>&1 | grep -q COMPLETE || { log "fold $k kernel not COMPLETE"; exit 1; }
    mkdir -p $D/kaggle_out_fold$k && (cd $D/kaggle_out_fold$k && timeout 600 kaggle kernels output anulum/fec-d4c-lcfsw-fold$k -p . > /dev/null 2>&1)
    mkdir -p $D/fits_fold$k $D/readouts_fold$k
    for d in $K/kaggle_out_fold$k/out/*/; do b=$(basename $d); case "$b" in *_hc_*) ;; *) ln -sfn ${d%/} $D/fits_fold$k/$b;; esac; done
    for d in $D/kaggle_out_fold$k/out/*/; do ln -sfn ${d%/} $D/fits_fold$k/$(basename $d); done
    for d in $K/readout_fold$k/*/; do b=$(basename $d); case "$b" in *_hc_*) ;; *) ln -sfn ${d%/} $D/readouts_fold$k/$b;; esac; done
    (cd $D && PYTHONPATH=$R/fec_adapter $PY kfold_ledger.py --fold $k --fits $D/fits_fold$k --work $DISK/_offroot_cache/fec_kfold_work > $D/ledger_fold$k.log 2>&1)
    grep -q "pending\|dropped" $D/ledger_fold$k.log && { log "fold $k has failed fits: see ledger_fold$k.log"; exit 1; }
    ssh -o BatchMode=yes $H "mkdir -p ~/fec_d4c/fits_fold$k"
    tar -C $D/kaggle_out_fold$k/out -chf - $(cd $D/kaggle_out_fold$k/out && ls -d *_k$k) | ssh -o BatchMode=yes $H "tar -x -C ~/fec_d4c/fits_fold$k"
    ssh -o BatchMode=yes $H "for i in 0 1 2 3 4; do systemd-run --user --unit fec-d4c-readout-k$k-m\$i -p Nice=10 -p WorkingDirectory=$HOME/fec_d4c/m\$i --setenv=PYTHONPATH=$HOME/fec_mast_v10/pkg/fec_adapter:$HOME/fec_d4c/m\$i --setenv=FEC_SCRATCH=$HOME/fec_kfold/scratch --setenv=FEC_TABLES=$HOME/fec_kfold/tables --setenv=FEC_TARGETS_ROOT=$HOME/fec_kfold/targets --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 /bin/bash -c \"$HOME/fec_mast_v10/venv/bin/python kfold_readout.py --fold $k --nets $HOME/fec_d4c/fits_fold$k --out $HOME/fec_d4c/readouts_fold${k}_m\$i > $HOME/fec_d4c/readout_k${k}_m\$i.log 2>&1\" > /dev/null 2>&1; done"
    log "fold $k readouts launched"
  fi
done
for k in 2 3 4; do
  until [ "$(ssh -o BatchMode=yes $H "systemctl --user list-units --type=service --no-pager | grep -c 'fec-d4c-readout-k$k.*running'")" = "0" ]; do sleep 60; done
  mkdir -p $D/readouts_hc_fold$k
  for i in 0 1 2 3 4; do scp -q -r $H:fec_d4c/readouts_fold${k}_m$i/discharge_hc_seed${i}_tc_lcfsw_k$k $D/readouts_hc_fold$k/ || { log "fold $k member $i readout missing"; exit 1; }; done
  for d in $D/readouts_hc_fold$k/*/; do ln -sfn ${d%/} $D/readouts_fold$k/$(basename $d); done
  log "fold $k readouts fetched ($(ls $D/readouts_fold$k | wc -l))"
  (cd $D && env PYTHONPATH=$R/fec_adapter:$D OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY kfold_build.py --fold $k --fits $D/fits_fold$k --readouts $D/readouts_fold$k --out $D/oof_fold$k --workers 5 > $D/build_fold$k.log 2>&1) || { log "build fold $k failed"; exit 1; }
  log "fold $k built"
done
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d4c_score.py 2 3 4 > $D/score_folds234.log 2>&1) || { log "score 234 failed"; exit 1; }
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d4c_score.py 0 1 2 3 4 > $D/score_folds01234.log 2>&1) || { log "score 01234 failed"; exit 1; }
log "DONE"
