#!/bin/bash
# D6-confirm chain (protocol_d6_confirm_folds234.json): after the local ROCm specialist fits of folds 2-4 —
# amendment-1 ledger (GPU retries, one process at a time), ML350 bank readouts, D4-spec maps, D6 maps, score.
C=$FEC_WORK/first_agent
D=$C/d6; S=$D/d4spec; R=$HOME/fusion-equilibrium-challenge; PY=$R/fec_adapter/.venv/bin/python
H=user@ml350
log() { echo "$(date -u +%FT%TZ) $*"; }
until grep -q "ALL_D4SPEC_FOLDS_DONE\|fits failed rc=" $S/local_folds234.log; do sleep 120; done
grep -q ALL_D4SPEC_FOLDS_DONE $S/local_folds234.log || { log "local fits failed: see local_folds234.log"; exit 1; }
log "fits done"
for k in 2 3 4; do
  (cd $R && export FEC_GPUS=1,2,3,4 && source env.sh gpu && unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS && export XLA_FLAGS=--xla_gpu_enable_command_buffer= \
    && cd $S && nice -n 10 $PY kfold_ledger.py --fold $k --fits $S/out_fold$k --work $DISK/_offroot_cache/fec_d6_work --retry) > $S/ledger_fold$k.log 2>&1
  python3 -c "
import json,sys;l=json.load(open('$S/kfold_fit_ledger.json'))['$k']
bad={m:e['status'] for m,e in l.items() if e['status'] not in ('ok','ok_after_retry')}
print(json.dumps(bad)); sys.exit(1 if bad else 0)" >> $S/ledger_fold$k.log 2>&1 || { log "fold $k needs amendment 2 / manual: see ledger_fold$k.log"; exit 1; }
  log "fold $k ledger ok"
done
for k in 2 3 4; do
  ssh -o BatchMode=yes $H "mkdir -p ~/fec_d6/fits_fold$k"
  tar -C $S/out_fold$k -chf - $(cd $S/out_fold$k && ls -d discharge_h*_k$k) | ssh -o BatchMode=yes $H "tar -x -C ~/fec_d6/fits_fold$k"
  ssh -o BatchMode=yes $H "cd ~/fec_d6/ml350_members && for d in m_*; do n=\${d#m_}; systemd-run --user --unit fec-d6-readout-k$k-\$n -p Nice=10 -p WorkingDirectory=$HOME/fec_d6/ml350_members/\$d --setenv=PYTHONPATH=$HOME/fec_mast_v10/pkg/fec_adapter:$HOME/fec_d6/ml350_members/\$d --setenv=FEC_SCRATCH=$HOME/fec_kfold/scratch --setenv=FEC_TABLES=$HOME/fec_kfold/tables --setenv=FEC_TARGETS_ROOT=$HOME/fec_kfold/targets --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 /bin/bash -c \"$HOME/fec_mast_v10/venv/bin/python kfold_readout.py --fold $k --nets $HOME/fec_d6/fits_fold$k --out $HOME/fec_d6/readouts_fold$k > $HOME/fec_d6/readout_k${k}_\$n.log 2>&1\" > /dev/null 2>&1; done"
  log "fold $k readouts launched"
  until [ "$(ssh -o BatchMode=yes $H "systemctl --user list-units --type=service --no-pager | grep -c 'fec-d6-readout-k$k-.*running'")" = "0" ]; do sleep 60; done
  mkdir -p $S/readouts_fold$k
  scp -q -r "$H:fec_d6/readouts_fold$k/*" $S/readouts_fold$k/ || { log "fold $k readouts fetch failed"; exit 1; }
  n=$(ls -d $S/readouts_fold$k/discharge_* | wc -l); [ "$n" = 12 ] || { log "fold $k has $n readouts, need 12"; exit 1; }
  log "fold $k readouts fetched"
done
for k in 2 3 4; do
  (cd $S && env PYTHONPATH=$R/fec_adapter:$S OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY kfold_build.py --fold $k --fits $S/out_fold$k --readouts $S/readouts_fold$k --out $S/oof_fold$k --workers 5 > $S/build_fold$k.log 2>&1) || { log "d4spec build fold $k failed"; exit 1; }
  (cd $D && env PYTHONPATH=$R/fec_adapter:$D OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY d6_build.py --fold $k --specialist $S/oof_fold$k --out $D/oof_fold$k --workers 5 > $D/build_fold$k.log 2>&1) || { log "d6 build fold $k failed"; exit 1; }
  log "fold $k built"
done
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d6_score.py 2 3 4 > $D/score_folds234.log 2>&1) || { log "score 234 failed"; exit 1; }
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d6_score.py 0 1 2 3 4 > $D/score_folds01234.log 2>&1) || { log "score 01234 failed"; exit 1; }
log DONE
python3 -c "
import json
for n in ('234','01234'):
  r=json.load(open('$D/results/summary_folds_'+n+'.json'));print(n,{k:r[k] for k in ('dS','ci95','P_positive','screen_passes','adoption_passes')});print(r['terms'])"
