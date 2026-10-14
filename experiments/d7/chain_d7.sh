#!/bin/bash
# D7 chain (protocol_d7_train_on_calibration.json): Kaggle outputs -> ML350 bank readouts -> D7 maps per fold -> score.
C=$FEC_WORK/first_agent
D=$C/d7; R=$HOME/fusion-equilibrium-challenge; PY=$R/fec_adapter/.venv/bin/python; H=user@ml350
log() { echo "$(date -u +%FT%TZ) $*"; }
fold_kernel() { [ $1 -le 2 ] && echo fec-d7-a || echo fec-d7-b; }
for n in a b; do
  if [ ! -d $D/kaggle_out_$n/out ]; then
    until timeout 60 kaggle kernels status anulum/fec-d7-$n 2>&1 | grep -q -E "COMPLETE|ERROR|CANCEL"; do sleep 120; done
    timeout 60 kaggle kernels status anulum/fec-d7-$n 2>&1 | grep -q COMPLETE || { log "kernel $n not COMPLETE"; exit 1; }
    mkdir -p $D/kaggle_out_$n && (cd $D/kaggle_out_$n && timeout 1800 kaggle kernels output anulum/fec-d7-$n -p . > download.log 2>&1)
  fi
  log "kernel $n fetched"
done
mkdir -p $D/fits; for d in $D/kaggle_out_a/out/*/ $D/kaggle_out_b/out/*/; do ln -sfn ${d%/} $D/fits/$(basename $d); done
for k in 0 1 2 3 4; do
  ssh -o BatchMode=yes $H "mkdir -p ~/fec_d7/fits_fold$k"
  tar -C $D/fits -chf - discharge_h_labels02_s76_tcall_k$k discharge_h_seed8_tcall_k$k discharge_hx_seed1_tcall_k$k | ssh -o BatchMode=yes $H "tar -x -C ~/fec_d7/fits_fold$k"
  ssh -o BatchMode=yes $H "cd ~/fec_d7/ml350_members && for d in m_*; do n=\${d#m_}; systemd-run --user --unit fec-d7-readout-k$k-\$n -p Nice=10 -p WorkingDirectory=$HOME/fec_d7/ml350_members/\$d --setenv=PYTHONPATH=$HOME/fec_mast_v10/pkg/fec_adapter:$HOME/fec_d7/ml350_members/\$d --setenv=FEC_SCRATCH=$HOME/fec_kfold/scratch --setenv=FEC_TABLES=$HOME/fec_kfold/tables --setenv=FEC_TARGETS_ROOT=$HOME/fec_kfold/targets --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 /bin/bash -c \"$HOME/fec_mast_v10/venv/bin/python kfold_readout.py --fold $k --nets $HOME/fec_d7/fits_fold$k --out $HOME/fec_d7/readouts_fold$k > $HOME/fec_d7/readout_k${k}_\$n.log 2>&1\" > /dev/null 2>&1; done"
  log "fold $k readouts launched"
done
for k in 0 1 2 3 4; do
  until [ "$(ssh -o BatchMode=yes $H "systemctl --user list-units --type=service --no-pager | grep -c 'fec-d7-readout-k$k-.*running'")" = "0" ]; do sleep 60; done
  mkdir -p $D/readouts_fold$k && scp -q -r "$H:fec_d7/readouts_fold$k/*" $D/readouts_fold$k/ || { log "fold $k readouts fetch failed"; exit 1; }
  n=$(ls -d $D/readouts_fold$k/discharge_* | wc -l); [ "$n" = 3 ] || { log "fold $k has $n readouts, need 3"; exit 1; }
  (cd $D && env PYTHONPATH=$R/fec_adapter:$D OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY d7_build.py --fold $k --fits $D/fits --readouts $D/readouts_fold$k --out $D/oof_fold$k --workers 5 > $D/build_fold$k.log 2>&1) || { log "d7 build fold $k failed"; exit 1; }
  log "fold $k built: $(python3 -c "import json;print({n:c['accepted'] for n,c in json.load(open('$D/oof_fold$k/receipt.json'))['choices'].items()})")"
done
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d7_score.py 0 1 2 3 4 > $D/score_folds01234.log 2>&1) || { log "score failed"; exit 1; }
for k in 0 1 2 3 4; do (cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d7_score.py $k > $D/score_fold$k.log 2>&1); done
log DONE
python3 -c "
import json
r=json.load(open('$D/results/summary_folds_01234.json'));print({k:r[k] for k in ('shots','dS','ci95','P_positive','adoption_passes')});print(r['terms']);print({k:round(v['D7']-v['V0'],6) for k,v in r['scalars'].items()})
for k in range(5):
  s=json.load(open('$D/results/summary_folds_%d.json'%k));print(k,round(s['dS'],7),round(s['P_positive'],3))"
