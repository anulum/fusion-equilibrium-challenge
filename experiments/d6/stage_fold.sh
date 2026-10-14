#!/bin/bash
# D6-confirm, one fold: fetch Kaggle output, amendment-1 ledger (report only), ML350 bank readouts (bank members are
# independent of the c family), fetch readouts. Builds wait for the c-family retries.
k=$1
C=$FEC_WORK/first_agent
D=$C/d6; S=$D/d4spec; R=$HOME/fusion-equilibrium-challenge; PY=$R/fec_adapter/.venv/bin/python; H=user@ml350
log() { echo "$(date -u +%FT%TZ) $*"; }
if [ ! -d $S/kaggle_out_fold$k/out ]; then
  until timeout 60 kaggle kernels status anulum/fec-d6conf-fold$k 2>&1 | grep -q -E "COMPLETE|ERROR|CANCEL"; do sleep 180; done
  timeout 60 kaggle kernels status anulum/fec-d6conf-fold$k 2>&1 | grep -q COMPLETE || { log "fold $k kernel not COMPLETE"; exit 1; }
  mkdir -p $S/kaggle_out_fold$k && (cd $S/kaggle_out_fold$k && timeout 1800 kaggle kernels output anulum/fec-d6conf-fold$k -p . > download.log 2>&1)
fi
(cd $S && PYTHONPATH=$R/fec_adapter $PY kfold_ledger.py --fold $k --fits $S/kaggle_out_fold$k/out --work $DISK/_offroot_cache/fec_d6_work) > $S/ledger_fold$k.log 2>&1
log "fold $k ledger: $(python3 -c "import json;l=json.load(open('$S/kfold_fit_ledger.json'))['$k'];print({m:e['status'] for m,e in l.items() if e['status']!='ok'})")"
bank_bad=$(python3 -c "import json;l=json.load(open('$S/kfold_fit_ledger.json'))['$k'];print(sum(1 for m,e in l.items() if '_c_' not in m and e['status']!='ok'))")
[ "$bank_bad" = 0 ] || { log "fold $k: a bank member failed; manual"; exit 1; }
ssh -o BatchMode=yes $H "mkdir -p ~/fec_d6/fits_fold$k"
tar -C $S/kaggle_out_fold$k/out -chf - $(cd $S/kaggle_out_fold$k/out && ls -d discharge_h*_k$k) | ssh -o BatchMode=yes $H "tar -x -C ~/fec_d6/fits_fold$k"
ssh -o BatchMode=yes $H "cd ~/fec_d6/ml350_members && for d in m_*; do n=\${d#m_}; systemd-run --user --unit fec-d6-readout-k$k-\$n -p Nice=10 -p WorkingDirectory=$HOME/fec_d6/ml350_members/\$d --setenv=PYTHONPATH=$HOME/fec_mast_v10/pkg/fec_adapter:$HOME/fec_d6/ml350_members/\$d --setenv=FEC_SCRATCH=$HOME/fec_kfold/scratch --setenv=FEC_TABLES=$HOME/fec_kfold/tables --setenv=FEC_TARGETS_ROOT=$HOME/fec_kfold/targets --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 /bin/bash -c \"$HOME/fec_mast_v10/venv/bin/python kfold_readout.py --fold $k --nets $HOME/fec_d6/fits_fold$k --out $HOME/fec_d6/readouts_fold$k > $HOME/fec_d6/readout_k${k}_\$n.log 2>&1\" > /dev/null 2>&1; done"
log "fold $k readouts launched"
until [ "$(ssh -o BatchMode=yes $H "systemctl --user list-units --type=service --no-pager | grep -c 'fec-d6-readout-k$k-.*running'")" = "0" ]; do sleep 60; done
mkdir -p $S/readouts_fold$k
scp -q -r "$H:fec_d6/readouts_fold$k/*" $S/readouts_fold$k/ || { log "fold $k readouts fetch failed"; exit 1; }
n=$(ls -d $S/readouts_fold$k/discharge_* | wc -l); [ "$n" = 12 ] || { log "fold $k has $n readouts, need 12"; exit 1; }
log "fold $k readouts fetched"
