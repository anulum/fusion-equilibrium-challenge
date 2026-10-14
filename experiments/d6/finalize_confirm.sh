#!/bin/bash
# D6-confirm finalisation: c-family retries (pre-committed order: retry, then amendment 2 gl 0.1), D4-spec maps,
# D6 maps, score folds 2-4 and 0-4.
C=$FEC_WORK/first_agent
D=$C/d6; S=$D/d4spec; R=$HOME/fusion-equilibrium-challenge; PY=$R/fec_adapter/.venv/bin/python
log() { echo "$(date -u +%FT%TZ) $*"; }
until timeout 60 kaggle kernels status anulum/fec-d6conf-retry 2>&1 | grep -q -E "COMPLETE|ERROR|CANCEL"; do sleep 120; done
timeout 60 kaggle kernels status anulum/fec-d6conf-retry 2>&1 | grep -q COMPLETE || { log "retry kernel not COMPLETE"; exit 1; }
mkdir -p $S/kaggle_out_retry && (cd $S/kaggle_out_retry && timeout 1800 kaggle kernels output anulum/fec-d6conf-retry -p . > download.log 2>&1)
for k in 2 3 4; do
  for d in $S/kaggle_out_retry/out/*_k${k}_retry $S/kaggle_out_retry/out/*_k${k}_gl01; do [ -d "$d" ] && ln -sfn "$d" $S/kaggle_out_fold$k/out/$(basename $d); done
  (cd $S && PYTHONPATH=$R/fec_adapter $PY kfold_ledger.py --fold $k --fits $S/kaggle_out_fold$k/out --work $DISK/_offroot_cache/fec_d6_work) > $S/ledger_fold${k}_after_retry.log 2>&1
done
python3 - <<PY || { log "ledger finalisation failed"; exit 1; }
import json
S='$S'; L=S+'/kfold_fit_ledger.json'; led=json.load(open(L))
for k in ('2','3','4'):
    for m,e in led[k].items():
        if e['status']=='dropped' or e['status']=='failed_retry_pending':
            gl=e['first_fit']+'_gl01'
            try: r=json.load(open(f"{S}/kaggle_out_fold{k}/out/{gl}/fit_report.json"))['calibration']['flux_r2']
            except FileNotFoundError: raise SystemExit(f'{k} {m}: no amendment-2 fit')
            e.update(amendment2_fit=gl, amendment2_calibration_flux_r2=r, amendment2_rule='original seed, gradient_limit 0.1, accept >= 0.999')
            if r>=0.999: e.update(status='ok_amendment2', fitted_name=gl)
            else: e['status']='dropped'
json.dump(led,open(L,'w'),indent=1)
for k in ('2','3','4'): print(k,{m:(e['status'],e.get('fitted_name')) for m,e in led[k].items() if e['status']!='ok'})
PY
log "ledger final"
for k in 2 3 4; do until grep -q "readouts fetched" $D/stage_fold$k.log; do grep -q "not COMPLETE\|manual\|need 12\|fetch failed" $D/stage_fold$k.log && { log "stage fold $k failed"; exit 1; }; sleep 60; done; done
for k in 2 3 4; do
  (cd $S && env PYTHONPATH=$R/fec_adapter:$S OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY kfold_build.py --fold $k --fits $S/kaggle_out_fold$k/out --readouts $S/readouts_fold$k --out $S/oof_fold$k --workers 4 > $S/build_fold$k.log 2>&1) || { log "d4spec build fold $k failed"; exit 1; }
  (cd $D && env PYTHONPATH=$R/fec_adapter:$D OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $PY d6_build.py --fold $k --specialist $S/oof_fold$k --out $D/oof_fold$k --workers 4 > $D/build_fold$k.log 2>&1) || { log "d6 build fold $k failed"; exit 1; }
  log "fold $k built"
done
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d6_score.py 2 3 4 > $D/score_folds234.log 2>&1) || { log "score 234 failed"; exit 1; }
(cd $D && PYTHONPATH=$R/fec_adapter nice -n 5 $PY d6_score.py 0 1 2 3 4 > $D/score_folds01234.log 2>&1) || { log "score 01234 failed"; exit 1; }
log DONE
python3 -c "
import json
for n in ('234','01234'):
  r=json.load(open('$D/results/summary_folds_'+n+'.json'));print(n,{k:r[k] for k in ('dS','ci95','P_positive','screen_passes','adoption_passes')});print(r['terms']);print(r['scalars'])"
