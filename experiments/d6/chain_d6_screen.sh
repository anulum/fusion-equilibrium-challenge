#!/bin/bash
cd $FEC_WORK/first_agent/d6 || exit 1
E="env PYTHONPATH=$HOME/fusion-equilibrium-challenge/fec_adapter:$FEC_WORK/first_agent/d6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n 5 $HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python"
$E d6_build.py --fold 0 --specialist $FEC_WORK/first_agent/d4/oof_fold0 --out $FEC_WORK/first_agent/d6/oof_fold0 --workers 5 > $FEC_WORK/first_agent/d6/build_fold0.log 2>&1 || { echo "build 0 failed"; exit 1; }
$E d6_build.py --fold 1 --specialist $FEC_WORK/first_agent/d4b/oof_fold1 --out $FEC_WORK/first_agent/d6/oof_fold1 --workers 5 > $FEC_WORK/first_agent/d6/build_fold1.log 2>&1 || { echo "build 1 failed"; exit 1; }
env PYTHONPATH=$HOME/fusion-equilibrium-challenge/fec_adapter nice -n 5 $HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python d6_score.py 0 1 > $FEC_WORK/first_agent/d6/score_folds01.log 2>&1 || { echo "score failed"; exit 1; }
echo DONE; python3 -c "import json;r=json.load(open('$FEC_WORK/first_agent/d6/results/summary_folds_01.json'));print({k:r[k] for k in ('dS','ci95','P_positive','screen_passes')});print(r['terms']);print(r['scalars'])"
