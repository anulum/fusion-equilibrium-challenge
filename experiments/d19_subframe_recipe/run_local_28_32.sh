#!/bin/bash
cd $HOME/fusion-equilibrium-challenge && export FEC_GPUS=1 && source env.sh gpu
unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
export PYTHONPATH=$PYTHONPATH:$FEC_WORK/first_agent/kfold XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
PY=$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python
cd $FEC_WORK/first_agent/d19_subframe_recipe && exec $PY $FEC_WORK/first_agent/d19_subframe_recipe/alldata_run_local.py --python $PY --tables $FEC_WORK/first_agent/d19_subframe_recipe/tables_v29 --work $FEC_WORK/first_agent/d19_subframe_recipe/work_local1 --out $FEC_WORK/first_agent/d19_subframe_recipe/fits_local1 --members 28,29,30,31,32
