#!/bin/bash
cd $HOME/fusion-equilibrium-challenge && export FEC_GPUS=1 && source env.sh gpu
unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
exec $HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python -m fec_adapter.diii_d_accuracy.fit_on_accelerators --tables $FEC_WORK/first_agent/d19_subframe_recipe/work_local/tables_c --out-root $FEC_WORK/first_agent/d19_subframe_recipe/fits_local_gl --reserved $FEC_WORK/first_agent/d19_subframe_recipe/work_local/reserved_none.json --variants $FEC_WORK/first_agent/d19_subframe_recipe/work_local/variants_c_plain_gl01.json
