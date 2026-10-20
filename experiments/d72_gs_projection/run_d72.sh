#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D72: parity qualification on three shots, then all 358 and the gate
set -euo pipefail
py=$FEC_RUNS/d60_clean_install_20261014/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
if [ "${1:-qualify}" = qualify ]; then
  "$py" -B -u run_d72.py --activation activation_v1.json --out qualification_v1 --workers 3 --qualify
else
  "$py" -B -u run_d72.py --activation activation_v1.json --out full358_v1 --workers 20
  "$py" -B -u gate_d72.py full358_v1 gate_v1.json
fi
