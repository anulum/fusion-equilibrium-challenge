#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71: qualification on three shots, then all 358, then the report
set -euo pipefail
py=$FEC_RUNS/d60_clean_install_20261014/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
stage=${1:-qualify}
if [ "$stage" = qualify ]; then
  "$py" -B -u run_physics.py --activation activation_v2.json --out qualification_v2 --workers 3 --qualify
  "$py" -B -u report_physics.py --measures qualification_v2 --out qualification_v2_report.json
else
  "$py" -B -u run_physics.py --activation activation_v2.json --out full358_v2 --workers 20
  "$py" -B -u report_physics.py --measures full358_v2 --out report_v2.json
fi
