#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71b: analytic tests must pass, then all 358 shots and the report
set -euo pipefail
py=$FEC_RUNS/d60_clean_install_20261014/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
"$py" -B test_critical.py > tests_critical.log
"$py" -B test_physics_b.py > tests_physics_b.log
if [ "${1:-qualify}" = qualify ]; then
  "$py" -B -u run_d71b.py --activation activation_v5.json --out qualification_v5 --workers 3 --qualify
  "$py" -B test_receipts.py qualification_v5 > tests_receipts_v5.log
  "$py" -B test_runner_cache.py qualification_v5 activation_v5.json > tests_runner_cache_v5.log
else
  "$py" -B -u run_d71b.py --activation activation_v5.json --out full358_v5 --workers 20
  "$py" -B -u report_d71b.py --measures full358_v5 --out report_v5.json --activation activation_v5.json --cases $FEC_RUNS/d70_boundary_second_agent_20261015/activation_v1.json
fi
