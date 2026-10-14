#!/bin/bash
set -euo pipefail
cd $FEC_EXPERIMENTS/D25_member_ablation_2026-10-08T154444Z
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import json; p=json.load(open("parity_1thread_result.json")); assert all(v["array_equal"] for v in p["parity"].values())'
sha256sum -c build_protocol.sha256
python -c 'import json,hashlib; p=json.load(open("build_protocol.json")); assert hashlib.sha256(open("build_h69.py","rb").read()).hexdigest()==p["script_sha256"]'
ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 user@ml350 'systemd-run --user --pipe --wait --collect --unit=fec-second_agent-h69-build-1602 -p MemoryMax=48G -p MemorySwapMax=0 -p CPUQuota=1600% -p RuntimeMaxSec=21600 -p TasksMax=256 /usr/bin/env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $HOME/fec_mast_v10/venv/bin/python -' < build_h69.py > diii_d_public_test.partial.npz 2> build_remote.log
python validate_h69.py
