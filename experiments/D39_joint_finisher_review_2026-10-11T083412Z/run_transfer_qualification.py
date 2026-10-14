# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — resource-gated central transfer qualification
"""Wait for completed D38 development and headroom before real prediction checks."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
PYTHON = '$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python'


def verify():
    for source, expected in json.loads((HERE / 'transfer_qualification_sources.json').read_text()).items():
        with Path(source).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, source
    for manifest in ['evaluation_sources.json', 'controller_resource_sources_v3.json']:
        for source, expected in json.loads((D38 / manifest).read_text()).items():
            with Path(source).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, source


def main():
    started = time.time()
    try:
        verify()
        deadline = time.monotonic() + 18 * 3600
        while True:
            assert time.monotonic() < deadline, 'bounded wait expired'
            assert not (D38 / 'evaluation/failure_v3.json').exists(), 'D38 development failed'
            completed = (D38 / 'evaluation/terminal_receipt.json').exists()
            state = subprocess.run(['systemctl', '--user', 'show', 'fec-d38-development-v3.service',
                                    '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip()
            if completed and state == 'inactive':
                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                if available >= 6 * 1024**3:
                    break
            elif not completed:
                assert state == 'active', state
            (HERE / 'transfer_qualification_state.json').write_text(json.dumps({
                'phase': 'waiting_for_D38_complete_and_6GiB_host_headroom', 'UTC_seconds': time.time()}))
            time.sleep(30)
        verify()
        assert json.loads((D38 / 'evaluation/development_analysis.json').read_text())['status'] == 'all_declared_development_arms_analyzed'
        env = dict(os.environ, JAX_PLATFORMS='cpu', ROCR_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        (HERE / 'transfer_qualification_state.json').write_text(json.dumps({'phase': 'real_whole_shot_native_and_fork_qualification', 'UTC_seconds': time.time()}))
        with (HERE / 'transfer_qualification_run.log').open('x') as stream:
            result = subprocess.run([PYTHON, '-B', '-u', str(HERE / 'qualify_joint_prediction.py')],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=4 * 3600)
        assert result.returncode == 0, result.returncode
        verify()
        receipt = json.loads((HERE / 'central_transfer_qualification.json').read_text())
        outcome = {'status': receipt['status'], 'seconds': time.time() - started, 'candidate_admitted': False}
        with (HERE / 'transfer_qualification_terminal.json').open('x') as stream:
            json.dump(outcome, stream, indent=2)
    except Exception as error:
        with (HERE / 'transfer_qualification_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time() - started,
                       'candidate_admitted': False}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
