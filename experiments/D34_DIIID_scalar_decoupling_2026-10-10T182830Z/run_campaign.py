# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite scalar-task campaign
"""Run the qualified full batch before any scalar comparison; prohibit automatic publication."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESTARTS = 0


def state(phase: str, detail: dict | None = None) -> None:
    """Atomically retain an operational state without candidate selection."""
    temporary = HERE / 'state.partial.json'
    temporary.write_text(json.dumps({'phase': phase, 'detail': detail, 'utc_seconds': time.time(),
                                      'primary_focus': 'DIII-D', 'intermediate_score_selection': False}, indent=2))
    temporary.replace(HERE / 'state.json')
    print(json.dumps({'phase': phase, 'detail': detail}), flush=True)


def verify_sources() -> None:
    """Refuse any changed fit recipe, input lineage or control dependency."""
    for file, expected in json.loads((HERE / 'frozen_sources.json').read_text()).items():
        with Path(file).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, file


def wait_resources() -> None:
    """Wait finitely for unused GPUs and sufficient host RAM without stopping peers."""
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        available = next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:'))
        holders = subprocess.run(['fuser', '/dev/kfd'], capture_output=True, text=True)
        if not holders.stdout.strip() and available >= int(6.5 * 1024**2):
            return
        state('waiting_idle_GPU_and_RAM', {'available_GiB': available / 1024**2})
        time.sleep(60)
    raise TimeoutError('resource deadline')


def run_stage(script: str, receipt: str, expected: str, gpu: bool) -> None:
    """Allow one operational checkpoint resume without changing scientific settings."""
    global RESTARTS
    path = HERE / receipt
    if path.exists():
        assert json.loads(path.read_text())['status'] == expected
        return
    if gpu:
        wait_resources()
    while True:
        verify_sources()
        state('running_' + script.removesuffix('.py'), {'operational_restarts_used': RESTARTS})
        environment = dict(os.environ)
        if gpu:
            environment['ROCR_VISIBLE_DEVICES'] = '1,2,3'
        argv = [sys.executable, '-u', str(HERE / script)]
        if gpu:
            argv += ['--out', str(HERE / 'fits')]
        result = subprocess.run(argv, env=environment, timeout=48 * 3600, check=False)
        if result.returncode == 0:
            assert json.loads(path.read_text())['status'] == expected
            return
        if RESTARTS >= 1:
            raise RuntimeError(f'{script} failed after bounded checkpoint restart: {result.returncode}')
        RESTARTS += 1
        state('operational_checkpoint_restart', {'stage': script, 'exit': result.returncode})


def main() -> None:
    """Check qualifications, run all thirty fits, then evaluate all5936 shots."""
    verify_sources()
    qualification = json.loads((HERE / 'qualification/qualification.json').read_text())
    assert qualification['status'] == 'full_real_scalar_ROCm_path_qualified'
    assert qualification['training_rows'] > 1_000_000
    assert len(qualification['results']) == 3
    assert all(r['CPU_GPU_double_max_abs'] < 1e-8 and r['checkpoint_exact'] for r in qualification['results'])
    assert json.loads((HERE / 'baseline_qualification.json').read_text())['status'] == 'all5fold_both_scalar_controls_bitexact'
    run_stage('train_batch.py', 'fits/receipt.json', 'all_30_neural_tasks_terminal', True)
    run_stage('evaluate_batch.py', 'evaluation/summary.json', 'all_scalar_comparisons_complete', False)
    verify_sources()
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    receipt = {'status': 'D34_workflow_complete', 'utc_seconds': time.time(),
               'successful_new_fits': neural['successful'], 'declared_fits': 30,
               'shots': summary['shots'], 'frames': summary['frames'], 'operational_restarts': RESTARTS,
               'production_admission': False, 'full_H69_out_of_sample_validation_required': True,
               'public_upload_or_submission': False}
    with (HERE / 'terminal_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    state('complete_scalar_hypotheses_not_production_admission')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        receipt = {'status': 'D34_failed_closed', 'utc_seconds': time.time(),
                   'type': type(error).__name__, 'error': str(error), 'operational_restarts': RESTARTS}
        if not (HERE / 'terminal_receipt.json').exists():
            with (HERE / 'terminal_receipt.json').open('x') as stream:
                json.dump(receipt, stream, indent=2)
        state('failed_closed_requires_review', receipt)
        raise
