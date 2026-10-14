# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite qualified DIII-D neural campaign
"""Run every declared training/readout stage before common comparison."""

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
    value = {'phase': phase, 'detail': detail, 'utc_seconds': time.time(),
             'primary_focus': 'DIII-D', 'intermediate_score_selection': False}
    temporary = HERE / 'state.partial.json'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(HERE / 'state.json')
    print(json.dumps(value), flush=True)


def verify_sources() -> None:
    for file, expected in json.loads((HERE / 'frozen_sources.json').read_text()).items():
        with Path(file).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, file


def wait_qualification() -> None:
    state('waiting_full_real_wide_ROCm_qualification')
    deadline = time.monotonic() + 2 * 3600
    while time.monotonic() < deadline:
        file = HERE / 'qualification/qualification.json'
        if file.exists():
            value = json.loads(file.read_text())
            assert value['status'] == 'full_real_ROCm_wide_learning_rate_path_qualified'
            assert value['architecture_width'] == 1024 and value['training_rows'] > 1_000_000
            assert value['CPU_GPU_double_max_abs'] < 1e-8
            assert value['checkpoint_restore_exact'] is True
            assert set(value['declared_schedules']) == {'lr1000', 'lr0500', 'lr0250'}
            return
        unit = subprocess.run(['systemctl', '--user', 'show', 'fec-d31-wide-qualify-v1',
                               '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True)
        assert unit.stdout.strip() == 'active', 'qualification stopped without success receipt'
        time.sleep(15)
    raise TimeoutError('wide resource qualification deadline')


def wait_resources() -> None:
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:')}
        holders = subprocess.run(['fuser', '/dev/kfd'], capture_output=True, text=True)
        if not holders.stdout.strip() and memory['MemAvailable'] >= int(6.5 * 1024**2):
            return
        state('waiting_idle_GPU_and_host_RAM', {'available_GiB': memory['MemAvailable'] / 1024**2,
                                               'kfd_held': bool(holders.stdout.strip())})
        time.sleep(60)
    raise TimeoutError('resources deadline; no other workloads stopped')


def run_stage(script: str, receipt: str, status: str, arguments: tuple = (), gpu: str | None = None) -> None:
    global RESTARTS
    file = HERE / receipt
    if file.exists():
        assert json.loads(file.read_text())['status'] == status
        return
    if gpu is not None:
        wait_resources()
    while True:
        verify_sources()
        state('running_' + script.removesuffix('.py'), {'global_restarts_used': RESTARTS})
        environment = dict(os.environ)
        if gpu is not None:
            environment['ROCR_VISIBLE_DEVICES'] = gpu
        result = subprocess.run([sys.executable, '-u', str(HERE / script), *arguments], env=environment,
                                timeout=48 * 3600, check=False)
        if result.returncode == 0:
            assert json.loads(file.read_text())['status'] == status
            return
        if RESTARTS >= 1:
            raise RuntimeError(f'{script} failed after one checkpoint restart; exit{result.returncode}')
        RESTARTS += 1
        state('bounded_operational_checkpoint_restart', {'script': script, 'exit_code': result.returncode})


def main() -> None:
    verify_sources()
    wait_qualification()
    run_stage('train_batch.py', 'fits/receipt.json', 'all_30_neural_tasks_terminal',
              ('--out', str(HERE / 'fits')), '1,2,3')
    run_stage('refit_readouts.py', 'readouts/receipt.json', 'all_readout_tasks_terminal', gpu='1')
    run_stage('evaluate_batch.py', 'evaluation/summary.json', 'all_three_learning_rate_comparisons_processed')
    run_stage('diagnose_training.py', 'training_diagnostic.json', 'all_30_training_trajectories_processed')
    verify_sources()
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    value = {'status': 'D31_workflow_complete', 'utc_seconds': time.time(), 'primary_focus': 'DIII-D',
             'successful_new_fits': neural['successful'], 'declared_new_fits': 30,
             'matched_controls_reused': 10, 'same_width_controls_reused': 10, 'restarts': RESTARTS,
             'summary': str(HERE / 'evaluation/summary.json'), 'public_upload_or_submission': False}
    with (HERE / 'terminal_receipt.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    assert json.loads((HERE / 'terminal_receipt.json').read_text()) == value
    state('complete_all_three_comparisons_processed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = {'status': 'D31_failed_closed', 'utc_seconds': time.time(), 'type': type(error).__name__,
                 'error': str(error), 'operational_restarts_used': RESTARTS}
        if not (HERE / 'terminal_receipt.json').exists():
            with (HERE / 'terminal_receipt.json').open('x') as stream:
                json.dump(value, stream, indent=2)
        state('failed_closed_requires_implementation_or_resource_review', value)
        raise
