# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete frozen physics batch without intermediate selection
"""Qualify resources, run every fitting phase, and evaluate only at the end."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def state(phase, detail=None):
    value = {'phase': phase, 'detail': detail, 'updated_utc_seconds': time.time(),
             'intermediate_winner_selection': False}
    temp = HERE / 'campaign_state.partial.json'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(HERE / 'campaign_state.json')
    print(json.dumps({'campaign_phase': phase, 'detail': detail}), flush=True)


def wait_prepared():
    deadline = time.monotonic() + 12 * 3600
    state('waiting_for_all_prepared_inputs')
    while time.monotonic() < deadline:
        file = HERE / 'prepared/receipt.json'
        if file.is_file():
            try:
                data = json.loads(file.read_text())
            except json.JSONDecodeError:
                time.sleep(15)
                continue
            assert data['status'] == 'all_physics_inputs_prepared'
            return
        status = subprocess.run(['systemctl', '--user', 'show', 'fec-d27-prepare-all-v1.service',
                                 '--property=ActiveState', '--value'], capture_output=True, text=True, check=True)
        assert status.stdout.strip() == 'active', 'preparation stopped without success receipt'
        time.sleep(15)
    raise TimeoutError('preparation deadline; no automatic restart')


def wait_gpu_resources():
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        holders = subprocess.run(['fuser', '/dev/kfd'], capture_output=True, text=True)
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:')}
        if not holders.stdout.strip() and memory['MemAvailable'] >= int(6.5 * 1024**2):
            return
        state('waiting_for_idle_GPU_and_host_RAM', {'kfd_in_use': bool(holders.stdout.strip()),
                                                  'available_GiB': memory['MemAvailable'] / 1024**2})
        time.sleep(60)
    raise TimeoutError('resource deadline; no other workloads stopped')


def run(script, arguments=(), physical_devices=None):
    environment = dict(os.environ)
    if physical_devices is not None:
        wait_gpu_resources()
        environment['ROCR_VISIBLE_DEVICES'] = physical_devices
    for name, expected in json.loads((HERE / 'frozen_sources.json').read_text()).items():
        with Path(name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    subprocess.run([sys.executable, '-u', str(HERE / script), *arguments], env=environment, check=True)


def main():
    qualification = json.loads((HERE / 'qualification_v1/qualification.json').read_text())
    assert qualification['status'] == 'ROCm_transfer_export_optimizer_qualified'
    assert qualification['forward_CPU_GPU_double_max_abs_difference'] < 1e-10
    large = json.loads((HERE / 'large_qualification_v1/qualification.json').read_text())
    assert large['status'] == 'full_size_ROCm_resource_path_qualified'
    assert json.loads((HERE / 'input_semantics_qualification.json').read_text())['status'] == 'joint_pressure_and_elapsed_time_memory_qualified'
    wait_prepared()
    state('running_all_50_neural_fits_no_validation_processing')
    run('train_batch.py', ('--out', str(HERE / 'fits')), '1,2,3')
    state('running_five_pressure_factor_fits_no_comparisons')
    run('fit_pressure_factor.py')
    state('building_MAST_scalar_development_predictions_no_MAST_labels')
    run('mast_factor_predictions.py')
    state('running_all_geometry_readouts_after_neural_fits')
    run('refit_readouts.py', physical_devices='1')
    state('all_experiment_runs_terminal_begin_common_evaluation')
    run('evaluate_batch.py')
    state('complete_all_comparisons_processed_together')
    with (HERE / 'campaign_receipt.json').open('x') as stream:
        json.dump({'status': 'full_frozen_campaign_complete', 'comparisons': str(HERE / 'evaluation/summary.json'),
                   'public_upload_or_submission': False, 'combined_candidate_built': False}, stream, indent=2)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        state('failed_closed_requires_implementation_or_resource_review',
              {'type': type(error).__name__, 'error': str(error)})
        raise
