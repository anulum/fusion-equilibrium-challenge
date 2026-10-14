# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite candidate qualification and sequential GPU entrantship
"""Wait for controls, qualify both implementations and complete every declared fit."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
PYTHON = '$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python'


def write(path: Path, data: dict) -> None:
    """Write a new terminal receipt without replacing previous evidence."""
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def verify() -> None:
    """Require the declared candidate sources to retain their launch hashes."""
    for name, expected in json.loads((HERE / 'candidate_sources.json').read_text()).items():
        with Path(name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name


def state(phase: str) -> None:
    """Expose operational progress without candidate performance comparisons."""
    (HERE / 'candidate_state.json').write_text(json.dumps({'phase': phase, 'UTC_seconds': time.time(), 'development_metrics_computed': False}))


def wait_for(path: Path, status: str, unit: str) -> None:
    """Wait at most six hours for a successful upstream job."""
    deadline = time.monotonic() + 6 * 3600
    while not path.exists():
        actual = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip()
        assert actual == 'active', (str(path), unit, actual)
        assert time.monotonic() < deadline, 'upstream readiness deadline'
        time.sleep(30)
    receipt = json.loads(path.read_text())
    assert receipt['status'] == status, receipt.get('status')


def run(script: str, out: str, qualification: bool) -> None:
    """Use one card for qualification and three only after upstream release."""
    verify()
    state(script + ('_qualification' if qualification else '_all_fits'))
    env = dict(os.environ)
    env['ROCR_VISIBLE_DEVICES'] = '1' if qualification else '1,2,3'
    argv = [PYTHON, '-B', '-u', str(HERE / script), '--out', str(HERE / out)]
    if qualification:
        argv.append('--qualify')
    log = HERE / (out + '_run.log')
    with log.open('x') as stream:
        result = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=18 * 3600)
    assert result.returncode == 0, (script, out, result.returncode)
    verify()


def main() -> None:
    """Finish all 48 candidates or preserve a clear non-admission failure."""
    started = time.time()
    try:
        verify()
        state('waiting_for_all77_controls_and_GPU_release')
        wait_for(HERE / 'fits/receipt.json', 'all_77_neural_tasks_terminal', 'fec-d38-matched-baseline-v1.service')
        assert json.loads((HERE / 'fits/receipt.json').read_text())['successful'] == 77
        deadline = time.monotonic() + 120
        while subprocess.run(['systemctl', '--user', 'show', 'fec-d38-matched-baseline-v1.service', '-p', 'ActiveState', '--value'], capture_output=True, text=True).stdout.strip() == 'active':
            assert time.monotonic() < deadline
            time.sleep(5)
        wait_for(HERE / 'supervision/receipt.json', 'all6326_eligible_native_geometry_and_ray_targets_complete', 'fec-d38-supervision-prepare-v1.service')
        wait_for(HERE / 'temporal/preparation_receipt.json', 'eligible_channel_statistics_and_all_development_windows_complete', 'fec-d38-temporal-prepare-v2.service')
        run('train_maps.py', 'map_qualification', True)
        run('train_temporal.py', 'temporal_qualification', True)
        run('train_maps.py', 'map_fits', False)
        run('train_temporal.py', 'temporal_fits', False)
        maps = json.loads((HERE / 'map_fits/receipt.json').read_text())
        temporal = json.loads((HERE / 'temporal_fits/receipt.json').read_text())
        write(HERE / 'candidate_terminal.json', {'status': 'all48_declared_candidate_fits_terminal',
              'map_successful': maps['successful'], 'temporal_successful': temporal['successful'],
              'seconds': time.time() - started, 'development_metrics_computed': False,
              'all125_successful': maps['successful'] == 32 and temporal['successful'] == 16})
        state('all_declared_candidate_fits_terminal_awaiting_geometry_readouts_and_evaluation')
    except Exception as error:
        write(HERE / 'candidate_failure.json', {'status': 'candidate_workflow_failed_not_admitted',
              'type': type(error).__name__, 'error': str(error), 'seconds': time.time() - started,
              'development_metrics_computed': False})
        raise


if __name__ == '__main__':
    main()
