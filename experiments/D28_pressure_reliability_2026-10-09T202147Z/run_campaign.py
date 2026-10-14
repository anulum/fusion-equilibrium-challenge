# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite checkpointed pressure experiment
"""Finish the declared stages with one bounded checkpoint restart."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verified_sources() -> None:
    """Refuse any mutation of frozen scientific and implementation sources."""
    for file, expected in json.loads((HERE / 'frozen_sources.json').read_text()).items():
        with Path(file).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, file


def state(phase: str, detail: dict | None = None) -> None:
    """Expose operational progress separately from immutable evidence."""
    value = {'phase': phase, 'detail': detail, 'utc_seconds': time.time(),
             'intermediate_score_selection': False}
    temporary = HERE / 'state.partial.json'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(HERE / 'state.json')
    print(json.dumps(value), flush=True)


def checkpoints() -> int:
    """Count verified-stage receipt locations for bounded restart accounting."""
    return sum(1 for directory in ('inputs/shots', 'factors', 'predictions/shots')
               for _ in (HERE / directory).rglob('*.json'))


def main() -> None:
    """Complete input census, all fits and predictions, then common scoring."""
    assert json.loads((HERE / 'qualification.json').read_text())['status'] == 'real_input_pressure_support_qualified'
    deadline = time.monotonic() + 35 * 3600
    retries = 0
    for script in ('prepare_support.py', 'fit_factors.py', 'build_predictions.py', 'evaluate.py'):
        while True:
            verified_sources()
            state('running_' + script.removesuffix('.py'), {'checkpoints': checkpoints(), 'restarts_used': retries})
            before = checkpoints()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('whole D28 wall-clock budget exhausted')
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(HERE)
            result = subprocess.run([sys.executable, '-u', str(HERE / script)], env=environment,
                                    timeout=remaining, check=False)
            if result.returncode == 0:
                break
            after = checkpoints()
            # One retry may consume the same checkpoint; a second never can.
            if retries >= 1:
                raise RuntimeError(f'{script} failed after restart budget: {result.returncode}')
            retries += 1
            state('bounded_checkpoint_restart', {'script': script, 'before': before,
                                                  'after': after, 'exit_code': result.returncode})
    verified_sources()
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    assert summary['status'] == 'all_three_guarded_comparisons_complete'
    value = {'status': 'D28_complete', 'completed_utc_seconds': time.time(), 'restarts_used': retries,
             'summary': str(HERE / 'evaluation/summary.json'), 'all_frames_retained': True,
             'public_upload_or_submission': False}
    with (HERE / 'terminal_receipt.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    assert json.loads((HERE / 'terminal_receipt.json').read_text()) == value
    state('complete_all_three_comparisons_processed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = {'status': 'D28_failed_closed', 'type': type(error).__name__, 'error': str(error),
                 'completed_utc_seconds': time.time(), 'checkpoint_count': checkpoints()}
        if not (HERE / 'terminal_receipt.json').exists():
            with (HERE / 'terminal_receipt.json').open('x') as stream:
                json.dump(value, stream, indent=2)
        state('failed_closed', value)
        raise
