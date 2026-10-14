# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bounded DIII-D attribution campaign
"""Run the qualified diagnostic with one checkpoint restart and a deadline."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify_sources() -> None:
    """Check the original implementation and every frozen model before work."""
    for file, expected in json.loads((HERE / 'frozen_sources.json').read_text()).items():
        with Path(file).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, file


def main() -> None:
    """Save results and end without leaving a supervising loop."""
    assert json.loads((HERE / 'qualification.json').read_text())['status'] == 'real_baseline_all_stage_geometry_parity_qualified'
    deadline = time.monotonic() + 35 * 3600
    for attempt in range(2):
        verify_sources()
        temporary = HERE / 'state.partial.json'
        temporary.write_text(json.dumps({'phase': 'running_all_400_geometry_cases', 'attempt': attempt,
            'utc_seconds': time.time(), 'checkpoint_count': len(list((HERE / 'shots').glob('*.json'))),
            'primary_focus': 'DIII-D', 'intermediate_selection': False}, indent=2) + '\n')
        temporary.replace(HERE / 'state.json')
        result = subprocess.run([sys.executable, '-u', str(HERE / 'audit_geometry.py')],
                                timeout=max(1, deadline - time.monotonic()), check=False)
        if result.returncode == 0:
            break
        if attempt == 1:
            raise RuntimeError('D29 restart budget exhausted')
    verify_sources()
    summary = json.loads((HERE / 'summary.json').read_text())
    assert summary['status'] == 'all_DIII_D_geometry_stages_attributed' and summary['frames'] == 89584
    value = {'status': 'D29_complete', 'utc_seconds': time.time(), 'restart_count': attempt,
             'summary': str(HERE / 'summary.json'), 'primary_focus': 'DIII-D',
             'new_neural_fit_or_public_candidate': False}
    with (HERE / 'terminal_receipt.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    assert json.loads((HERE / 'terminal_receipt.json').read_text()) == value


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if not (HERE / 'terminal_receipt.json').exists():
            with (HERE / 'terminal_receipt.json').open('x') as stream:
                json.dump({'status': 'D29_failed_closed', 'error_type': type(error).__name__,
                           'error': str(error), 'utc_seconds': time.time()}, stream, indent=2)
        raise
