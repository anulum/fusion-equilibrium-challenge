# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — qualified D26 gate successor
"""Run the frozen full comparison only after the end-to-end smoke succeeds."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    smoke = root / 'full_score_native_smoke_v1'
    deadline = time.monotonic() + 1200
    while not (smoke / 'comparison.json').is_file():
        status = subprocess.run(['systemctl', '--user', 'show', 'fec-d26-finish-native-smoke.service',
                                 '--property=ActiveState', '--value'], text=True, capture_output=True, check=True)
        assert status.stdout.strip() == 'active', 'smoke stopped before completion; full build refused'
        assert time.monotonic() < deadline, 'smoke qualification timeout'
        time.sleep(15)
    comparison = json.loads((smoke / 'comparison.json').read_text())
    manifest = json.loads((smoke / 'manifest.json').read_text())
    assert comparison['status'] == 'smoke_complete' and comparison['shots'] == 1
    assert manifest['smoke_only']
    script = root / 'finish_and_score.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == manifest['script_sha256']
    native = json.loads((root / 'native_reading_qualification.json').read_text())
    assert native['status'] == 'native_python_exact_sample_parity'
    assert all(x['scalars_equal'] and x['contour_equal'] for x in native['samples'])
    out = root / 'full_score_v2'
    print(json.dumps({'phase': 'qualified_full_successor', 'out': str(out)}), flush=True)
    subprocess.run([sys.executable, '-u', str(script), '--out', str(out), '--workers', '3'], check=True)
    subprocess.run([sys.executable, '-u', str(root / 'physical_error_diagnostics.py'), str(out)], check=True)
    with (root / 'finished_gate_successor_receipt.json').open('x') as stream:
        json.dump({'status': 'full_score_and_physical_diagnostics_completed', 'out': str(out),
                   'upload_or_submission': False}, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
