# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — resource-gated local X diagnostic
"""Run the finite four-arm diagnostic with immutable failures and source checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'X_target_causal_split'
PYTHON = '$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify():
    for source, expected in json.loads((HERE / 'X_target_causal_split_sources.json').read_text()).items():
        with Path(source).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, source


def run(extra, timeout):
    verify()
    env = dict(os.environ, JAX_PLATFORMS='cpu', ROCR_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    label = 'qualification' if '--qualify' in extra else 'all120'
    (ROOT / 'state.json').write_text(json.dumps({'phase': label, 'UTC_seconds': time.time()}))
    with (ROOT / (label + '_run.log')).open('x') as log:
        result = subprocess.run([PYTHON, '-B', '-u', str(HERE / 'X_target_causal_split.py'), *extra],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    assert result.returncode == 0, (label, result.returncode)
    verify()


def main():
    ROOT.mkdir(exist_ok=True)
    started = time.time()
    try:
        verify()
        deadline = time.monotonic() + 18 * 3600
        while True:
            assert time.monotonic() < deadline, 'bounded resource wait expired'
            assert not (HERE / 'transfer_qualification_failure.json').exists(), 'central transfer integrity failure requires review'
            state = subprocess.run(['systemctl', '--user', 'show', 'fec-d39-central-transfer-qualify-v1.service',
                                    '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip()
            available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
            if (HERE / 'transfer_qualification_terminal.json').exists() and state == 'inactive' and available >= 8 * 1024**3:
                break
            (ROOT / 'state.json').write_text(json.dumps({'phase': 'waiting_for_transfer_complete_and8GiB_headroom',
                                                      'UTC_seconds': time.time()}))
            time.sleep(30)
        run(('--qualify',), 4 * 3600)
        run(('--workers', '2'), 12 * 3600)
        outcome = json.loads((ROOT / 'analysis.json').read_text())
        assert outcome['status'] == 'all4x120_post_smoothing_X_causal_diagnostic_complete'
        write(ROOT / 'terminal.json', {'status': outcome['status'], 'seconds': time.time() - started,
              'diagnostic_only_nonselectable': True, 'no357_targets_read': True, 'publication': False})
    except Exception as error:
        write(ROOT / 'failure.json', {'type': type(error).__name__, 'error': str(error),
              'seconds': time.time() - started, 'candidate_admitted': False, 'publication': False})
        raise


if __name__ == '__main__':
    main()
