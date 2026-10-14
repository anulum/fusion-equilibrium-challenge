# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — resource-gated axis target attribution
"""Wait for released anatomy resources before the finite source attribution."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'axis_target_attribution'
PYTHON = '$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python'


def main():
    ROOT.mkdir(exist_ok=True)
    started = time.time()
    try:
        deadline = time.monotonic() + 6 * 3600
        while True:
            assert time.monotonic() < deadline, 'bounded resource wait expired'
            state = subprocess.run(['systemctl', '--user', 'show', 'fec-d37-timeout-continuation-v2.service',
                                    '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip()
            available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
            if state in ('inactive', 'failed') and available >= 6 * 1024**3:
                break
            (ROOT / 'state.json').write_text(json.dumps({'phase': 'waiting_for_D37_release_and6GiB_headroom', 'UTC_seconds': time.time()}))
            time.sleep(30)
        env = dict(os.environ, JAX_PLATFORMS='cpu', ROCR_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        (ROOT / 'state.json').write_text(json.dumps({'phase': 'native_axis_target_and_finished_map_attribution', 'UTC_seconds': time.time()}))
        with (ROOT / 'run.log').open('x') as log:
            result = subprocess.run([PYTHON, '-B', '-u', str(HERE / 'axis_target_attribution.py')], env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=2 * 3600)
        assert result.returncode == 0, result.returncode
        assert json.loads((ROOT / 'analysis.json').read_text())['status'] == 'all358_fixed_five_time_samples_axis_attribution_complete'
        with (ROOT / 'terminal.json').open('x') as stream:
            json.dump({'status': 'axis_target_attribution_complete', 'seconds': time.time() - started,
                       'diagnostic_only': True, 'no357_targets_read': True, 'publication': False}, stream, indent=2)
    except Exception as error:
        with (ROOT / 'controller_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time() - started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
