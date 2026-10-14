# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite matched69 and static8 control preparation
"""Wait for real qualification, refit all77 declared controls, retain every outcome."""
import json
import time
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P = Path(__file__).resolve().parent


def state(phase: str) -> None:
    """Record operational state without reading development metrics."""
    value = {'phase': phase, 'UTC_seconds': time.time(), 'confirmation_excluded': True}
    temporary = P / 'baseline_state.partial.json'
    temporary.write_text(json.dumps(value) + '\n')
    temporary.replace(P / 'baseline_state.json')
    print(json.dumps(value), flush=True)


try:
    state('waiting_full_ROCm_baseline_qualification')
    deadline = time.monotonic() + 3600
    receipt = P / 'qualification/qualification.json'
    while not receipt.exists():
        assert time.monotonic() < deadline
        active = subprocess.run(['systemctl', '--user', 'is-active', 'fec-d38-baseline-qualification-v1.service'], capture_output=True, text=True)
        assert active.stdout.strip() == 'active', 'qualification stopped without successful receipt'
        time.sleep(30)
    assert json.loads(receipt.read_text())['status'] == 'matched69_two_cohort_excluded_ROCm_path_qualified'
    for name, expected in json.loads((P / 'baseline_preparation_sources.json').read_text()).items():
        with Path(name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    import train_batch as fit
    import jax
    fit.STATE.update(data=fit.initialize(), tasks=json.loads((P / 'baseline_tasks.json').read_text())['tasks'])
    fit.STATE['scaling'] = {tc: fit.scaling_for(fit.STATE['data'], tc) for tc in (False, True)}
    assert len(jax.devices()) == 3 and len(fit.STATE['tasks']) == 77
    state('all77_matched_control_fits_active')
    assignments = [(card, fit.STATE['tasks'][card::3], P / 'fits') for card in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [row for group in pool.map(fit.worker, assignments) for row in group]
    assert len(results) == 77
    fit.write_json(P / 'fits/receipt.json', {'status': 'all77_matched_two_cohort_excluded_controls_terminal',
        'tasks': results, 'successful': sum(row['status'] == 'fit_complete' for row in results),
        'held_out_metrics_computed': False, 'confirmation_shots': 357, 'development_shots': 358})
    state('all77_matched_controls_terminal')
    outcome = {'status': 'complete_baseline_preparation', 'successful': sum(row['status'] == 'fit_complete' for row in results), 'tasks': 77}
except Exception as error:
    outcome = {'status': 'failed_requires_repair', 'type': type(error).__name__, 'error': str(error)}
    state('baseline_preparation_failed')
finally:
    (P / 'baseline_terminal.json').write_text(json.dumps(outcome, indent=2) + '\n')
if outcome['status'] != 'complete_baseline_preparation' or outcome['successful'] != 77:
    raise SystemExit(1)
