# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — serial GPU optimizer qualification process controller
"""Release each GPU process before starting the next checkpoint verification."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    subprocess.run([sys.executable, '-B', '-u', str(HERE / 'qualify_X_optimizer_resume.py')], check=True)
    for mode in ('static', 'mean', 'ordered'):
        subprocess.run([sys.executable, '-B', '-u', str(HERE / 'verify_X_process_resume.py'), '--mode', mode], check=True)
    reports = {mode: json.loads((HERE / 'optimizer_resume_qualification' / mode / 'new_process_resume.json').read_text())
               for mode in ('static', 'mean', 'ordered')}
    with (HERE / 'optimizer_resume_qualification' / 'process_chain_receipt.json').open('x') as stream:
        json.dump({'status': 'all3_actual_new_process_optimizer_RNG_history_resume_bitexact',
                   'reports': reports, 'no_real_candidate_fits_or_true_labels': True,
                   'seconds': time.monotonic()-started}, stream, indent=2)


if __name__ == '__main__':
    main()
