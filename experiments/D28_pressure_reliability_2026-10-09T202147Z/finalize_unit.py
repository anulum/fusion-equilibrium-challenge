# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — terminal record for external job termination
"""Record a unit timeout or kill when the controller could not finish."""

import json
import os
import time
from pathlib import Path

root = Path(__file__).resolve().parent
terminal = root / 'terminal_receipt.json'
if not terminal.exists():
    with terminal.open('x') as stream:
        json.dump({'status': 'D28_external_termination', 'unit_result': os.environ.get('SERVICE_RESULT'),
                   'exit_code': os.environ.get('EXIT_CODE'), 'exit_status': os.environ.get('EXIT_STATUS'),
                   'completed_utc_seconds': time.time(), 'checkpoint_location': str(root)}, stream, indent=2)
