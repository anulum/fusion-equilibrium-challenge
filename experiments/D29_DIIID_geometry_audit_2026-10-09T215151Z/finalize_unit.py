# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — terminal record after external job termination
"""Preserve terminal evidence if timeout or kill prevented normal completion."""

import json
import os
import time
from pathlib import Path

root = Path(__file__).resolve().parent
if not (root / 'terminal_receipt.json').exists():
    with (root / 'terminal_receipt.json').open('x') as stream:
        json.dump({'status': 'D29_external_termination', 'utc_seconds': time.time(),
                   'unit_result': os.environ.get('SERVICE_RESULT'),
                   'exit_code': os.environ.get('EXIT_CODE'), 'exit_status': os.environ.get('EXIT_STATUS'),
                   'checkpoints': str(root / 'shots')}, stream, indent=2)
