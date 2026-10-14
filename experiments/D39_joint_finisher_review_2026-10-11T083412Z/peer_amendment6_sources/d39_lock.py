# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D39: write the locked combination the frozen selector chose, once
"""Read the complete screening report and freeze its combination as the one round-2 variant."""
import hashlib
import json
import sys
from pathlib import Path

from d39_common import X

report_path = X / 'reports/screen120_all32_selection.json'
report = json.loads(report_path.read_text())
selection = report['selection']
assert report['shots'] == 120 and len(report['variants']) == 32
combination = selection['combination']
assert not ({'oracle', 'oracle_x'} & set(combination))
lock = {'name': 'combo', 'change': combination, 'winners': selection['winners'],
        'screening_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'source_manifest': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('.').glob('d39_*'))},
        'cache_sha256_of_all_358': hashlib.sha256('\n'.join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted((X / 'cache').glob('*.npz'))).encode()).hexdigest(),
        'no_winner': not combination}
with (X / 'src/d39_combination_locked.json').open('x') as stream:
    json.dump(lock, stream, indent=1)
with (X / 'src/d39_variants_r2.json').open('x') as stream:
    json.dump({'id': 'D39-round2-locked', 'variants': {'base': {}, 'combo': combination}}, stream, indent=1)
print(json.dumps(lock['winners']), json.dumps(combination))
sys.exit(3 if not combination else 0)
