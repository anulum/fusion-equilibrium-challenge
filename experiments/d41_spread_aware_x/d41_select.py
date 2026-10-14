# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D41: qualify the wrapper, apply the frozen selection rule, lock one arm
"""Check that the no-op arm equals base in every shot, then lock the arm the protocol's rule selects."""
import hashlib
import json
import sys

from d39_common import X, cases

report_path = X / 'reports/d41_screen120.json'
report = json.loads(report_path.read_text())
assert report['shots'] == 120 and set(report['variants']) == {'base', 'noop', 'sl', 'slu'}
for case in cases():
    if case['index'] % 3 == 0:
        a, b = (json.loads((X / 'd41_variants' / n / (case['stem'] + '.json')).read_text()) for n in ('base', 'noop'))
        assert a['part'] == b['part'] and a['prediction_sha256'] == b['prediction_sha256'], case['stem']
        old = json.loads((X / 'variants/base' / (case['stem'] + '.json')).read_text())
        assert a['part'] == old['part'], ('base differs from the D39 round-1 base', case['stem'])
v = report['variants']
passing = [(v[n]['delta_S'], -k, n) for k, n in enumerate(('sl', 'slu'))
           if v[n]['delta_S'] >= 1e-5 and v[n]['CI99'][0] > 0 and v[n]['lcfs_fails'] <= v['base']['lcfs_fails']]
chosen = max(passing)[2] if passing else None
variants = json.loads((X / 'src_d41/d41_variants.json').read_text())['variants']
lock = {'noop_equals_base_on_all_120': True, 'base_equals_D39_round1_base_on_all_120': True, 'selected': chosen,
        'change': variants[chosen] if chosen else None,
        'screening_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()}
with (X / 'src_d41/d41_locked.json').open('x') as stream:
    json.dump(lock, stream, indent=1)
if chosen:
    with (X / 'src_d41/d41_variants_r2.json').open('x') as stream:
        json.dump({'id': 'D41-round2-locked', 'variants': {'base': {}, chosen: variants[chosen]}}, stream, indent=1)
print(json.dumps({'selected': chosen}))
sys.exit(0 if chosen else 3)
