# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — qualification of the H69 + X-specialist builder on three public-test shots
"""Three checks on three shots: share 0 equals the D25 path; the label blend equals the second agent's wrapper at the X edit; D36 parity."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import build_h69x as h
from native_x_intervention import use_x_specialist

Y = Path(__file__).resolve().parent.parent
listing = Path(sys.argv[1])
specialists = h.b.verified(listing, [])
pd, nets, finish, files = h.plan(3)
core = h.core_task(files, nets, finish)
native = h.b._predict_many(core)
zero = h.run((core, specialists, 0.0))
assert all(np.array_equal(a[k], c[k]) for a, c in zip(native, zero) for k in range(3)), 'share 0 differs from the D25 path'
mine = h.run((core, specialists, 0.5))
assert all(np.array_equal(a[1], c[1]) and np.array_equal(a[2], c[2]) for a, c in zip(native, mine)), 'scalars changed'
changed = [float(np.max(np.abs(a[0] - c[0]))) for a, c in zip(native, mine)]
networks = [pd.Network.load(p) for p in specialists]
captured = {}
native_blend = h.NATIVE_BLEND


def capturing(nets_, features):
    captured['features'] = features['discharge'].copy()
    return native_blend(nets_, features)


def experts(_targets):
    padded = np.concatenate([captured['features'], np.zeros((len(captured['features']), 32))], axis=1)
    answers = [n.predict(padded)[1] for n in networks]
    return {name: np.mean([a[name] for a in answers], axis=0) for name in h.LABELS}


wrapper = []
for path in files:
    h.b._blended = capturing
    try:
        with use_x_specialist(pd, experts, .5, half_window=0):
            wrapper.extend(h.b._predict_many((core[0], [path], *core[2:])))
    finally:
        h.b._blended = native_blend
equal = [bool(np.array_equal(a[0], c[0])) for a, c in zip(mine, wrapper)]
with np.load(sys.argv[2], allow_pickle=False) as d36:
    parity = [float(np.max(np.abs(d36[f'shot_{i:04d}_psirz'].astype(float) - native[i][0].astype(float)))) for i in range(3)]
result = {'share0_equals_D25_path': True, 'scalars_unchanged_by_x_blend': True, 'max_map_change_at_share_half': changed,
          'label_blend_equals_wrapper_at_x_edit': equal, 'max_abs_map_difference_native_vs_D36_archive': parity,
          'specialists': 'fit-disjoint D40 static 500-507 (mechanics only; deployment uses all-data refits)',
          'frames': [int(len(a[1])) for a in native]}
print(json.dumps(result))
Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
