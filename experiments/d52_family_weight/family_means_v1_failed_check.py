# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D52: q95 / betaN family means on the 358 development shots
"""Per-frame mean answers of the hx (21), hc (48) and joint (8) fit-disjoint networks, from the cached features."""
import json

import numpy as np

import d39_common as c


def main():
    import predict_diii_d as pd
    tasks = json.loads((c.X / 'd38/tasks.json').read_text())['tasks']
    load = lambda root, t: pd.Network.load(c.X / 'd38' / root / t['name'] / 'network.npz')
    families = {f: [load('readouts', t) for t in tasks if t['family'] == f] for f in ('hx', 'hc')}
    families['joint'] = [load('fits', t) for t in tasks if t['family'] == 'joint']
    assert [len(families[f]) for f in ('hx', 'hc', 'joint')] == [21, 48, 8]
    z = np.load(c.X / 'reports/d49_qb_frames_358.npz')
    features = np.concatenate([c.load_cache(str(s)).features.astype(float) for s in z['stems']])
    out = {}
    for family, nets in families.items():
        x = np.pad(features, ((0, 0), (0, 183))) if family == 'joint' else features
        _, labels = pd.predict_together(nets, {'discharge': x})
        for key, name in (('q95', 'q95'), ('betaN', 'beta_n')):
            out[f'{key}_{family}'] = np.asarray(labels[name], dtype=float)
    for key in ('q95', 'betaN'):
        both = (21 * out[f'{key}_hx'] + 48 * out[f'{key}_hc']) / 69
        assert np.max(np.abs(both - z[f'{key}_map'])) < 1e-9 and np.max(np.abs(out[f'{key}_joint'] - z[f'{key}_joint'])) < 1e-9
        out[f'{key}_truth'] = z[f'{key}_truth']
    np.savez_compressed(c.X / 'reports/d52_family_means_358.npz', shot=z['shot'], stems=z['stems'], **out)
    print(json.dumps({'frames': int(len(features)), 'keys': sorted(out)}))


if __name__ == '__main__':
    main()
