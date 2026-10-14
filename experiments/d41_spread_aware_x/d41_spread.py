# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D41: spread of the 69 held-out members' X-point outputs per frame
"""Write, for every development shot, the standard deviation over the 69 members of the four X-point heads."""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from d39_common import X, cases, digest, load_cache

LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')
NETS = []


def one(case: dict) -> str:
    out = X / 'xspread' / (case['stem'] + '.npz')
    if out.exists():
        return case['stem']
    features = load_cache(case['stem']).features.astype(float)
    members = np.stack([np.column_stack([net.predict(features)[1][name] for name in LABELS]) for net in NETS])
    temporary = out.with_name(case['stem'] + '.partial.npz')
    np.savez_compressed(temporary, sd=members.std(axis=0), mean=members.mean(axis=0))
    temporary.replace(out)
    return case['stem']


def main() -> None:
    import predict_diii_d as pd
    tasks = json.loads((X / 'd38/tasks.json').read_text())['tasks']
    for task in tasks:
        if task['family'] != 'joint':
            path = X / 'd38/readouts' / task['name'] / 'network.npz'
            assert digest(path) == json.loads(path.with_name('fit_report.json').read_text())['network_sha256']
            NETS.append(pd.Network.load(path))
    assert len(NETS) == 69
    (X / 'xspread').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=20, mp_context=multiprocessing.get_context('fork')) as pool:
        done = list(pool.map(one, cases(), chunksize=1))
    print(json.dumps({'xspread_complete': len(done)}), flush=True)


if __name__ == '__main__':
    main()
