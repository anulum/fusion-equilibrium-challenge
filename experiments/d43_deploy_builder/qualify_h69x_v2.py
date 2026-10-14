# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — qualification v2 of the H69 + X-specialist builder on three public-test shots
"""Gated checks, every build task in a process of its own (the D25 builder stacks its wrappers on a second call).

v1 failed with a RecursionError for that reason; its log is kept as qualify.log.
"""
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import build_h69x_v2 as h
from native_x_intervention import use_x_specialist


def fresh(function, *args):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        return pool.submit(function, *args).result()


def native_task(core):
    return h.b._predict_many(core)


def wrapper_task(core, specialists):
    pd = h.b._load_entrypoint(h.PKG)
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

    h.b._blended = capturing
    with use_x_specialist(pd, experts, .5, half_window=0):
        return h.b._predict_many(core)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    listing, d36_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    specialists = h.b.verified(listing, [])
    pd, nets, finish, files = h.plan(3)
    assert len(files) == 3
    core = h.core_task(files, nets, finish)
    native = fresh(native_task, core)
    zero = fresh(h.run, (core, specialists, 0.0))
    mine = fresh(h.run, (core, specialists, 0.5))
    wrapper = [fresh(wrapper_task, h.core_task([path], nets, finish), specialists)[0] for path in files]
    assert len(native) == len(zero) == len(mine) == len(wrapper) == len(files) == 3
    result = {'shots': [Path(f).stem for f in files], 'frames': [int(len(a[1])) for a in native],
              'specialists': 'fit-disjoint D40 static 500-507, mechanics only; deployment uses all-data refits',
              'builder_sha256': h.b.digest(h.__file__), 'pinned_builder_sha256': h.b.digest(h.b.__file__)}
    result['share0_equals_D25_path'] = [all(same(a[k], c[k]) for k in range(3)) for a, c in zip(native, zero)]
    result['scalars_unchanged_by_x_blend'] = [same(a[1], c[1]) and same(a[2], c[2]) for a, c in zip(native, mine)]
    result['label_blend_equals_wrapper_at_x_edit'] = [all(same(a[k], c[k]) for k in range(3)) for a, c in zip(mine, wrapper)]
    result['max_map_change_at_share_half'] = [float(np.max(np.abs(a[0] - c[0]))) for a, c in zip(native, mine)]
    with np.load(d36_path, allow_pickle=False) as d36:
        archived = [d36[f'shot_{i:04d}_psirz'] for i in range(3)]
    result['native_maps_equal_D36_archive'] = [same(a[0], m) for a, m in zip(native, archived)]
    result['max_abs_difference_native_vs_D36_archive'] = [float(np.max(np.abs(a[0].astype(float) - m.astype(float)))) for a, m in zip(native, archived)]
    gates = ('share0_equals_D25_path', 'scalars_unchanged_by_x_blend', 'label_blend_equals_wrapper_at_x_edit', 'native_maps_equal_D36_archive')
    result['pass'] = all(all(result[g]) for g in gates) and all(v > 0 for v in result['max_map_change_at_share_half'])
    with out.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result))
    assert result['pass'], 'qualification failed; the receipt is kept'


if __name__ == '__main__':
    main()
