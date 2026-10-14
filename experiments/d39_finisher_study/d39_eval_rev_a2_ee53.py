# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D39 finisher study: finish the cached answers as each variant tells and score
"""Run the deployed finisher with one setting changed on every held-out shot and score it officially."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from d39_common import (JOINT_SHARE, PKG, TREE_WEIGHTS, V29, X, cases, load_cache, reference_of,
                        score_maps, serial, statistics)

STATE = SimpleNamespace()


def initialize(variant_file: Path) -> None:
    import predict_diii_d as pd
    import run_parallel_score as scorer
    spec = importlib.util.spec_from_file_location('builder', V29 / 'deploy_build_subframe.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args = pd._parser().parse_args(['--data-root', '/dev/null', '--network', 'none', '--basis', 'none',
                                    '--out', 'none', '--split', 'diii_d_train', *builder.FLAGS])
    _rows, total, means = statistics()
    STATE.__dict__.update(pd=pd, scorer=scorer, finish=pd._finish_from(args), means=means, total=total,
                          basis=pd.FluxBasis.load(PKG / 'models/flux_basis.npz').truncated(64),
                          full_basis=pd.FluxBasis.load(PKG / 'models/flux_basis.npz'),
                          variants=json.loads(variant_file.read_text())['variants'])


def finish_of(change: dict):
    change = {k: v for k, v in change.items() if k not in ('tree_scale', 'warp_order', 'oracle')}
    if 'scalar_shares' in change:
        change['scalar_shares'] = tuple((name, float(share)) for name, share in change['scalar_shares'])
    return replace(STATE.finish, **change)


class Resampler:
    """scipy.ndimage as an edit module sees it, with the order of whole-grid resampling replaced."""

    def __init__(self, real, order: int) -> None:
        self.real, self.order = real, order

    def __getattr__(self, name):
        return getattr(self.real, name)

    def map_coordinates(self, flux, coordinates, order=1, mode='nearest', **kwargs):
        if np.ndim(coordinates[0]) == 2:
            order = self.order
        return self.real.map_coordinates(flux, coordinates, order=order, mode=mode, **kwargs)


def predicted(case: dict, cache: SimpleNamespace, change: dict, truth_maps=None) -> dict:
    import sys
    from scipy import ndimage
    pd = STATE.pd
    modules = [pd.contour_edit, sys.modules['shape_edit_fixed_axis_volume']]
    assert all(module.ndimage is ndimage for module in modules)
    basis, coordinates = STATE.basis, cache.coordinates.astype(float)
    if 'oracle' in change:  # diagnostic only: the true map's own coordinates, never a deployable answer
        size = int(change['oracle'])
        basis = STATE.full_basis.truncated(size)
        coordinates = STATE.full_basis.truncated(128).encode(truth_maps.astype(float))[:, :size]
    names = cache.label_names.astype(str).tolist()
    labels = {name: cache.labels[:, i].astype(float).copy() for i, name in enumerate(names)}
    scale = float(change.get('tree_scale', 1.0))
    for i, name in enumerate(cache.tree_names.astype(str).tolist()):
        labels[name] += scale * TREE_WEIGHTS[name] * (cache.tree[:, i] - labels[name])
    shot = SimpleNamespace(n_frames=len(cache.coordinates), r_grid=cache.r_grid, z_grid=cache.z_grid,
                           source=str(cache.source))
    kept = pd.read_row, pd.aligned_shot, pd.layout_features, pd.predict_together
    pd.read_row = lambda _path: {'efit_times': cache.efit_times}
    pd.aligned_shot = lambda *_args, **_kwargs: shot
    pd.layout_features = lambda *_args: {'discharge': None}
    pd.predict_together = lambda *_args: (coordinates.copy(), {k: v.copy() for k, v in labels.items()})
    finish = finish_of(change)
    if 'oracle' in change:
        finish = replace(finish, kept_coordinates=int(change['oracle']))
    if 'warp_order' in change:
        for module in modules:
            module.ndimage = Resampler(ndimage, int(change['warp_order']))
    try:
        maps, q95, beta_n = pd.predict_shot(Path(case['path']), [SimpleNamespace(feature_layout='discharge')],
                                            basis, finish)
    finally:
        pd.read_row, pd.aligned_shot, pd.layout_features, pd.predict_together = kept
        for module in modules:
            module.ndimage = ndimage
    return {'psirz': maps,
            'q95': ((1 - JOINT_SHARE) * q95.astype(float) + JOINT_SHARE * cache.joint_q95).astype(np.float32),
            'betaN': ((1 - JOINT_SHARE) * beta_n.astype(float) + JOINT_SHARE * cache.joint_beta_n).astype(np.float32)}


def one(case: dict) -> int:
    stem = case['stem']
    todo = [name for name in STATE.names if not (X / 'variants' / name / (stem + '.json')).exists()]
    if not todo:
        return case['index']
    cache = load_cache(stem)
    targets = pq.read_table(case['path'], columns=['efit_psirz', 'efit_q95', 'efit_beta_n'], use_threads=False)
    truth = {'psi': np.asarray(targets['efit_psirz'][0].as_py(), dtype=np.float32),
             'q95': np.asarray(targets['efit_q95'][0].as_py(), dtype=float),
             'betaN': np.asarray(targets['efit_beta_n'][0].as_py(), dtype=float)}
    del targets
    reference = reference_of(stem)
    for name in todo:
        started = time.monotonic()
        values = predicted(case, cache, STATE.variants[name], truth['psi'])
        assert all(np.isfinite(v).all() and v.dtype == np.float32 for v in values.values())
        part, distances, derived = score_maps(STATE.scorer, STATE.means, truth, reference, values)
        out = X / 'variants' / name
        np.savez_compressed(out / (stem + '.partial.npz'), distance=distances, derived=derived,
                            flux_mse=np.mean((values['psirz'].astype(float) - truth['psi']) ** 2, axis=(1, 2)))
        (out / (stem + '.partial.npz')).replace(out / (stem + '.npz'))
        temporary = out / (stem + '.partial.json')
        temporary.write_text(json.dumps(serial({'stem': stem, 'index': case['index'], 'frames': len(truth['psi']),
                                                'part': part, 'seconds': time.monotonic() - started})))
        temporary.replace(out / (stem + '.json'))
    print(json.dumps({'shot': case['index'], 'frames': len(truth['psi']), 'variants': len(todo)}), flush=True)
    return case['index']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--offsets', type=int, nargs='*', default=[0])
    args = parser.parse_args()
    initialize(args.variants)
    STATE.names = args.only or list(STATE.variants)
    for name in STATE.names:
        (X / 'variants' / name).mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases()[:args.limit] if c['index'] % args.stride in args.offsets]
    todo = sorted(todo, key=lambda c: c['span'][0] - c['span'][1])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        done = list(pool.map(one, todo, chunksize=1))
    print(json.dumps({'evaluation_complete': len(done), 'variants': STATE.names}), flush=True)


if __name__ == '__main__':
    main()
