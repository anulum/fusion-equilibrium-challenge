# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — label-free sanity statistic of X-point specialists against the base heads
"""How far the specialists' mean X targets lie from the tree-adjusted base heads; no truth, no maps, no finisher.

``--cohort public``: all 874 public-test shots, base = the 69 production networks with the production tree blend.
``--cohort development120``: the 120 screening shots, base = the held-out 69 of D38 with the D9 trees (cached).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import build_h69x_v2 as h

LABELS = h.LABELS
STATE = {}


def experts(matrix):
    padded = np.concatenate([matrix, np.zeros((len(matrix), 32))], axis=1)
    answers = [network.predict(padded)[1] for network in STATE['networks']]
    return np.column_stack([np.mean([a[name] for a in answers], axis=0) for name in LABELS])


def public_task(task):
    core, specialists = task
    pd = h.b._load_entrypoint(h.PKG)
    STATE['networks'] = [pd.Network.load(p) for p in specialists]
    rows = []

    def heads_only(path, networks, basis, finish, gains=None):
        row = pd.read_row(path)
        shot = pd.aligned_shot(row, Path(path).stem, np.asarray(row['efit_times'], dtype=np.float64))
        features = pd.layout_features(row, shot, {network.feature_layout for network in networks})
        _coordinates, labels = pd.predict_together(networks, features)   # tree-adjusted by the builder's wrapper
        base = np.column_stack([labels[name] for name in LABELS])
        rows.append((Path(path).stem, base, experts(features['discharge'])))
        frames = len(base)
        return np.zeros((frames, 1, 1), np.float32), np.zeros(frames, np.float32), np.zeros(frames, np.float32)

    pd.predict_shot = heads_only
    h.b._predict_many(core)
    return rows


def development_task(task):
    stems, specialists = task
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pd = h.b._load_entrypoint(h.PKG)
    STATE['networks'] = [pd.Network.load(p) for p in specialists]
    X = h.RUNS / 'd39_finisher_20261011'
    weights = {'li_scored': .26, 'r_axis': .14, 'kappa': .09, 'x_lower_r': .16}
    rows = []
    for stem in stems:
        with np.load(X / 'cache' / (stem + '.npz'), allow_pickle=False) as cache:
            names = cache['label_names'].astype(str).tolist()
            labels = {name: cache['labels'][:, names.index(name)].astype(float) for name in LABELS}
            tree_names = cache['tree_names'].astype(str).tolist()
            for name in LABELS:
                if name in weights:
                    labels[name] += weights[name] * (cache['tree'][:, tree_names.index(name)] - labels[name])
            rows.append((stem, np.column_stack([labels[name] for name in LABELS]), experts(cache['features'].astype(float))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=('public', 'development120'), required=True)
    parser.add_argument('--x-list', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    specialists = h.b.verified(args.x_list, [])
    assert len(specialists) == 8
    if args.cohort == 'public':
        pd, nets, finish, files = h.plan(None)
        bounds = np.linspace(0, len(files), args.workers + 1).astype(int)
        tasks = [(h.core_task(files[x:y], nets, finish), specialists) for x, y in zip(bounds[:-1], bounds[1:])]
        function = public_task
    else:
        X = h.RUNS / 'd39_finisher_20261011'
        cases = json.loads((X / 'd32/manifest.json').read_text())['cases']
        stems = [c['stem'] for c in cases if c['index'] % 3 == 0]
        assert len(stems) == 120
        tasks = [(stems[i::args.workers], specialists) for i in range(args.workers)]
        function = development_task
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('fork')) as pool:
        rows = [row for part in pool.map(function, tasks) for row in part]
    base = np.concatenate([r[1] for r in rows]); expert = np.concatenate([r[2] for r in rows])
    difference = np.abs(expert - base)
    result = {'cohort': args.cohort, 'shots': len(rows), 'frames': int(len(base)), 'labels': list(LABELS),
              'specialists': specialists, 'x_list_sha256': h.b.digest(args.x_list),
              'base': 'tree-adjusted heads, before the X intervention', 'truth_or_maps_read': False,
              'experts_all_finite': bool(np.isfinite(expert).all()),
              'base_finite_counts': np.isfinite(base).sum(axis=0).astype(int).tolist(),
              'mean_abs_difference_m': np.nanmean(difference, axis=0).tolist(),
              'p99_abs_difference_m': np.nanquantile(difference, .99, axis=0).tolist(),
              'median_abs_difference_m': np.nanmedian(difference, axis=0).tolist(),
              'this_script_sha256': h.b.digest(__file__), 'builder_sha256': h.b.digest(h.__file__)}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'specialists'}))


if __name__ == '__main__':
    main()
