# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D39 finisher study: cache the held-out 69's raw answers per shot
"""Predict every development shot once with the 69 + 8 networks that never saw it, and keep the raw answers."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from d39_common import PKG, V29, X, cases, digest, remap

STATE = SimpleNamespace()


def initialize() -> None:
    import predict_diii_d as pd
    import xgboost as xgb
    tasks = json.loads((X / 'd38/tasks.json').read_text())['tasks']

    def load(root, name):
        out = X / 'd38' / root / name
        assert digest(out / 'network.npz') == json.loads((out / 'fit_report.json').read_text())['network_sha256']
        return pd.Network.load(out / 'network.npz')

    maps = [load('readouts', t['name']) for t in tasks if t['family'] != 'joint']
    joint = [load('fits', t['name']) for t in tasks if t['family'] == 'joint']
    assert len(maps) == 69 and len(joint) == 8
    spec = importlib.util.spec_from_file_location('subframe', V29 / 'subframe_features.py')
    subframe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subframe)
    trees = {}
    for name, item in json.loads((X / 'd32/manifest.json').read_text())['trees'].items():
        path = remap(item['path'])
        assert digest(path) == item['sha256']
        tree = xgb.Booster()
        tree.load_model(path)
        tree.set_param({'device': 'cpu', 'nthread': 1})
        with np.load(X / 'd32/tree_lookups' / (name + '.npz'), allow_pickle=False) as stored:
            trees[name] = tree, stored['leaf_values'].copy(), stored['base_score'].copy()
    STATE.__dict__.update(pd=pd, xgb=xgb, maps=maps, joint=joint, subframe=subframe, trees=trees)


def read_inputs(path: Path) -> dict:
    pd = STATE.pd
    columns = pd.input_columns(dict.fromkeys(pq.read_schema(path).names))
    table = pq.read_table(path, columns=columns, use_threads=False)
    row = {}
    for name in columns:
        value = table[name][0]
        flat = name.startswith('magnetics_') and str(table.schema.field(name).type) in (
            'list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
    return row


def one(case: dict) -> str:
    out = X / 'cache' / (case['stem'] + '.npz')
    if out.exists():
        return case['stem']
    started = time.monotonic()
    pd, path = STATE.pd, Path(case['path'])
    row = read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([pd.layout_features(row, shot, {'discharge'})['discharge'],
                               STATE.subframe.features_of(path).astype(float)], axis=1)
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    coordinates, labels = pd.predict_together(STATE.maps, {'discharge': features})
    matrix = STATE.xgb.DMatrix(np.asarray(features[:, :367], dtype=np.float32, order='C'), nthread=1)
    tree = {}
    for name, (booster, leaves, base) in STATE.trees.items():
        indices = booster.predict(matrix, pred_leaf=True, iteration_range=(0, leaves.shape[0])).astype(np.int32)
        tree[name] = np.cumsum(leaves[np.arange(leaves.shape[0])[None, :], indices], axis=1, dtype=np.float32)[:, -1] + base
    _, joint = pd.predict_together(STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    names = sorted(labels)
    temporary = out.with_name(case['stem'] + '.partial.npz')
    np.savez_compressed(
        temporary, features=features.astype(np.float32), coordinates=coordinates,
        label_names=np.array(names), labels=np.column_stack([labels[n] for n in names]),
        tree_names=np.array(sorted(tree)), tree=np.column_stack([tree[n] for n in sorted(tree)]).astype(float),
        joint_q95=joint['q95'], joint_beta_n=joint['beta_n'], r_grid=shot.r_grid, z_grid=shot.z_grid,
        source=np.array(shot.source), efit_times=np.asarray(row['efit_times'], dtype=float),
        times_ms=shot.times_ms, ip_A=shot.plasma_current)
    temporary.replace(out)
    print(json.dumps({'cached': case['index'], 'frames': int(shot.n_frames), 'seconds': time.monotonic() - started}), flush=True)
    return case['stem']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    initialize()
    todo = cases()[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        done = list(pool.map(one, todo, chunksize=1))
    print(json.dumps({'cache_complete': len(done)}), flush=True)


if __name__ == '__main__':
    main()
