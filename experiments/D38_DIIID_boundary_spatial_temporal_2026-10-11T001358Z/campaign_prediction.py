# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete-69 prospective candidate prediction
"""Mix declared raw outputs before the unchanged native map finisher."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from campaign_common import HERE, PKG, V29, digest
from temporal_model import TemporalNetwork

D32 = HERE.parent / 'D32_DIIID_separate_cohort_confirmation_2026-10-10T144836Z'
MAP_FAMILIES = ('control', 'boundary_frame', 'boundary_shot', 'spatial')
ARMS = ('matched69_scalar8', 'map_control25', 'map_boundary_frame25', 'map_boundary_shot25', 'map_spatial25',
        'scalar_static_conv25', 'scalar_learned_conv25')
STATE = SimpleNamespace()
TREE_WEIGHTS = {'li_scored': .26, 'r_axis': .14, 'kappa': .09, 'x_lower_r': .16}


def load_network(root: str, name: str):
    """Require actual network receipts before including a model."""
    out = HERE / root / name
    receipt = json.loads((out / 'fit_report.json').read_text())
    assert digest(out / 'network.npz') == receipt['network_sha256']
    return STATE.pd.Network.load(out / 'network.npz')


def initialize() -> None:
    """Load only two-cohort-excluded neural models and the shared development trees."""
    assert json.loads((HERE / 'readouts/receipt.json').read_text())['successful'] == 101
    for path in (PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter', PKG / 'entrypoints', D32 / 'vendor'):
        sys.path.insert(0, str(path))
    import predict_diii_d as pd
    import xgboost as xgb
    original_read = pq.read_table

    def read(*args, **kwargs):
        kwargs['use_threads'] = False
        return original_read(*args, **kwargs)

    pq.read_table = read
    STATE.pd = pd
    tasks = json.loads((HERE / 'tasks.json').read_text())['tasks']
    maps = [load_network('readouts', t['name']) for t in tasks if t['family'] != 'joint']
    joint = [load_network('fits', t['name']) for t in tasks if t['family'] == 'joint']
    assert len(maps) == 69 and len(joint) == 8
    candidate = {family: [load_network('readouts', f'{family}_{seed}') for seed in range(400, 408)] for family in MAP_FAMILIES}
    temporal = {family: [TemporalNetwork(HERE / 'temporal_fits' / f'{family}_{seed}') for seed in range(400, 408)]
                for family in ('static_matched_conv_control', 'learned_conv_encoder')}
    decoder = []
    for seed in range(400, 408):
        path = HERE / 'map_fits' / f'spatial_{seed}' / 'decoder.npy'
        receipt = json.loads(path.with_name('fit_report.json').read_text())
        assert digest(path) == receipt['decoder_sha256']
        decoder.append(np.load(path, allow_pickle=False))
    spec = importlib.util.spec_from_file_location('campaign_subframe', V29 / 'subframe_features.py')
    subframe = importlib.util.module_from_spec(spec); spec.loader.exec_module(subframe)
    spec = importlib.util.spec_from_file_location('campaign_native_builder', V29 / 'deploy_build_subframe.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    args = pd._parser().parse_args(['--data-root', '/dev/null', '--network', tasks[0]['production_readout'],
        '--basis', str(PKG / 'models/flux_basis.npz'), '--out', str(HERE / 'evaluation'), '--split', 'diii_d_train', *builder.FLAGS])
    trees = {}
    manifest = json.loads((D32 / 'manifest.json').read_text())
    for name, item in manifest['trees'].items():
        assert digest(Path(item['path'])) == item['sha256']
        tree = xgb.Booster(); tree.load_model(item['path']); tree.set_param({'device': 'cpu', 'nthread': 1})
        with np.load(D32 / 'tree_lookups' / (name + '.npz'), allow_pickle=False) as stored:
            trees[name] = tree, stored['leaf_values'].copy(), stored['base_score'].copy()
    STATE.__dict__.update(maps=maps, joint=joint, candidate=candidate, temporal=temporal, decoder=decoder,
        trees=trees, xgb=xgb, subframe=subframe, finish=pd._finish_from(args),
        basis=pd.FluxBasis.load(PKG / 'models/flux_basis.npz').truncated(64),
        together=pd.predict_together, read_row=pd.read_row, layout_features=pd.layout_features,
        decode=pd.FluxBasis.decode, residual=None)

    def decode(basis, coordinates):
        result = STATE.decode(basis, coordinates)
        return result if STATE.residual is None else result + STATE.residual

    pd.FluxBasis.decode = decode


def read_inputs(path: Path) -> dict:
    """Keep long magnetic traces as bounded numeric arrays."""
    pd = STATE.pd
    columns = pd.input_columns(dict.fromkeys(pq.read_schema(path).names))
    table = pq.read_table(path, columns=columns, use_threads=False)
    row = {}
    for name in columns:
        value = table[name][0]
        flat = name.startswith('magnetics_') and str(table.schema.field(name).type) in ('list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
    return row


def trees_of(features: np.ndarray) -> dict:
    """Preserve the frozen original float32 tree accumulation order."""
    matrix = STATE.xgb.DMatrix(np.asarray(features[:, :367], dtype=np.float32, order='C'), nthread=1)
    result = {}
    for name, (booster, leaves, base) in STATE.trees.items():
        indices = booster.predict(matrix, pred_leaf=True, iteration_range=(0, leaves.shape[0])).astype(np.int32)
        values = leaves[np.arange(leaves.shape[0])[None, :], indices]
        result[name] = np.cumsum(values, axis=1, dtype=np.float32)[:, -1] + base
    return result


def finished(path: Path, row: dict, features: np.ndarray, coordinates: np.ndarray, labels: dict,
             trees: dict, residual: np.ndarray | None = None) -> tuple:
    """Use the native complete finisher with shared trees after prospective mixing."""
    pd = STATE.pd
    merged = {n: v.copy() for n, v in labels.items()}
    for name, weight in TREE_WEIGHTS.items():
        merged[name] += weight * (trees[name].astype(float) - merged[name])
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    pd.predict_together = lambda *_args: (coordinates.copy(), {k: v.copy() for k, v in merged.items()})
    STATE.residual = residual
    try:
        return pd.predict_shot(path, STATE.maps, STATE.basis, STATE.finish)
    finally:
        STATE.residual = None
        pd.read_row, pd.layout_features, pd.predict_together = STATE.read_row, STATE.layout_features, STATE.together


def predict(path: Path, qualify: bool = False) -> tuple:
    """Predict all predeclared whole-shot arms without selective frame omission."""
    pd = STATE.pd
    row = read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([STATE.layout_features(row, shot, {'discharge'})['discharge'], STATE.subframe.features_of(path).astype(float)], axis=1)
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    coordinates, labels = STATE.together(STATE.maps, {'discharge': features})
    trees = trees_of(features)
    base = finished(path, row, features, coordinates, labels, trees)
    if qualify:
        native_row = STATE.read_row(path)
        native_shot = pd.aligned_shot(native_row, path.stem, np.asarray(native_row['efit_times'], dtype=float))
        native_inputs = np.concatenate([STATE.layout_features(native_row, native_shot, {'discharge'})['discharge'], STATE.subframe.features_of(path).astype(float)], axis=1)
        assert np.array_equal(features, native_inputs)
        native_coordinates, native_labels = STATE.together(STATE.maps, {'discharge': native_inputs})
        for name, weight in TREE_WEIGHTS.items():
            native_labels[name] += weight * (trees[name].astype(float) - native_labels[name])
        pd.layout_features = lambda *_args: {'discharge': native_inputs}
        pd.predict_together = lambda *_args: (native_coordinates.copy(), {n: v.copy() for n, v in native_labels.items()})
        try:
            native = pd.predict_shot(path, STATE.maps, STATE.basis, STATE.finish)
        finally:
            pd.layout_features, pd.predict_together = STATE.layout_features, STATE.together
        assert all(np.array_equal(a, b) for a, b in zip(base, native, strict=True)), 'current native full69 parity'
    _, joint = STATE.together(STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    scalars = {n: (.75 * base[index].astype(float) + .25 * joint[label]).astype(np.float32)
               for n, index, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}
    result = {'matched69_scalar8': {'psirz': base[0], **scalars}}
    for family in MAP_FAMILIES:
        candidate_coordinates, candidate_labels = STATE.together(STATE.candidate[family], {'discharge': features})
        mixed_coordinates = .75 * coordinates + .25 * candidate_coordinates
        mixed_labels = {n: .75 * values + .25 * candidate_labels[n] for n, values in labels.items()}
        residual = None
        if family == 'spatial':
            fields = []
            for model, decoder in zip(STATE.candidate[family], STATE.decoder, strict=True):
                _, heads = model.predict(features)
                latent = np.column_stack([heads[f'spatial_{i:02d}'] for i in range(16)])
                fields.append((latent @ decoder).reshape(len(features), 65, 65))
            residual = .25 * np.mean(fields, axis=0)
            residual = pd.smooth_in_time(residual, STATE.finish.smooth_frames)
        values = finished(path, row, features, mixed_coordinates, mixed_labels, trees, residual)
        result['map_' + family + '25'] = {'psirz': values[0], **scalars}
    raw = np.load(HERE / 'temporal/development' / (path.stem + '.npy'), allow_pickle=False)
    assert raw.shape == (len(features), 840)
    for family, arm in [('static_matched_conv_control', 'scalar_static_conv25'), ('learned_conv_encoder', 'scalar_learned_conv25')]:
        values = [model.predict(features, raw) for model in STATE.temporal[family]]
        temporal = {name: np.mean([v[name] for v in values], axis=0) for name in ('q95', 'beta_n')}
        blended = {n: (.75 * base[index].astype(float) + .25 * temporal[label]).astype(np.float32)
                   for n, index, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}
        result[arm] = {'psirz': base[0], **blended}
    assert set(result) == set(ARMS)
    assert all(np.isfinite(v).all() and v.dtype == np.float32 for arm in result.values() for v in arm.values())
    return result, shot
