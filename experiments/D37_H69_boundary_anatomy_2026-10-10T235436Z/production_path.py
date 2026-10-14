# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — instrument the verified H69 map pipeline
"""Capture native production stages without changing prediction arithmetic."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parent
R = Path('$HOME/fusion-equilibrium-challenge')
SAS = Path('$FEC_RUNS')
PKG = SAS / 'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
V29 = SAS / 'ml350/runs/ensembles_20261007/v29'
D32 = E / 'D32_DIIID_separate_cohort_confirmation_2026-10-10T144836Z'
STATE = SimpleNamespace()
WEIGHTS = {'li_scored': .26, 'r_axis': .14, 'kappa': .09, 'x_lower_r': .16}


def digest(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(name: str, values: np.ndarray) -> None:
    """Retain float32 submission-stage maps while capture is enabled."""
    if STATE.capture:
        STATE.stages[name] = np.asarray(values, dtype=np.float32).copy()


def initialize() -> None:
    """Load exact production members, native finisher and qualified tree sums."""
    for path in (PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter',
                 PKG / 'entrypoints', D32 / 'vendor'):
        sys.path.insert(0, str(path))
    import predict_diii_d as pd
    import xgboost as xgb
    import pyarrow.parquet as pq
    read_table = pq.read_table

    def single_thread_table(*args, **kwargs):
        kwargs['use_threads'] = False
        return read_table(*args, **kwargs)

    pq.read_table = single_thread_table
    spec = importlib.util.spec_from_file_location('production_subframe', V29 / 'subframe_features.py')
    subframe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subframe)
    builder_spec = importlib.util.spec_from_file_location('published_builder', V29 / 'deploy_build_subframe.py')
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    protocol = json.loads((HERE / 'protocol_v3.json').read_text())
    networks = []
    for item in protocol['models']:
        assert digest(Path(item['path'])) == item['sha256']
        networks.append(pd.Network.load(item['path']))
    assert len(networks) == 69
    joint = []
    for item in protocol['joint_models']:
        assert digest(Path(item['path'])) == item['sha256']
        joint.append(pd.Network.load(item['path']))
    assert len(joint) == 8 and all(network.coordinates == 0 for network in joint)
    manifest = json.loads((D32 / 'manifest.json').read_text())
    trees = {}
    for name, item in manifest['trees'].items():
        assert digest(Path(item['path'])) == item['sha256']
        booster = xgb.Booster()
        booster.load_model(item['path'])
        booster.set_param({'device': 'cpu', 'nthread': 1})
        with np.load(D32 / 'tree_lookups' / (name + '.npz'), allow_pickle=False) as stored:
            trees[name] = booster, stored['leaf_values'], stored['base_score']
    args = pd._parser().parse_args(['--data-root', '/dev/null', '--network', protocol['models'][0]['path'],
        '--basis', str(PKG / 'models/flux_basis.npz'), '--out', str(HERE),
        '--split', 'diii_d_train', *builder.FLAGS])
    basis = pd.FluxBasis.load(PKG / 'models/flux_basis.npz')
    assert len(basis.directions) >= 128
    STATE.__dict__.update(pd=pd, xgb=xgb, networks=networks, joint=joint, trees=trees, subframe=subframe,
        finish=pd._finish_from(args), basis=basis, together=pd.predict_together,
        read_row=pd.read_row, layout_features=pd.layout_features, decode=pd.FluxBasis.decode,
        capture=False, stages={}, edits=0)

    def bounded_inputs(path):
        columns = pd.input_columns(dict.fromkeys(pq.read_schema(path).names))
        table = pq.read_table(path, columns=columns)
        row = {}
        for name in columns:
            value = table[name][0]
            kind = table.schema.field(name).type
            flat = name.startswith('magnetics_') and str(kind) in ('list<element: double>', 'list<element: float>')
            row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
        return row

    STATE.legacy_read_row = STATE.read_row
    STATE.read_row = bounded_inputs

    def decode(basis, coordinates):
        result = STATE.decode(basis, coordinates)
        capture('filtered45', result)
        return result

    pd.FluxBasis.decode = decode
    for function, label in (('moved_onto', 'axis'), ('_contour_bent', 'contour'),
                            ('edited_x_points', 'x_points'), ('_kept_within_margin', 'final')):
        original = getattr(pd, function)

        def wrapped(*args, _original=original, _label=label, **kwargs):
            result = _original(*args, **kwargs)
            capture(_label, result)
            return result

        setattr(pd, function, wrapped)
    original_edit = pd.edited

    def edited(*args, **kwargs):
        result = original_edit(*args, **kwargs)
        if STATE.capture:
            STATE.edits += 1
            capture('shape' if STATE.edits == 1 else 'reshape', result[0])
        return result

    pd.edited = edited


def tree_answers(features: np.ndarray) -> dict[str, np.ndarray]:
    """Use the verified original GPU scalar tree accumulation order on CPU."""
    matrix = STATE.xgb.DMatrix(np.asarray(features[:, :367], dtype=np.float32, order='C'), nthread=1)
    result = {}
    for name, (booster, leaves, base) in STATE.trees.items():
        indices = booster.predict(matrix, pred_leaf=True, iteration_range=(0, leaves.shape[0])).astype(np.int32)
        contribution = leaves[np.arange(leaves.shape[0])[None, :], indices]
        assert np.isfinite(contribution).all()
        result[name] = np.cumsum(contribution, axis=1, dtype=np.float32)[:, -1] + base
    return result


def maps_of(path: Path, truth: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], object]:
    """Predict native stages and optional truth-coefficient diagnostic interventions."""
    pd = STATE.pd
    row = STATE.read_row(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([STATE.layout_features(row, shot, {'discharge'})['discharge'],
                               STATE.subframe.features_of(path).astype(float)], axis=1)
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    coordinates, labels = STATE.together(STATE.networks, {'discharge': features})
    tree = tree_answers(features)
    if path.parent.name == 'diii_d_public_test':
        with np.load(V29 / 'tree_heads_public_test.npz', allow_pickle=False) as stored:
            stems = stored['stems'].astype(str).tolist()
            index = stems.index(path.stem)
            start, stop = stored['offsets'][index:index + 2]
            names = stored['targets'].astype(str).tolist()
            precomputed = {name: stored['pred'][int(start):int(stop), names.index(name)] for name in WEIGHTS}
        differences = {name: float(np.max(np.abs(tree[name] - precomputed[name]))) for name in WEIGHTS}
        (HERE / 'profiles/public_tree_context_differences.json').write_text(json.dumps(differences, indent=2))
        tree = precomputed
    for name, weight in WEIGHTS.items():
        labels[name] = labels[name] + weight * (tree[name].astype(float) - labels[name])
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    pd.predict_together = lambda *_args: (coordinates.copy(), {k: v.copy() for k, v in labels.items()})
    STATE.stages, STATE.edits = {}, 0
    STATE.capture = True
    capture('raw64', STATE.decode(STATE.basis.truncated(64), coordinates))
    final = pd.predict_shot(path, STATE.networks, STATE.basis.truncated(64), STATE.finish)
    STATE.capture = False
    assert np.array_equal(STATE.stages['final'], final[0])
    stages = STATE.stages.copy()
    if truth is not None:
        true_coordinates = STATE.basis.truncated(128).encode(truth.astype(float))
        for size in (45, 64, 128):
            prefix = 'oracle' + str(size)
            stages[prefix + '_raw'] = STATE.decode(STATE.basis.truncated(size), true_coordinates[:, :size]).astype(np.float32)
            pd.predict_together = lambda *_args, _size=size: (true_coordinates[:, :_size].copy(), {k: v.copy() for k, v in labels.items()})
            stages[prefix + '_final'] = pd.predict_shot(path, STATE.networks, STATE.basis.truncated(size),
                replace(STATE.finish, kept_coordinates=size))[0]
    _, joint_labels = STATE.together(STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    scalars = {name: (.75 * final[index].astype(float) + .25 * joint_labels[label]).astype(np.float32)
               for name, index, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}
    return stages, scalars, shot
