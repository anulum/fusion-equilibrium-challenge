# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — locked-candidate independent confirmation prediction
"""Predict only the frozen winner and its matched control using shared eligible trees."""
from pathlib import Path
import json
import numpy as np

from campaign_common import HERE, digest
from window_cache import mapped_array
import campaign_prediction as cp

STATE = {}


def initialize(winner: str, original_trees: bool = False) -> None:
    """Keep all neural weights fixed and replace only the shared tree corpus."""
    assert winner in cp.ARMS[1:]
    cp.initialize()
    alignment = json.loads((HERE / 'temporal/validation_alignment.json').read_text())
    offsets = alignment['offsets']
    STATE.update(winner=winner, bounds={stem: (offsets[i], offsets[i + 1]) for i, stem in enumerate(alignment['stems'])},
                 windows=mapped_array(HERE / 'temporal/windows/rawwindow_validation.npz', 'features'))
    if original_trees:
        return
    assert json.loads((HERE / 'confirmation_trees/receipt.json').read_text())['status'] == 'all_four_eligible_shared_confirmation_trees_complete'
    trees = {}
    for name in cp.TREE_WEIGHTS:
        folder = HERE / 'confirmation_trees' / name
        receipt = json.loads((folder / 'fit_report.json').read_text())
        assert digest(folder / 'tree.ubj') == receipt['model_sha256']
        assert digest(folder / 'lookup.npz') == receipt['lookup_sha256']
        assert receipt['both358_development_and357_confirmation_excluded']
        booster = cp.STATE.xgb.Booster(); booster.load_model(folder / 'tree.ubj'); booster.set_param({'device': 'cpu', 'nthread': 1})
        with np.load(folder / 'lookup.npz', allow_pickle=False) as stored:
            trees[name] = booster, stored['leaf_values'].copy(), stored['base_score'].copy()
    cp.STATE.trees = trees


def predict(path: Path) -> tuple:
    """Preserve native arithmetic while evaluating exactly two predeclared arms."""
    pd = cp.STATE.pd
    row = cp.read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'], cp.STATE.subframe.features_of(path).astype(float)], axis=1)
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    coordinates, labels = cp.STATE.together(cp.STATE.maps, {'discharge': features})
    trees = cp.trees_of(features)
    base = cp.finished(path, row, features, coordinates, labels, trees)
    _, joint = cp.STATE.together(cp.STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    scalars = {n: (.75 * base[index].astype(float) + .25 * joint[label]).astype(np.float32)
               for n, index, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}
    result = {'matched69_scalar8': {'psirz': base[0], **scalars}}
    winner = STATE['winner']
    if winner.startswith('map_'):
        family = winner[4:-2]
        models = cp.STATE.candidate[family]
        candidate_coordinates, candidate_labels = cp.STATE.together(models, {'discharge': features})
        mixed_coordinates = .75 * coordinates + .25 * candidate_coordinates
        mixed_labels = {n: .75 * values + .25 * candidate_labels[n] for n, values in labels.items()}
        residual = None
        if family == 'spatial':
            fields = []
            for model, decoder in zip(models, cp.STATE.decoder, strict=True):
                _, heads = model.predict(features)
                latent = np.column_stack([heads[f'spatial_{i:02d}'] for i in range(16)])
                fields.append((latent @ decoder).reshape(len(features), 65, 65))
            residual = pd.smooth_in_time(.25 * np.mean(fields, axis=0), cp.STATE.finish.smooth_frames)
        final = cp.finished(path, row, features, mixed_coordinates, mixed_labels, trees, residual)
        result[winner] = {'psirz': final[0], **scalars}
    else:
        family = 'learned_conv_encoder' if winner == 'scalar_learned_conv25' else 'static_matched_conv_control'
        if path.stem in STATE['bounds']:
            lo, hi = STATE['bounds'][path.stem]
            raw = STATE['windows'][lo:hi]
        else:
            raw = np.load(HERE / 'temporal/development' / (path.stem + '.npy'), allow_pickle=False)
        assert raw.shape == (len(features), 840)
        values = [model.predict(features, raw) for model in cp.STATE.temporal[family]]
        temporal = {name: np.mean([v[name] for v in values], axis=0) for name in ('q95', 'beta_n')}
        blended = {n: (.75 * base[index].astype(float) + .25 * temporal[label]).astype(np.float32)
                   for n, index, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}
        result[winner] = {'psirz': base[0], **blended}
    assert set(result) == {'matched69_scalar8', winner}
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for fields in result.values() for v in fields.values())
    return result, shot
