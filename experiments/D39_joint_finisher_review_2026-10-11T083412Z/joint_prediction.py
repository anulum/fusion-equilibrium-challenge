# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed D38/D39 central prediction
"""Apply one locked finisher to the predeclared neural contribution."""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
import campaign_prediction as cp

BASE = 'matched69_scalar8'


class Resampler:
    """Match the frozen D39 override of whole-grid warp interpolation."""

    def __init__(self, module, order: int):
        self.module, self.order = module, order

    def __getattr__(self, name):
        return getattr(self.module, name)

    def map_coordinates(self, flux, coordinates, order=1, mode='nearest', **kwargs):
        if np.ndim(coordinates[0]) == 2:
            order = self.order
        return self.module.map_coordinates(flux, coordinates, order=order, mode=mode, **kwargs)


def validate_change(change: dict) -> None:
    """Reject target-dependent diagnostics and unknown deployment parameters."""
    allowed = {f.name for f in fields(cp.STATE.finish)} | {'tree_scale', 'warp_order'}
    assert set(change) <= allowed, sorted(set(change) - allowed)
    if 'warp_order' in change:
        assert type(change['warp_order']) is int and change['warp_order'] == 3
    assert np.isfinite(float(change.get('tree_scale', 1.)))


def finish_of(change: dict):
    validate_change(change)
    options = {k: v for k, v in change.items() if k not in ('tree_scale', 'warp_order')}
    if 'scalar_shares' in options:
        options['scalar_shares'] = tuple((n, float(w)) for n, w in options['scalar_shares'])
    return replace(cp.STATE.finish, **options)


def finished(path, row, features, coordinates, labels, trees, change, residual=None):
    """Retain D39's scalar operations and restore every temporary native override."""
    from scipy import ndimage

    pd = cp.STATE.pd
    finish = finish_of(change)
    merged = {n: v.copy() for n, v in labels.items()}
    scale = float(change.get('tree_scale', 1.))
    for name, weight in cp.TREE_WEIGHTS.items():
        merged[name] += scale * weight * (trees[name].astype(float) - merged[name])
    modules = [pd.contour_edit, sys.modules['shape_edit_fixed_axis_volume']]
    assert all(m.ndimage is ndimage for m in modules)
    kept = pd.read_row, pd.layout_features, pd.predict_together, cp.STATE.residual
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    pd.predict_together = lambda *_args: (coordinates.copy(), {n: v.copy() for n, v in merged.items()})
    cp.STATE.residual = residual
    if 'warp_order' in change:
        for module in modules:
            module.ndimage = Resampler(ndimage, change['warp_order'])
    try:
        return pd.predict_shot(path, cp.STATE.maps, cp.STATE.basis, finish)
    finally:
        pd.read_row, pd.layout_features, pd.predict_together, cp.STATE.residual = kept
        for module in modules:
            module.ndimage = ndimage


def initialize() -> None:
    """Load the already qualified, two-cohort-excluded D38 models and trees."""
    cp.initialize()


def blend_scalars(base, heads):
    return {n: (.75 * base[i].astype(float) + .25 * heads[label]).astype(np.float32)
            for n, i, label in (('q95', 1, 'q95'), ('betaN', 2, 'beta_n'))}


def predict(path: Path, d38_winner: str | None, d39_change: dict | None,
            temporal_raw: np.ndarray | None = None) -> tuple:
    """Predict only the baseline, eligible finalists and their one fixed composition."""
    assert d38_winner is None or d38_winner in cp.ARMS[1:]
    pd = cp.STATE.pd
    row = cp.read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'],
                               cp.STATE.subframe.features_of(path).astype(float)], axis=1)
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    coordinates, labels = cp.STATE.together(cp.STATE.maps, {'discharge': features})
    trees = cp.trees_of(features)
    base = finished(path, row, features, coordinates, labels, trees, {})
    _, joint = cp.STATE.together(cp.STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    scalars = blend_scalars(base, joint)
    result = {BASE: {'psirz': base[0], **scalars}}
    d39 = None
    if d39_change is not None:
        d39 = finished(path, row, features, coordinates, labels, trees, d39_change)
        result['d39_finalist'] = {'psirz': d39[0], **blend_scalars(d39, joint)}
    if d38_winner is not None and d38_winner.startswith('map_'):
        family = d38_winner[4:-2]
        models = cp.STATE.candidate[family]
        candidate_coordinates, candidate_labels = cp.STATE.together(models, {'discharge': features})
        mixed_coordinates = .75 * coordinates + .25 * candidate_coordinates
        mixed_labels = {n: .75 * values + .25 * candidate_labels[n] for n, values in labels.items()}
        residual_raw = None
        if family == 'spatial':
            extra = []
            for model, decoder in zip(models, cp.STATE.decoder, strict=True):
                _, heads = model.predict(features)
                latent = np.column_stack([heads[f'spatial_{i:02d}'] for i in range(16)])
                extra.append((latent @ decoder).reshape(len(features), 65, 65))
            residual_raw = .25 * np.mean(extra, axis=0)
        for arm, change in [('d38_finalist', {}), *([('composition', d39_change)] if d39_change is not None else [])]:
            residual = None if residual_raw is None else pd.smooth_in_time(residual_raw, finish_of(change).smooth_frames)
            final = finished(path, row, features, mixed_coordinates, mixed_labels, trees, change, residual)
            result[arm] = {'psirz': final[0], **scalars}
    elif d38_winner is not None:
        family = 'learned_conv_encoder' if d38_winner == 'scalar_learned_conv25' else 'static_matched_conv_control'
        raw = temporal_raw if temporal_raw is not None else np.load(D38 / 'temporal/development' / (path.stem + '.npy'), allow_pickle=False)
        assert raw.shape == (shot.n_frames, 840)
        values = [model.predict(features, raw) for model in cp.STATE.temporal[family]]
        temporal = {n: np.mean([v[n] for v in values], axis=0) for n in ('q95', 'beta_n')}
        candidate_scalars = blend_scalars(base, temporal)
        result['d38_finalist'] = {'psirz': base[0], **candidate_scalars}
        if d39 is not None:
            result['composition'] = {'psirz': d39[0], **candidate_scalars}
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for arm in result.values() for v in arm.values())
    return result, shot
