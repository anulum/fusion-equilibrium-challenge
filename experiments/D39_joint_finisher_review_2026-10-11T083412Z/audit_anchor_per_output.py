# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible training anchor-error attribution
"""Measure exported control errors without equating nominal weights with gradients."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import TABLES, metadata, mapped_array, digest
from fec_adapter.diii_d_accuracy.network import Network


def main():
    cohort = json.loads((D38 / 'cohort.json').read_text())
    allowed = set(cohort['training'])
    assert len(allowed) == 5954
    assert not (allowed & (set(cohort['development']) | set(cohort['reserved_confirmation'])))
    meta = metadata('training')
    chosen = [(stem, (meta['bounds'][i] + meta['bounds'][i + 1]) // 2)
              for i, stem in enumerate(meta['stems']) if stem in allowed]
    assert len(chosen) == 5954 and {s for s, _row in chosen} == allowed
    ids = np.asarray([i for _s, i in chosen], dtype=np.int64)
    assert all(meta['bounds'][i] <= (meta['bounds'][i] + meta['bounds'][i + 1]) // 2 < meta['bounds'][i + 1]
               for i, s in enumerate(meta['stems']) if s in allowed)
    table = TABLES / 'table_training.npz'
    features = mapped_array(table, 'features')[ids].astype(float)
    labels = mapped_array(table, 'labels')[ids].astype(float)
    coordinates = mapped_array(table, 'coordinates')[ids, :64].astype(float)
    truth = np.column_stack([coordinates, labels])
    assert features.shape == (5954, 582) and truth.shape == (5954, 142)
    names = list(meta['label_names'])
    x_names = ['x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
    x_indices = [64 + names.index(n) for n in x_names]
    groups = {'coordinates64': np.arange(64), 'X_positions4': np.asarray(x_indices),
              'other_labels74': np.asarray([i for i in range(64, 142) if i not in x_indices])}
    weights = np.r_[np.ones(64), np.full(78, .05)]
    results = {}
    for seed in range(400, 408):
        path = D38 / 'map_fits' / f'control_{seed}' / 'network.npz'
        receipt = json.loads(path.with_name('fit_report.json').read_text())
        model_hash = digest(path)
        assert model_hash == receipt['network_sha256']
        model = Network.load(path)
        assert list(model.label_names) == names and model.coordinates == 64
        coords, heads = model.predict(features)
        predicted = np.column_stack([coords, *[heads[n] for n in names]])
        assert np.isfinite(predicted).all()
        valid = np.isfinite(truth)
        standardized = np.where(valid, (predicted - np.nan_to_num(truth)) / model.output_scale, 0.)
        errors = standardized**2 * weights
        attribution = {n: {"weighted_MSE_over_all_shots": float(errors[:, i].mean()),
                            "finite_targets": int(valid[:, i].sum()),
                            "output_scale": float(model.output_scale[i]),
                            "fraction_of_total_anchor_error": float(errors[:, i].sum() / errors.sum())}
                       for i, n in enumerate([f"coordinate_{j}" for j in range(64)] + names)}
        contribution = {group: float(np.mean(np.sum(errors[:, indices], axis=1))) for group, indices in groups.items()}
        total = sum(contribution.values())
        rms = {n: float(np.sqrt(np.mean((predicted[valid[:, i], i] - truth[valid[:, i], i])**2)))
               for n, i in zip(x_names, x_indices, strict=True)}
        results[str(seed)] = {'model_sha256': model_hash, 'per_output_attribution': attribution, 'weighted_anchor_error_components': contribution,
                              'X_fraction_of_this_sample_anchor_error': contribution['X_positions4'] / total,
                              'X_position_RMSE_metres': rms,
                              'X_valid_midpoints': {n: int(valid[:, i].sum()) for n, i in zip(x_names, x_indices, strict=True)}}
    result = {'status': 'eight_map_control_eligible_training_midpoint_per_output_anchor_audit_complete',
              'shots': 5954, 'one_midpoint_per_eligible_training_shot': True,
              'sample_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
              'development_and357_labels_not_read': True,
              'model_stage': 'fixed_epoch60_map_controls_before_geometry_readout',
              'native_double_export_error_attribution_not_actual_float32_training_gradient': True,
              'not_finished_map_performance_or_generalization': True,
              'current_protocol_unchanged': True, 'candidate_admitted': False, 'results': results}
    with (HERE / 'training_anchor_per_output_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
