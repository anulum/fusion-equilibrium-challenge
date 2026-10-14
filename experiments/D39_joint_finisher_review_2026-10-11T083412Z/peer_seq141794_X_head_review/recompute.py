# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — descriptive peer X-head variance review
"""Recompute empirical averaging effects without identifying a universal error floor."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    protocol = json.loads((HERE / 'protocol_frozen.json').read_text())
    for source, expected in protocol['source_sha256'].items():
        assert digest(HERE / Path(source).name) == expected
    with np.load(HERE / 'xmembers.npz', allow_pickle=False) as stored:
        members = stored['members'].astype(float)
        truth = stored['truth'].copy()
        shots = stored['shot'].copy()
        families = stored['family'].astype(str)
    assert members.shape == (69, 13329, 4) and truth.shape == (13329, 4)
    assert np.isfinite(members).all()
    peer = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')
    cases = json.loads((peer / 'd32/manifest.json').read_text())['cases']
    selected = [c for c in cases if c['index'] % 3 == 0]
    assert len(selected) == 120 and set(shots) == {c['index'] for c in selected}
    for case in selected:
        assert int((shots == case['index']).sum()) == (case['span'][1] - case['span'][0] + 1) // 2
    mean = members.mean(axis=0)
    estimators = {'mean69': mean, 'median69': np.median(members, axis=0),
                  'coordinatewise_trim14_each_tail': np.sort(members, axis=0)[14:55].mean(axis=0)}
    reports = {}
    for name, start in (('lower', 0), ('upper', 2)):
        valid = np.isfinite(truth[:, start:start+2]).all(axis=1)
        target = truth[valid, start:start+2]
        predictions = members[:, valid, start:start+2]
        errors = predictions - target
        average = predictions.mean(axis=0)
        ensemble_error = average-target
        member_mse = float(np.mean(np.sum(errors**2, axis=2)))
        ensemble_mse = float(np.mean(np.sum(ensemble_error**2, axis=1)))
        variance = float(np.mean(np.sum((predictions-average)**2, axis=2)))
        assert np.isclose(member_mse, ensemble_mse+variance, rtol=1e-12, atol=1e-18)
        distances = np.linalg.norm(ensemble_error, axis=1)
        spread = np.sqrt(np.sum(predictions.var(axis=0), axis=1))
        top = np.argsort(-spread)[:len(spread)//10]
        offset_square = 0.
        for shot in np.unique(shots[valid]):
            residual = ensemble_error[shots[valid] == shot]
            offset_square += len(residual)*float(np.sum(residual.mean(axis=0)**2))
        estimates = {}
        for label, prediction in estimators.items():
            values = np.linalg.norm(prediction[valid, start:start+2]-target, axis=1)*1000
            estimates[label] = {'mean_distance_mm': float(values.mean()),
                               'median_distance_mm': float(np.median(values)),
                               'p90_distance_mm': float(np.quantile(values, .9)),
                               'p99_distance_mm': float(np.quantile(values, .99))}
        reports[name] = {'finite_truth_frames': int(valid.sum()), 'estimates': estimates,
                         'mean_individual_member_squared_2D_error_m2': member_mse,
                         'ensemble_squared_2D_error_m2': ensemble_mse,
                         'member_variance_m2': variance,
                         'decomposition_residual_m2': member_mse-ensemble_mse-variance,
                         'ensemble_residual_fraction_of_mean_member_MSE': ensemble_mse/member_mse,
                         'mean_of_all69_individual_member_mean_distances_mm': float(np.linalg.norm(errors, axis=2).mean()*1000),
                         'tie_aware_spread_error_Spearman': float(spearmanr(spread, distances).statistic),
                         'sum_error_fraction_in_top10percent_spread_frames': float(distances[top].sum()/distances.sum()),
                         'truth_derived_per_shot_offset_fraction_of_squared_ensemble_error': offset_square/float(np.sum(ensemble_error**2))}
    result = {'status': 'peer120_every_second_frame_69_neural_head_variance_review_complete',
              'shots': 120, 'sampled_frames': 13329, 'reports': reports,
              'family_counts': {str(name): int((families == name).sum()) for name in np.unique(families)},
              'protocol_sha256': digest(HERE / 'protocol_frozen.json'),
              'saved_float32_members_recomputed_in_float64': True,
              'heads_before_native_tree_blend_and_finisher': True,
              'current_native_x_lower_r_tree_weight': .16,
              'ensemble_residual_is_not_an_unavoidable_model_class_error_floor': True,
              'truth_derived_offset_cannot_be_subtracted_at_inference_without_a_predictor_or_anchor': True,
              'no_linear_conversion_from_X_error_reduction_to_official_score_established': True,
              'no_new_candidate_or_current_grid_change': True, 'no357_targets_read': True, 'publication': False}
    with (HERE / 'analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
