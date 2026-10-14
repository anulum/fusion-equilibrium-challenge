# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — paired whole-shot prospective admission analysis
"""Resample entire shots and recompute pooled denominators and equal-shot LCFS."""
from __future__ import annotations

import json
import numpy as np

from campaign_common import HERE, write_json
from evaluate_development import CASES, part_numbers, aggregate, anatomy, prediction


def scores(counts: np.ndarray, references: list, parts: list) -> dict:
    """Compute unrounded official components for paired full-shot resampling."""
    totals = {k: counts @ np.asarray([r[k] for r in references], dtype=float) for k in references[0]}

    def denominator(prefix: str) -> np.ndarray:
        return totals[prefix + '_sumsq'] - totals[prefix + '_sum'] ** 2 / np.maximum(totals[prefix + '_n'], 1.)

    flux_total = denominator('psi')
    scalar_total = denominator('scal')
    consistency_total = denominator('cons')
    flux_residual = counts @ np.asarray([p['ss_res_psi'][1.] for p in parts])
    scalar_residual = counts @ np.asarray([p['ss_res_scal'] for p in parts])
    consistency_residual = counts @ np.asarray([p['ss_res_cons'] for p in parts])
    flux = np.where(flux_total > 0., 1. - flux_residual / np.maximum(flux_total, 1e-300), 0.)
    scalar = np.where(scalar_total > 0., 1. - scalar_residual / np.maximum(scalar_total, 1e-300), 0.).mean(axis=1)
    consistency_each = np.where(consistency_total > 0., 1. - consistency_residual / np.maximum(consistency_total, 1e-300), np.nan)
    consistency = np.nanmean(np.maximum(0., consistency_each), axis=1)
    distance = np.asarray([np.nan if p['shot_dlcfs'] is None else p['shot_dlcfs'] for p in parts])
    valid = np.isfinite(distance)
    valid_count = counts @ valid.astype(float)
    boundary = np.where(valid_count > 0., (counts @ np.nan_to_num(distance)) / np.maximum(valid_count, 1.), 1.)
    result = .55 * np.maximum(0., flux) + .15 * np.maximum(0., scalar) + .1 * (1. - np.minimum(1., boundary)) + .2 * consistency
    return {'S': result, 'r2_psi': flux, 'r2_qb': scalar, 'dlcfs': boundary, 'consistency': consistency}


def main() -> None:
    """Report all arms and a frozen winner only after every declared result is complete."""
    assert json.loads((HERE / 'candidate_terminal.json').read_text())['all125_successful']
    completed = json.loads((HERE / 'evaluation/development_scores.json').read_text())
    assert completed['status'] == 'all358_whole_shot_seven_arm_scores_complete'
    config = json.loads((HERE / 'bootstrap_protocol.json').read_text())
    rows = [json.loads((HERE / 'evaluation/development' / (c['stem'] + '.json')).read_text()) for c in CASES]
    references = anatomy.STATISTICS
    official = aggregate(rows)
    exact = {}
    parts = {arm: [part_numbers(r['parts'][arm]) for r in rows] for arm in prediction.ARMS}
    all_once = np.ones((1, 358), dtype=float)
    for arm in prediction.ARMS:
        exact[arm] = {k: float(v[0]) for k, v in scores(all_once, references, parts[arm]).items()}
        assert all(round(v, 6) == official[arm][k] for k, v in exact[arm].items()), (arm, exact[arm], official[arm])
    rng = np.random.default_rng(config['seed'])
    weights = rng.multinomial(358, np.full(358, 1. / 358), size=config['replicates']).astype(float)
    distributions = {arm: scores(weights, references, parts[arm])['S'] for arm in prediction.ARMS}
    baseline = 'matched69_scalar8'
    decisions = {}
    for arm in prediction.ARMS[1:]:
        differences = distributions[arm] - distributions[baseline]
        interval = np.quantile(differences, [.005, .995])
        delta = exact[arm]['S'] - exact[baseline]['S']
        failures = sum(p['lcfs_fails'] for p in parts[arm])
        baseline_failures = sum(p['lcfs_fails'] for p in parts[baseline])
        nonfinite = sum(p['psi_nonfinite'] for p in parts[arm])
        decisions[arm] = {'delta_S': delta, 'paired_shot_CI99': interval.tolist(),
                          'LCFS_failures': failures, 'baseline_LCFS_failures': baseline_failures,
                          'nonfinite_pixels': nonfinite,
                          'development_gate_pass': bool(delta >= 3e-5 and interval[0] > 0 and failures <= baseline_failures and nonfinite == 0)}
    contrasts = {}
    for arm, control in [('map_boundary_frame25', 'map_control25'), ('map_boundary_shot25', 'map_control25'),
                         ('map_spatial25', 'map_control25'), ('scalar_learned_conv25', 'scalar_static_conv25'),
                         ('map_boundary_shot25', 'map_boundary_frame25')]:
        differences = distributions[arm] - distributions[control]
        contrasts[arm + '_minus_' + control] = {'delta_S': exact[arm]['S'] - exact[control]['S'],
                                             'paired_shot_CI99': np.quantile(differences, [.005, .995]).tolist(),
                                             'descriptive_only': True}
    passed = [arm for arm, result in decisions.items() if result['development_gate_pass']]
    winner = max(passed, key=lambda arm: exact[arm]['S']) if passed else None
    write_json(HERE / 'evaluation/development_analysis.json', {'status': 'all_declared_development_arms_analyzed',
               'unrounded_scores': exact, 'official_scores': official, 'decisions': decisions,
               'paired_family_contrasts': contrasts, 'frozen_confirmation_candidate': winner,
               'bootstrap_protocol': config, 'candidate_admitted': False,
               'next': 'separate357_confirmation_with_refitted_shared_trees' if winner else 'no_candidate_passed_no_submission',
               'CI_is_per_contrast_not_familywise': True})


if __name__ == '__main__':
    main()
