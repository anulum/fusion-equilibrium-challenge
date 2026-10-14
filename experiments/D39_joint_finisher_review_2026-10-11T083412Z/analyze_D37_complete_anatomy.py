# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete anatomy descriptive attribution
"""Quantify fixed-finisher effects; these production-model cases are in sample."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D37 = HERE.parent / 'D37_H69_boundary_anatomy_2026-10-10T235436Z'
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    for source, expected in json.loads((HERE / 'D37_complete_attribution_sources.json').read_text()).items():
        assert digest(source) == expected, source
    terminal = json.loads((D37 / 'timeout_resume_terminal_v2.json').read_text())
    assert terminal['status'] == 'all358_all14stages_completed_after_operational_timeout'
    summary = json.loads((D37 / 'summary.json').read_text())
    rows = sorted([json.loads(p.read_text()) for p in (D37 / 'cases').glob('*.json')], key=lambda r: r['index'])
    assert len(rows) == 358 and sum(r['frames'] for r in rows) == 78875
    cases = json.loads((PEER / 'd32/manifest.json').read_text())['cases']
    assert [r['stem'] for r in rows] == [c['stem'] for c in cases]
    for row in rows:
        assert digest(D37 / 'cases' / (row['stem'] + '.npz')) == row['arrays_sha256']
        assert set(row['parts']) == set(summary['results'])
    load('d39_common', HERE / 'peer_common_frozen.py')
    report = load('d39_anatomy_score', HERE / 'peer_amendment5_sources/d39_report_v3.py')
    references = json.loads((PEER / 'd32/reference_statistics.json').read_text())
    parts = {arm: [r['parts'][arm] for r in rows] for arm in summary['results']}
    exact = {arm: {k: float(v[0]) for k, v in report.scores(np.ones((1, 358)), references, parts[arm]).items()
                   if k != 'cons_each'} for arm in parts}
    keys = {'S': 'S', 'psi': 'r2_psi', 'qb': 'r2_qb', 'D': 'dlcfs', 'C': 'consistency'}
    assert all(round(v, 6) == summary['results'][arm][keys[k]] for arm, values in exact.items() for k, v in values.items())
    weights = np.random.default_rng(3937).multinomial(358, np.full(358, 1. / 358), size=10000).astype(float)
    selected = ('raw64', 'final', 'oracle45_raw', 'oracle45_final', 'oracle64_raw', 'oracle64_final', 'oracle128_raw', 'oracle128_final')
    sampled = {arm: report.scores(weights, references, parts[arm])['S'] for arm in selected}
    pairs = [('final', 'raw64'), *[(f'oracle{k}_final', f'oracle{k}_raw') for k in (45, 64, 128)],
             ('oracle128_raw', 'oracle45_raw'), ('oracle128_final', 'oracle45_final')]
    comparisons = {a + '_minus_' + b: {'delta_S': exact[a]['S'] - exact[b]['S'],
                    'CI99': np.quantile(sampled[a] - sampled[b], [.005, .995]).tolist(),
                    'weighted_components': {'flux': .55 * (exact[a]['psi'] - exact[b]['psi']),
                                            'scalars': .15 * (exact[a]['qb'] - exact[b]['qb']),
                                            'LCFS': -.1 * (exact[a]['D'] - exact[b]['D']),
                                            'consistency': .2 * (exact[a]['C'] - exact[b]['C'])}}
                   for a, b in pairs}
    interaction = sampled['oracle128_final'] - sampled['oracle45_final'] - sampled['oracle128_raw'] + sampled['oracle45_raw']
    comparisons['finisher_changes_45_to128_resolution_benefit'] = {
        'delta_S': exact['oracle128_final']['S'] - exact['oracle45_final']['S'] - exact['oracle128_raw']['S'] + exact['oracle45_raw']['S'],
        'CI99': np.quantile(interaction, [.005, .995]).tolist()}
    sequence = ('raw64', 'filtered45', 'axis', 'contour', 'shape', 'x_points', 'reshape', 'final')
    output = {'status': 'complete_all14_unrounded_descriptive_anatomy_attribution',
              'shots': 358, 'frames': 78875, 'unrounded_scores': exact, 'comparisons': comparisons,
              'adjacent_stage_delta_S': {b + '_minus_' + a: exact[b]['S'] - exact[a]['S'] for a, b in zip(sequence[:-1], sequence[1:], strict=True)},
              'bootstrap_seed': 3937, 'replicates': 10000, 'CI_per_contrast_not_familywise': True,
              'comparisons_declared_after_complete_summary_read': True,
              'in_sample_production_models': True, 'not_generalization_or_candidate_admission': True,
              'projected_truth_oracles_non_deployable': True,
              'observed_intervention_not_formal_performance_upper_bound': True,
              'fixed_predicted_geometry_heads_retained_for_all_oracles': True,
              'D_is_normalized_distance_not_metres': True, 'no357_new_candidate_scored': True, 'publication': False}
    with (HERE / 'D37_complete_descriptive_attribution.json').open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
