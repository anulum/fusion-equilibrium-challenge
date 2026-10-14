# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — matched controls after the complete D38 study
"""Explain objective/temporal effects without changing original candidate admission."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    for path, expected in json.loads((HERE / 'D38_matched_contrast_sources.json').read_text()).items():
        assert digest(path) == expected, path
    assert json.loads((D38 / 'evaluation/terminal_receipt.json').read_text())['status'] == 'all_seven_development_arms_and_prospective_analysis_complete'
    analysis_path = D38 / 'evaluation/development_analysis.json'
    original_hash = digest(analysis_path)
    original = json.loads(analysis_path.read_text())
    assert original['status'] == 'all_declared_development_arms_analyzed'
    protocol_path = HERE / 'D38_matched_control_secondary_contrasts_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    sys.path.insert(0, str(D38))
    from analyze_development import scores, CASES, anatomy, part_numbers
    rows = [json.loads((D38 / 'evaluation/development' / (c['stem'] + '.json')).read_text()) for c in CASES]
    assert len(rows) == 358 and sum(r['frames'] for r in rows) == 78875
    for case, row in zip(CASES, rows, strict=True):
        assert row['stem'] == case['stem']
        assert digest(D38 / 'evaluation/development' / (case['stem'] + '.npz')) == row['arrays_sha256']
    arms = {arm for pair in protocol['contrasts'] for arm in pair}
    parts = {arm: [part_numbers(r['parts'][arm]) for r in rows] for arm in arms}
    weights = np.random.default_rng(protocol['bootstrap_seed']).multinomial(358, np.full(358, 1. / 358), size=protocol['replicates']).astype(float)
    exact = {arm: scores(np.ones((1, 358)), anatomy.STATISTICS, parts[arm]) for arm in arms}
    sampled = {arm: scores(weights, anatomy.STATISTICS, parts[arm])['S'] for arm in arms}
    contrasts = {}
    for candidate, control in protocol['contrasts']:
        interval = np.quantile(sampled[candidate] - sampled[control], protocol['CI_quantiles'])
        contrasts[candidate + '_versus_' + control] = {
            'candidate': candidate, 'control': control,
            'delta_S': float(exact[candidate]['S'][0] - exact[control]['S'][0]),
            'CI99': interval.tolist(),
            'component_differences': {k: float(exact[candidate][k][0] - exact[control][k][0]) for k in exact[candidate] if k != 'S'},
            'LCFS_failures': {a: sum(p['lcfs_fails'] for p in parts[a]) for a in (candidate, control)},
            'diagnostic_only_not_admission': True}
    assert digest(analysis_path) == original_hash
    with (HERE / 'D38_matched_control_secondary_analysis.json').open('x') as stream:
        json.dump({'status': 'all358_four_predeclared_matched_control_contrasts_complete',
                   'contrasts': contrasts, 'protocol_sha256': digest(protocol_path),
                   'original_analysis_sha256': original_hash,
                   'original_frozen_confirmation_candidate': original['frozen_confirmation_candidate'],
                   'original_selection_and357_rules_unchanged': True, 'not_independent_replication': True,
                   'bootstrap_seed': protocol['bootstrap_seed'], 'replicates': protocol['replicates'],
                   'CI_per_contrast_not_familywise': True, 'publication': False}, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
