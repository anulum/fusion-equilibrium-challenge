# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete scalar-task comparisons
"""Reproduce existing scalar archive bytes on real held-out shots before new fits."""

import json
import numpy as np
from campaign_common import HERE, D26, D27, base_maps, feature_block, write_json
from evaluate_batch import baseline_models, prediction, NAMES


def main() -> None:
    """Check one complete real shot per fold against both original controls."""
    base = base_maps()
    cases = json.loads((D27 / 'evaluation/manifest.json').read_text())['cases']
    archives = {'stronger_p45': D26 / 'full_score_v2/p45.npz',
                'matched_control': D27 / 'evaluation/control45.npz'}
    results = []
    for fold in range(5):
        case = next(c for c in cases if c['fold'] == fold)
        a, b = case['span']
        features = feature_block(base['training'], {}, np.arange(a, b), 'control')
        networks = baseline_models(fold)
        hx = sum(prediction(n, features, NAMES) for n in networks['hx_shared'])
        for arm, role in [('stronger_p45', 'hc_p'), ('matched_control', 'control')]:
            actual = ((hx + sum(prediction(n, features, NAMES) for n in networks[role])) / 4).astype(np.float32)
            with np.load(archives[arm], allow_pickle=False) as stored:
                expected = np.column_stack([stored[f"shot_{case['index']:04d}_{field}"] for field in ('q95', 'betaN')])
            exact = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
            assert exact, (fold, arm, float(np.max(np.abs(actual - expected))))
            results.append({'fold': fold, 'arm': arm, 'stem': case['stem'], 'frames': b-a,
                            'public_numpy_predictor_archive_float32_bitexact': True})
    write_json(HERE / 'baseline_qualification.json', {
        'status': 'all5fold_both_scalar_controls_bitexact', 'cases': results,
        'comparative_candidate_scores_computed': False})


if __name__ == '__main__':
    main()
