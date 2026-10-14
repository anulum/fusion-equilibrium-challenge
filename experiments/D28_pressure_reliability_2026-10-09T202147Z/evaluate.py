# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — exact scalar-only common evaluation
"""Reuse unchanged-map statistics after validating the full scorer surface."""

from __future__ import annotations

import copy
import json
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from reliability import D26, D27, HERE, digest, write_json
from build_predictions import ARMS, ARCHIVE

sys.path.insert(0, str(D26))
import finish_and_score as previous


def parts_from_json() -> list[dict]:
    """Restore the exact cached official per-shot sufficient statistics."""
    parts = json.loads((D26 / 'full_score_v2/p45_parts.json').read_text())
    for row in parts:
        for key in ('ss_res_psi', 'shot_r2_psi'):
            row[key] = {float(k): v for k, v in row[key].items()}
        for key in ('ss_res_scal', 'ss_res_cons', 'cons_frames', 'cons_fails'):
            row[key] = np.asarray(row[key])
    return parts


def check_full(task: tuple) -> dict:
    """Compare cached-term substitution with the original public shot scorer."""
    case, arm, expected, means = task
    score = previous.scorer()
    truth = score._truth(Path(case['path']))
    reference = pickle.loads((D26 / 'full_score_v2/references' / (case['stem'] + '.pkl')).read_bytes())
    prediction = score._prediction(ARCHIVE, case['index'])
    with np.load(HERE / 'predictions/shots' / (case['stem'] + '.npz'), allow_pickle=False) as values:
        prediction['betaN'] = values[arm].astype(float)
    r, z, mask, floating = score._grid()
    actual = score.local_score.score_shot(truth, reference, prediction, r, z, mask, floating, 1., means)
    maximum = 0.
    for key in actual:
        a, b = actual[key], expected[key]
        if isinstance(a, dict):
            for entry in a:
                delta = abs(a[entry] - b[entry]) / (1 + abs(b[entry]))
                maximum = max(maximum, delta)
                assert delta < 1e-10, (key, entry, delta)
        elif a is None or isinstance(a, (bool, int)):
            assert a == b, key
        else:
            delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / (1 + np.abs(b))))
            maximum = max(maximum, delta)
            assert delta < 1e-10, (key, delta)
    return {'index': case['index'], 'arm': arm, 'full_public_scorer_relative_max': maximum}


def main() -> None:
    """Score all complete arms against the learned baseline on every shot."""
    fit = json.loads((HERE / 'factors/receipt.json').read_text())
    predictions = json.loads((HERE / 'predictions/receipt.json').read_text())
    inputs = json.loads((HERE / 'inputs/receipt.json').read_text())
    assert len(fit['models']) == 5 and predictions['shots'] == 400
    assert inputs['status'] == 'all_input_support_prepared'
    assert digest(ARCHIVE) == predictions['baseline_archive_sha256']
    out = HERE / 'evaluation'
    out.mkdir(exist_ok=True)
    if (out / 'summary.json').exists():
        return
    cases = json.loads((D27 / 'evaluation/manifest.json').read_text())['cases']
    assert [c['index'] for c in cases] == list(range(400))
    baseline = parts_from_json()
    parts = {arm: copy.deepcopy(baseline) for arm in ARMS}
    for case in cases:
        index = case['index']
        file = HERE / 'predictions/shots' / (case['stem'] + '.npz')
        assert digest(file) == predictions['records'][index]['sha256']
        truth = np.asarray(pq.read_table(case['path'], columns=['efit_beta_n'], use_threads=False)['efit_beta_n'][0].as_py(), dtype=float)
        defined = np.isfinite(truth)
        with np.load(file, allow_pickle=False) as values:
            original_sse = float(np.sum((truth[defined] - values['baseline'][defined].astype(float))**2))
            assert np.isclose(original_sse, baseline[index]['ss_res_scal'][1], rtol=1e-10, atol=1e-10)
            for arm in ARMS:
                assert len(values[arm]) == len(truth) and np.isfinite(values[arm]).all()
                parts[arm][index]['ss_res_scal'][1] = float(np.sum((truth[defined] - values[arm][defined].astype(float))**2))
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        statistics = list(pool.map(previous.reference_case,
            [(Path(c['path']), D26 / 'full_score_v2/references') for c in cases]))
    stats = {key: sum(x[key] for x in statistics) for key in statistics[0]}
    means = (stats['psi_sum'] / stats['psi_n'], stats['scal_sum'] / stats['scal_n'],
             stats['cons_sum'] / stats['cons_n'])
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        parity = list(pool.map(check_full, [(cases[0], ARMS[0], parts[ARMS[0]][0], means),
                                          (cases[227], ARMS[2], parts[ARMS[2]][227], means)]))
    score = previous.scorer()
    results = {}
    for arm, rows in {'baseline_nn': baseline, **parts}.items():
        acc = score.Accum()
        for row in rows:
            acc.add(row)
        acc.psi_sign = 1.
        result = score.finalize_machine(acc, stats)
        result['S_unrounded'] = previous.unrounded_score(rows, statistics, np.ones(400))
        assert abs(result['S_unrounded'] - result['S']) < 5.01e-7
        results[arm] = result
    denominator = stats['scal_sumsq'][1] - stats['scal_sum'][1]**2 / stats['scal_n'][1]
    rng = np.random.default_rng(991)
    draws = [np.bincount(rng.integers(0, 400, 400), minlength=400) for _ in range(2000)]
    comparisons = {}
    for arm, rows in parts.items():
        boots = [previous.unrounded_score(rows, statistics, draw) - previous.unrounded_score(baseline, statistics, draw)
                 for draw in draws]
        delta = results[arm]['S_unrounded'] - results['baseline_nn']['S_unrounded']
        probability = float(np.mean(np.asarray(boots) > 0))
        beta_gain = (sum(r['ss_res_scal'][1] for r in baseline) - sum(r['ss_res_scal'][1] for r in rows)) / denominator
        comparisons[arm] = {'control': 'baseline_nn', 'delta_S': delta, 'delta_R2_betaN_unrounded': beta_gain,
            'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'P_positive': probability,
            'admission_passes': bool(delta >= 3e-5 and probability >= .99 and beta_gain >= .0004),
            'no_changed_map_or_q95': True, 'all_frames_scored': True}
    write_json(out / 'summary.json', {'status': 'all_three_guarded_comparisons_complete', 'results': results,
        'comparisons': comparisons, 'full_scorer_parity': parity, 'shots': 400, 'frames': 89584,
        'development_population_reused_requires_reserved_confirmation': True,
        'MAST_score_computed': False, 'public_upload_or_submission': False})
    print(json.dumps({'all_D28_comparisons_complete': True}), flush=True)


if __name__ == '__main__':
    main()
