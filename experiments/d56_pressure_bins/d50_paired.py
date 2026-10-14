# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D50 gate: treatment against control, paired over shots
"""Apply the frozen rule of protocol_d50_gate.json.

Each arm's corrected heads are blended into the network heads exactly as d9_gate_eval.py does (weight by least
squares on the other four folds, clipped to [0, 1]); the head proxy of the score change is then compared between
the arms on the same bootstrap resamples of shots.

    python d50_paired.py NN_OOF.npz OUT_DIR RESULT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from d9_gate_eval import FOLDS, RESAMPLES, SEED, pooled_r2, proxy


def main():
    heads_file, out, result_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with np.load(heads_file, allow_pickle=False) as z:
        nn, labels, row_fold, shot = z['nn'].astype(np.float64), z['labels'].astype(np.float64), z['row_fold'], z['shot_of_row']
        targets = z['targets'].astype(str).tolist()
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(p, minlength=shots) for p in rng.integers(0, shots, size=(RESAMPLES, shots))]).astype(float)
    fold_of_shot = np.full(shots, -1)
    fold_of_shot[shot[row_fold >= 0]] = row_fold[row_fold >= 0]
    delta, boot, by_fold, report = {}, {}, {}, {}
    for arm in ('control', 'treatment'):
        gbm = np.full(nn.shape, np.nan)
        for k in range(FOLDS):
            with np.load(out / arm / f'gbm_oof_fold{k}.npz', allow_pickle=False) as z:
                assert z['targets'].astype(str).tolist() == targets
                assert np.array_equal(z['rows'], np.flatnonzero(row_fold == k))
                gbm[z['rows']] = z['pred']
        delta[arm], boot[arm], by_fold[arm], report[arm] = {}, {}, {}, {}
        for j, name in enumerate(targets):
            y, p, g = labels[:, j], nn[:, j], gbm[:, j]
            ok = np.isfinite(y) & (row_fold >= 0)
            assert np.isfinite(p[ok]).all() and np.isfinite(g[ok]).all(), name
            blend, weights = np.full(len(y), np.nan), np.zeros(FOLDS)
            for k in range(FOLDS):
                fit, use = ok & (row_fold != k), ok & (row_fold == k)
                d = g[fit] - p[fit]
                weights[k] = float(np.clip(np.dot(d, y[fit] - p[fit]) / np.dot(d, d), 0.0, 1.0))
                blend[use] = p[use] + weights[k] * (g[use] - p[use])
            s = shot[ok]
            per_shot = lambda v: np.bincount(s, weights=v, minlength=shots)
            n, t, t2 = per_shot(np.ones(ok.sum())), per_shot(y[ok]), per_shot(y[ok] ** 2)
            e_nn, e_blend = per_shot((y[ok] - p[ok]) ** 2), per_shot((y[ok] - blend[ok]) ** 2)
            r2_nn = float(pooled_r2(e_nn.sum(), t.sum(), t2.sum(), n.sum()))
            r2_blend = float(pooled_r2(e_blend.sum(), t.sum(), t2.sum(), n.sum()))
            delta[arm][name] = r2_blend - r2_nn
            boot[arm][name] = (pooled_r2(counts @ e_blend, counts @ t, counts @ t2, counts @ n)
                               - pooled_r2(counts @ e_nn, counts @ t, counts @ t2, counts @ n))
            by_fold[arm][name] = [float(pooled_r2(e_blend[m].sum(), t[m].sum(), t2[m].sum(), n[m].sum())
                                        - pooled_r2(e_nn[m].sum(), t[m].sum(), t2[m].sum(), n[m].sum()))
                                  for m in (fold_of_shot == k for k in range(FOLDS))]
            report[arm][name] = {'r2_nn': r2_nn, 'r2_blend': r2_blend, 'dR2': delta[arm][name],
                                 'error_variance_ratio': float(e_blend.sum() / e_nn.sum()),
                                 'weights_by_fold': weights.tolist()}
    value = {arm: float(proxy(delta[arm])) for arm in delta}
    difference = proxy(boot['treatment']) - proxy(boot['control'])
    paired = value['treatment'] - value['control']
    folds = [float(proxy({n: by_fold['treatment'][n][k] for n in targets if n in by_fold['treatment']})
                   - proxy({n: by_fold['control'][n][k] for n in targets if n in by_fold['control']})) for k in range(FOLDS)]
    per_target = {name: {'dR2_treatment_minus_control': delta['treatment'][name] - delta['control'][name],
                         'P_positive': float(np.mean(boot['treatment'][name] - boot['control'][name] > 0)),
                         'error_variance_ratio_treatment': report['treatment'][name]['error_variance_ratio'],
                         'error_variance_ratio_control': report['control'][name]['error_variance_ratio']}
                  for name in targets}
    result = {'protocol': 'protocol_d50_gate.json', 'dS_proxy': value, 'paired_dS_proxy': paired,
              'paired_ci95': [float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))],
              'P_paired_positive': float(np.mean(difference > 0)), 'paired_by_fold': folds,
              'passes': bool(paired >= 1.5e-5 and np.mean(difference > 0) >= 0.95 and sum(f > 0 for f in folds) >= 4),
              'per_target': per_target, 'arms': report, 'resamples': RESAMPLES, 'seed': SEED}
    result_file.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=1))


if __name__ == '__main__':
    main()
