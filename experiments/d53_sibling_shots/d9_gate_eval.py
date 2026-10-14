# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D9-0 gate: network heads, boosted heads and their nested blend, out of fold
"""Apply the frozen rule of protocol_d9_0_gate.json.

    python d9_gate_eval.py --nn nn_oof.npz --gbm DIR_WITH_gbm_oof_fold{k}.npz --out result.json

Per target: pooled R2 of the network heads, the boosted heads and the blend NN + w (GBM - NN), with w fitted by
least squares on the other four folds and clipped to [0, 1]. The proxy of the score change and its bootstrap over
shots are the protocol's.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FOLDS, RESAMPLES, SEED = 5, 2000, 990
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25,
          'li_scored': 0.6}


def pooled_r2(sse, total, total_sq, count):
    """R2 from sums: residual squares, label sum, label square sum and frame count."""
    return 1.0 - sse / (total_sq - total ** 2 / count)


def proxy(delta):
    """The protocol's head-level proxy of the score change from each target's dR2."""
    return (0.15 * np.mean([delta[n] for n in QB], axis=0)
            + 0.20 / 7.0 * sum(share * delta[name] for name, share in SHARES.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nn', type=Path, required=True)
    ap.add_argument('--gbm', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.nn, allow_pickle=False) as z:
        nn, labels, row_fold, shot = z['nn'].astype(np.float64), z['labels'].astype(np.float64), z['row_fold'], z['shot_of_row']
        targets = z['targets'].astype(str).tolist()
    gbm = np.full(nn.shape, np.nan)
    for k in range(FOLDS):
        with np.load(a.gbm / f'gbm_oof_fold{k}.npz', allow_pickle=False) as z:
            assert z['targets'].astype(str).tolist() == targets
            assert np.array_equal(z['rows'], np.flatnonzero(row_fold == k))
            gbm[z['rows']] = z['pred']
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, shots, size=(RESAMPLES, shots))
    counts = np.stack([np.bincount(p, minlength=shots) for p in picks]).astype(np.float64)
    report, delta, delta_boot, delta_fold = {}, {}, {}, {}
    for j, name in enumerate(targets):
        y, p, g = labels[:, j], nn[:, j], gbm[:, j]
        ok = np.isfinite(y) & (row_fold >= 0)
        assert np.isfinite(p[ok]).all() and np.isfinite(g[ok]).all(), name
        weights = np.zeros(FOLDS)
        blend = np.full(len(y), np.nan)
        for k in range(FOLDS):
            fit, use = ok & (row_fold != k), ok & (row_fold == k)
            d = g[fit] - p[fit]
            weights[k] = float(np.clip(np.dot(d, y[fit] - p[fit]) / np.dot(d, d), 0.0, 1.0))
            blend[use] = p[use] + weights[k] * (g[use] - p[use])
        s = shot[ok]
        per_shot = lambda v: np.bincount(s, weights=v, minlength=shots)
        n, t, t2 = per_shot(np.ones(ok.sum())), per_shot(y[ok]), per_shot(y[ok] ** 2)
        e = {'nn': per_shot((y[ok] - p[ok]) ** 2), 'gbm': per_shot((y[ok] - g[ok]) ** 2),
             'blend': per_shot((y[ok] - blend[ok]) ** 2)}
        r2 = {key: float(pooled_r2(v.sum(), t.sum(), t2.sum(), n.sum())) for key, v in e.items()}
        boot = {key: pooled_r2(counts @ v, counts @ t, counts @ t2, counts @ n) for key, v in e.items()}
        delta[name], delta_boot[name] = r2['blend'] - r2['nn'], boot['blend'] - boot['nn']
        fold_of_shot = np.full(shots, -1)
        fold_of_shot[shot[row_fold >= 0]] = row_fold[row_fold >= 0]
        delta_fold[name] = []
        for k in range(FOLDS):
            m = fold_of_shot == k
            delta_fold[name].append(float(pooled_r2(e['blend'][m].sum(), t[m].sum(), t2[m].sum(), n[m].sum())
                                          - pooled_r2(e['nn'][m].sum(), t[m].sum(), t2[m].sum(), n[m].sum())))
        report[name] = {'frames': int(ok.sum()), 'r2': r2, 'dR2_blend_minus_nn': delta[name],
                        'error_variance_ratio_blend_over_nn': float(e['blend'].sum() / e['nn'].sum()),
                        'error_correlation_nn_gbm': float(np.corrcoef(y[ok] - p[ok], y[ok] - g[ok])[0, 1]),
                        'weights_by_fold': weights.tolist(),
                        'P_dR2_positive': float(np.mean(delta_boot[name] > 0))}
    x_points = {}
    for side in ('lower', 'upper'):
        jr, jz = targets.index(f'x_{side}_r'), targets.index(f'x_{side}_z')
        ok = np.isfinite(labels[:, jr]) & np.isfinite(labels[:, jz]) & (row_fold >= 0)
        out = {}
        for key, source in (('nn', nn), ('gbm', gbm)):
            out[key] = float(np.mean(np.hypot(source[ok, jr] - labels[ok, jr], source[ok, jz] - labels[ok, jz])) * 1e3)
        mean = 0.5 * (nn + gbm)
        out['equal_mean'] = float(np.mean(np.hypot(mean[ok, jr] - labels[ok, jr], mean[ok, jz] - labels[ok, jz])) * 1e3)
        x_points[side] = dict(out, frames=int(ok.sum()), unit='mm, mean Euclidean error')
    value, boot = float(proxy(delta)), proxy(delta_boot)
    by_fold = [float(proxy({name: delta_fold[name][k] for name in delta_fold})) for k in range(FOLDS)]
    result = {'protocol': 'protocol_d9_0_gate.json', 'shots': int(np.unique(shot[row_fold >= 0]).size),
              'dS_proxy': value, 'dS_proxy_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              'P_dS_proxy_positive': float(np.mean(boot > 0)), 'dS_proxy_by_fold': by_fold,
              'passes': bool(value >= 3e-5 and np.mean(boot > 0) >= 0.95), 'targets': report, 'x_points': x_points,
              'resamples': RESAMPLES, 'seed': SEED}
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
