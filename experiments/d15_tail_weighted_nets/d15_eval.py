# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D15-0: tail-weighted networks blended into the S16 heads, out of fold
"""Apply protocol_d15_0_gate.json.

Variants, all nested (weights fitted on the other four folds, clipped to [0, 1]):
  one      S16 head + w (tail-weighted nets - S16), one weight for all frames
  regime   the same with separate weights inside and outside the tail regime      <- the rule's variant
  trees    the D9-0 blend with boosted trees, one weight (reference, +3.11e-5)
  both     regime blend with the tail-weighted nets, then the D9 trees blended on top with regime weights
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D9 = HERE.parent / 'd9_gbm_heads'
FOLDS = 5
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25, 'li_scored': 0.6}


def load(directories, n, targets):
    out = np.full((n, len(targets)), np.nan)
    for directory in directories:
        for path in sorted(directory.glob('gbm_oof_fold*.npz')):
            with np.load(path, allow_pickle=False) as z:
                names = z['targets'].astype(str).tolist()
                out[np.ix_(z['rows'], [targets.index(t) for t in names])] = z['pred']
    return out


def nested(y, p, g, fold, masks):
    b, weights = p.copy(), []
    for m in masks:
        for k in range(FOLDS):
            fit, use = m & (fold != k), m & (fold == k)
            d = g[fit] - p[fit]
            w = float(np.clip(np.dot(d, y[fit] - p[fit]) / np.dot(d, d), 0.0, 1.0))
            b[use] = p[use] + w * (g[use] - p[use]); weights.append(w)
    return b, weights


def main():
    z = np.load(D9 / 'nn_oof.npz')
    nn, lab, fold, shot = z['nn'].astype(float), z['labels'].astype(float), z['row_fold'], z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    c = np.load(HERE.parent / 'd13_shot_anatomy' / 'columns.npz')
    col = {int(k): c['values'][:, i] for i, k in enumerate(c['cols'])}
    last = np.r_[shot[1:] != shot[:-1], True]
    from_end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1] - np.arange(n)
    tail_all = (from_end < 25) | (col[62] < 0.6) | (col[64] < 0.3)
    nets = load([HERE / 'kaggle_out_a', HERE / 'kaggle_out_b'], n, targets)
    trees = load([D9 / 'kaggle_out'], n, targets)
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    names = ('one', 'regime', 'trees', 'both')
    delta, boot, result = {v: {} for v in names}, {v: {} for v in names}, {}
    for name in targets[:9]:
        j = targets.index(name)
        y, p, g, t = lab[:, j], nn[:, j], nets[:, j], trees[:, j]
        ok = np.isfinite(y) & (fold >= 0)
        assert np.isfinite(g[ok]).all() and np.isfinite(t[ok]).all(), name
        tail, rest = ok & tail_all, ok & ~tail_all
        one, w_one = nested(y, p, g, fold, [ok])
        regime, w_regime = nested(y, p, g, fold, [tail, rest])
        tr, _ = nested(y, p, t, fold, [ok])
        both, _ = nested(y, regime, t, fold, [tail, rest])
        per = lambda v: np.bincount(shot[ok], weights=v, minlength=shots)
        cnt, t1, t2 = per(np.ones(ok.sum())), per(y[ok]), per(y[ok] ** 2)
        r2 = lambda e: 1.0 - e.sum() / (t2.sum() - t1.sum() ** 2 / cnt.sum())
        r2b = lambda e: 1.0 - counts @ e / (counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt))
        e_nn = per((p[ok] - y[ok]) ** 2)
        sse = lambda v, m: float(np.sum((v[m] - y[m]) ** 2))
        entry = {'nn_r2': float(r2(e_nn)), 'nets_alone_r2': float(r2(per((g[ok] - y[ok]) ** 2))),
                 'nets_over_nn_sse_tail': sse(g, tail) / sse(p, tail), 'nets_over_nn_sse_rest': sse(g, rest) / sse(p, rest),
                 'weight_one': float(np.mean(w_one)), 'weight_tail': float(np.mean(w_regime[:FOLDS])),
                 'weight_rest': float(np.mean(w_regime[FOLDS:])),
                 'error_correlation': float(np.corrcoef(p[ok] - y[ok], g[ok] - y[ok])[0, 1])}
        for v, b in zip(names, (one, regime, tr, both)):
            e = per((b[ok] - y[ok]) ** 2)
            entry[v] = {'r2': float(r2(e)), 'error_variance_ratio': float(e.sum() / e_nn.sum())}
            delta[v][name] = r2(e) - r2(e_nn); boot[v][name] = r2b(e) - r2b(e_nn)
        result[name] = entry
    proxy = lambda d: 0.15 * np.mean([d[q] for q in QB], axis=0) + 0.20 / 7.0 * sum(s * d[k] for k, s in SHARES.items())
    summary = {}
    for v in names:
        b = proxy(boot[v])
        summary[v] = {'dS_proxy': float(proxy(delta[v])), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      'P_positive': float(np.mean(b > 0))}
    extra = proxy(boot['both']) - proxy(boot['trees'])
    summary['both_minus_trees'] = {'dS_proxy': float(proxy(delta['both']) - proxy(delta['trees'])),
                                   'ci95': [float(np.percentile(extra, 2.5)), float(np.percentile(extra, 97.5))],
                                   'P_positive': float(np.mean(extra > 0))}
    summary['passes'] = bool(summary['regime']['dS_proxy'] >= 3e-5 and summary['regime']['P_positive'] >= 0.95)
    (HERE / 'd15_result.json').write_text(json.dumps({'summary': summary, 'targets': result}, indent=1))
    print(json.dumps(summary, indent=1))
    for name, e in result.items():
        print(f"{name:10s} NN {e['nn_r2']:.5f} nets alone {e['nets_alone_r2']:.5f} (SSE ratio tail {e['nets_over_nn_sse_tail']:.2f} rest "
              f"{e['nets_over_nn_sse_rest']:.2f}, corr {e['error_correlation']:.2f}) | one {e['one']['r2']:.5f} w {e['weight_one']:.2f} | regime "
              f"{e['regime']['r2']:.5f} w tail {e['weight_tail']:.2f} rest {e['weight_rest']:.2f} | trees {e['trees']['r2']:.5f} | both {e['both']['r2']:.5f}")


main()
