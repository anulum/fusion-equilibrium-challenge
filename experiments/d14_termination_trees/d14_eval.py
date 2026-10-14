# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D14-0: tail-regime trees against the general trees, nested blend with the network heads
"""Apply protocol_d14_0_gate.json: per target the nested blend inside the tail regime, and the head-level proxy.

Variants compared on all 5936 shots (frames outside the tail keep the network head unless stated):
  general   D9 trees A, one nested weight for all frames (the D9-0 result)
  regime    D9 trees A, separate nested weights for tail and non-tail frames
  special   tail-regime trees (this run) in the tail with a nested weight; D9 trees A outside with their own weight
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D9 = HERE.parent / 'd9_gbm_heads'
FOLDS = 5
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25, 'li_scored': 0.6}


def load_trees(directory, n, targets):
    out = {t: np.full(n, np.nan) for t in targets}
    for k in range(FOLDS):
        with np.load(directory / f'gbm_oof_fold{k}.npz', allow_pickle=False) as z:
            names = z['targets'].astype(str).tolist()
            for t in names:
                if t in out:
                    out[t][z['rows']] = z['pred'][:, names.index(t)]
    return out


def nested(y, p, g, fold, masks):
    """Blend p + w (g - p) with w fitted on the other folds, separately inside each mask; other frames keep p."""
    b = p.copy()
    for m in masks:
        for k in range(FOLDS):
            fit, use = m & (fold != k) & (fold >= 0), m & (fold == k)
            d = g[fit] - p[fit]
            w = float(np.clip(np.dot(d, y[fit] - p[fit]) / np.dot(d, d), 0.0, 1.0)) if fit.any() else 0.0
            b[use] = p[use] + w * (g[use] - p[use])
    return b


def main():
    z = np.load(D9 / 'nn_oof.npz')
    nn, lab, fold, shot = z['nn'].astype(float), z['labels'].astype(float), z['row_fold'], z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    general = load_trees(D9 / 'kaggle_out', n, targets)
    special = load_trees(HERE / 'kaggle_out', n, targets)
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    result, delta, boot = {}, {v: {} for v in ('general', 'regime', 'special')}, {v: {} for v in ('general', 'regime', 'special')}
    for name in targets[:9]:
        j = targets.index(name)
        y, p, g = lab[:, j], nn[:, j], general[name]
        ok = np.isfinite(y) & (fold >= 0)
        tail = np.isfinite(special[name]) if name in special and np.isfinite(special[name]).any() else None
        variants = {'general': nested(y, p, g, fold, [ok])}
        if tail is None:
            variants['regime'] = variants['special'] = variants['general']
            tail_share = None
        else:
            tail &= ok
            variants['regime'] = nested(y, p, g, fold, [tail, ok & ~tail])
            s = nested(y, p, np.where(tail, special[name], p), fold, [tail])
            rest = nested(y, p, g, fold, [ok & ~tail])
            variants['special'] = np.where(tail, s, rest)
            tail_share = float(np.sum((p[tail] - y[tail]) ** 2) / np.sum((p[ok] - y[ok]) ** 2))
        per = lambda v: np.bincount(shot[ok], weights=v, minlength=shots)
        cnt, t1, t2 = per(np.ones(ok.sum())), per(y[ok]), per(y[ok] ** 2)
        r2 = lambda e: 1.0 - e.sum() / (t2.sum() - t1.sum() ** 2 / cnt.sum())
        r2b = lambda e: 1.0 - counts @ e / (counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt))
        e_nn = per((p[ok] - y[ok]) ** 2)
        entry = {'nn_r2': float(r2(e_nn)), 'tail_share_of_nn_sse': tail_share}
        for v, b in variants.items():
            e = per((b[ok] - y[ok]) ** 2)
            entry[v] = {'r2': float(r2(e)), 'error_variance_ratio': float(e.sum() / e_nn.sum())}
            delta[v][name] = r2(e) - r2(e_nn); boot[v][name] = r2b(e) - r2b(e_nn)
        if tail is not None:
            entry['tail_error_variance_ratio'] = {
                v: float(np.sum((variants[v][tail] - y[tail]) ** 2) / np.sum((p[tail] - y[tail]) ** 2)) for v in variants}
            entry['special_tree_over_nn_sse_in_tail'] = float(np.sum((special[name][tail] - y[tail]) ** 2) / np.sum((p[tail] - y[tail]) ** 2))
        result[name] = entry
    proxy = lambda d: 0.15 * np.mean([d[q] for q in QB], axis=0) + 0.20 / 7.0 * sum(s * d[k] for k, s in SHARES.items())
    summary = {}
    for v in delta:
        b = proxy(boot[v])
        summary[v] = {'dS_proxy': float(proxy(delta[v])), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      'P_positive': float(np.mean(b > 0))}
    diff = proxy(boot['special']) - proxy(boot['general'])
    summary['special_minus_general'] = {'dS_proxy': float(proxy(delta['special']) - proxy(delta['general'])),
                                        'ci95': [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                        'P_positive': float(np.mean(diff > 0))}
    summary['passes'] = bool(summary['special_minus_general']['dS_proxy'] >= 1e-5 and summary['special_minus_general']['P_positive'] >= 0.95)
    (HERE / 'd14_result.json').write_text(json.dumps({'summary': summary, 'targets': result}, indent=1))
    print(json.dumps(summary, indent=1))
    for name, e in result.items():
        t = e.get('tail_error_variance_ratio')
        print(f"{name:10s} NN {e['nn_r2']:.5f} general {e['general']['r2']:.5f} regime {e['regime']['r2']:.5f} special {e['special']['r2']:.5f}"
              + (f" | tail holds {e['tail_share_of_nn_sse']:.2f} of SSE; in tail: tree/NN {e['special_tree_over_nn_sse_in_tail']:.2f}, blends general {t['general']:.3f} regime {t['regime']:.3f} special {t['special']:.3f}" if t else ''))


main()
