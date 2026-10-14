# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D24-0 gate: evaluation of the sub-frame tree heads against the table tree heads
"""For each of the eight heads, on all held-out frames: the network head (arm S of D18-0, four networks with
sub-frame inputs), the D9 table trees and the new sub-frame trees; each tree head blended into the network head
with a nested weight (least squares on the other four folds, clipped to [0, 1]). Reported: tree R2, blend R2,
weight, the squared-error ratio of the sub-frame blend against the table blend with a shot bootstrap (2000, seed
990), and the D9 head proxy of both blends over the network heads. Rule: protocol_d24_0_gate.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'li_scored': 0.6}
HEADS = ['q95', 'beta_n', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'x_lower_r', 'x_lower_z']


def main():
    z = np.load(HERE.parent / 'd9_gbm_heads' / 'nn_oof.npz')
    lab, shot, fold = z['labels'].astype(float), z['shot_of_row'], z['row_fold']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    net = np.full((n, len(targets)), np.nan)
    for path in sorted((HERE.parent / 'd18_subframe_nets').glob('kaggle_out_*/d18_fold*.npz')):
        with np.load(path, allow_pickle=False) as f:
            net[f['rows']] = f['pred_s']
    tab, sub = np.full((n, len(targets)), np.nan), np.full((n, len(targets)), np.nan)
    for path in sorted((HERE.parent / 'd9_gbm_heads' / 'kaggle_out').glob('gbm_oof_fold*.npz')):
        with np.load(path, allow_pickle=False) as f:
            names = f['targets'].astype(str).tolist()
            for name in HEADS:
                tab[f['rows'], targets.index(name)] = f['pred'][:, names.index(name)]
    files = sorted(HERE.glob('kaggle_out_*/gbm_sub_oof_fold*.npz'))
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            names = f['targets'].astype(str).tolist()
            for name in names[:int(f['done'])]:
                sub[f['rows'], targets.index(name)] = f['pred'][:, names.index(name)]
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    per = lambda v, m: np.bincount(shot[m], weights=v, minlength=shots)
    out, d_sub, d_tab, b_sub, b_tab = {'files': [p.name for p in files], 'heads': {}}, {}, {}, {}, {}
    for name in HEADS:
        j = targets.index(name); y = lab[:, j]
        ok = np.isfinite(y) & np.isfinite(net[:, j]) & np.isfinite(tab[:, j]) & np.isfinite(sub[:, j]) & (fold >= 0)
        if not ok.any():
            continue
        blends, weights = {}, {}
        for key, tree in (('tab', tab), ('sub', sub)):
            blend, ws = np.full(n, np.nan), []
            for k in range(5):
                tr, te = ok & (fold != k), ok & (fold == k)
                a, b = tree[tr, j] - net[tr, j], y[tr] - net[tr, j]
                w = float(np.clip((a @ b) / (a @ a), 0.0, 1.0)); ws.append(w)
                blend[te] = net[te, j] + w * (tree[te, j] - net[te, j])
            blends[key], weights[key] = blend, ws
        cnt, t1, t2 = per(np.ones(ok.sum()), ok), per(y[ok], ok), per(y[ok] ** 2, ok)
        var, varb = t2.sum() - t1.sum() ** 2 / cnt.sum(), counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt)
        e = {k: per((v[ok] - y[ok]) ** 2, ok) for k, v in (('net', net[:, j]), ('tree_tab', tab[:, j]), ('tree_sub', sub[:, j]),
                                                           ('blend_tab', blends['tab']), ('blend_sub', blends['sub']))}
        ratio = (counts @ e['blend_sub']) / (counts @ e['blend_tab'])
        out['heads'][name] = {'frames': int(ok.sum()), 'r2': {k: float(1 - v.sum() / var) for k, v in e.items()},
                              'weights_tab': weights['tab'], 'weights_sub': weights['sub'],
                              'blend_sub_over_blend_tab': {'sse_ratio': float(e['blend_sub'].sum() / e['blend_tab'].sum()),
                                                           'ci95': [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
                                                           'P_sub_better': float(np.mean(ratio < 1.0))},
                              'tree_sub_over_tree_tab': float(e['tree_sub'].sum() / e['tree_tab'].sum())}
        d_sub[name], d_tab[name] = (e['net'].sum() - e['blend_sub'].sum()) / var, (e['net'].sum() - e['blend_tab'].sum()) / var
        b_sub[name], b_tab[name] = (counts @ e['net'] - counts @ e['blend_sub']) / varb, (counts @ e['net'] - counts @ e['blend_tab']) / varb
    def proxy(d):
        qb = [d[q] for q in QB if q in d]
        return (0.15 * np.mean(qb, axis=0) if qb else 0.0) + 0.20 / 7.0 * sum(s * d[k] for k, s in SHARES.items() if k in d)
    for key, d, b in (('proxy_blend_sub_over_net', d_sub, b_sub), ('proxy_blend_tab_over_net', d_tab, b_tab)):
        bb = proxy(b)
        out[key] = {'dS': float(proxy(d)), 'ci95': [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], 'P_positive': float(np.mean(bb > 0))}
    diff = proxy(b_sub) - proxy(b_tab)
    out['proxy_sub_minus_tab'] = {'dS': float(proxy(d_sub) - proxy(d_tab)), 'ci95': [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                  'P_positive': float(np.mean(diff > 0))}
    out['complete'] = len(out['heads']) == len(HEADS) and len(files) == 10
    out['passes'] = bool(out['complete'] and out['proxy_sub_minus_tab']['dS'] >= 1.0e-5 and out['proxy_sub_minus_tab']['P_positive'] >= 0.95)
    (HERE / 'd24_result.json').write_text(json.dumps(out, indent=1))
    for name, h in out['heads'].items():
        r = h['r2']; q = h['blend_sub_over_blend_tab']
        print(f"{name:10s} net {r['net']:.5f} tree tab {r['tree_tab']:.5f} sub {r['tree_sub']:.5f} | blend tab {r['blend_tab']:.5f} sub {r['blend_sub']:.5f} | w tab {np.mean(h['weights_tab']):.3f} sub {np.mean(h['weights_sub']):.3f} | sub/tab {q['sse_ratio']:.4f} [{q['ci95'][0]:.4f}, {q['ci95'][1]:.4f}] P {q['P_sub_better']:.3f}")
    print(json.dumps({k: out[k] for k in ('proxy_blend_tab_over_net', 'proxy_blend_sub_over_net', 'proxy_sub_minus_tab', 'complete', 'passes')}, indent=1))


main()
