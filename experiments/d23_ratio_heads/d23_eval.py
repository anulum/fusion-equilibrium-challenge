# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D23-0 gate: evaluation of the ratio heads for q95 and betaN
"""On the held-out frames of the five folds: the networks' direct q95 and betaN heads (arm R), the same quantities
read from the ratio heads (head times ratio), and their nested blend (weight of the ratio estimate fitted by least
squares on the other four folds, clipped to [0, 1]). Reference: arm S of D18-0 (same members without the two extra
labels, fitted on an accelerator). Squared-error ratios with a shot bootstrap (2000, seed 990); the qb term's
change as 0.15 x mean dR2 of q95 and betaN; the D9 head proxy and the flux coordinates of arm R against arm S.
Rule: protocol_d23_0_gate.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25, 'li_scored': 0.6}


def main():
    z = np.load(HERE.parent / 'd9_gbm_heads' / 'nn_oof.npz')
    lab, shot, fold = z['labels'].astype(float), z['shot_of_row'], z['row_fold']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    r, s = np.full((n, len(targets)), np.nan), np.full((n, len(targets)), np.nan)
    cr, cs = np.full(n, np.nan), np.full(n, np.nan)
    from_ratio = {q: np.full(n, np.nan) for q in QB}
    files = sorted(HERE.glob('kaggle_out_*/d23_fold*.npz'))
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            assert f['targets'].astype(str).tolist() == targets
            r[f['rows']] = f['pred_r']; cr[f['rows']] = f['coord_sse_r']
            for q in QB:
                from_ratio[q][f['rows']] = f[f'{q}_from_ratio']
    for path in sorted((HERE.parent / 'd18_subframe_nets').glob('kaggle_out_*/d18_fold*.npz')):
        with np.load(path, allow_pickle=False) as f:
            s[f['rows']] = f['pred_s']; cs[f['rows']] = f['coord_sse_s']
    have = np.isfinite(cr) & np.isfinite(cs)
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    per = lambda v, m: np.bincount(shot[m], weights=v, minlength=shots)

    def versus(new, old, y, ok):
        en, eo = per((new[ok] - y[ok]) ** 2, ok), per((old[ok] - y[ok]) ** 2, ok)
        cnt, t1, t2 = per(np.ones(ok.sum()), ok), per(y[ok], ok), per(y[ok] ** 2, ok)
        var, varb = t2.sum() - t1.sum() ** 2 / cnt.sum(), counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt)
        with np.errstate(invalid='ignore', divide='ignore'):
            b = (counts @ en) / (counts @ eo)
            d = (counts @ eo - counts @ en) / varb
        return {'r2_old': float(1 - eo.sum() / var), 'r2_new': float(1 - en.sum() / var), 'sse_ratio': float(en.sum() / eo.sum()),
                'ci95': [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], 'P_new_better': float(np.nanmean(b < 1.0)),
                'dR2': float((eo.sum() - en.sum()) / var)}, d

    out = {'files': [p.name for p in files], 'frames': int(have.sum()), 'q': {}}
    boot_direct, boot_ref = [], []
    for q in QB:
        j = targets.index(q); y = lab[:, j]
        ok = have & np.isfinite(y) & np.isfinite(from_ratio[q])
        blend, weights = np.full(n, np.nan), []
        for k in range(5):
            tr, te = ok & (fold != k), ok & (fold == k)
            if not te.any():
                continue
            a, b = from_ratio[q][tr] - r[tr, j], y[tr] - r[tr, j]
            w = float(np.clip((a @ b) / (a @ a), 0.0, 1.0)); weights.append(w)
            blend[te] = r[te, j] + w * (from_ratio[q][te] - r[te, j])
        okb = ok & np.isfinite(blend)
        e = {'weights_by_fold': weights}
        e['ratio_head_vs_direct'], _ = versus(from_ratio[q], r[:, j], y, okb)
        e['blend_vs_direct'], d1 = versus(blend, r[:, j], y, okb)
        e['blend_vs_D18_S'], d2 = versus(blend, s[:, j], y, okb)
        e['direct_vs_D18_S'], _ = versus(r[:, j], s[:, j], y, okb)
        out['q'][q] = e; boot_direct.append(d1); boot_ref.append(d2)
    for key, boots, name in (('qb_gain_blend_over_direct', boot_direct, 'blend_vs_direct'), ('qb_gain_blend_over_D18_S', boot_ref, 'blend_vs_D18_S')):
        b = 0.15 * np.mean(boots, axis=0)
        out[key] = {'dS': float(0.15 * np.mean([out['q'][q][name]['dR2'] for q in QB])),
                    'ci95': [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], 'P_positive': float(np.nanmean(b > 0))}
    co, cn = per(cs[have], have), per(cr[have], have)
    b = (counts @ cn) / (counts @ co)
    out['flux_coordinates_R_over_S'] = {'sse_ratio': float(cn.sum() / co.sum()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    other = {}
    for name in SHARES:
        j = targets.index(name); y = lab[:, j]; ok = have & np.isfinite(y)
        other[name], _ = versus(r[:, j], s[:, j], y, ok)
    out['other_heads_R_vs_S'] = {k: {'sse_ratio': v['sse_ratio'], 'dR2': v['dR2']} for k, v in other.items()}
    out['other_heads_proxy_R_minus_S'] = float(0.20 / 7.0 * sum(SHARES[k] * other[k]['dR2'] for k in SHARES))
    complete = len(files) == 5
    out['complete'] = complete
    g = out['qb_gain_blend_over_direct']
    out['passes'] = bool(complete and g['dS'] >= 1.5e-5 and g['P_positive'] >= 0.95 and out['qb_gain_blend_over_D18_S']['dS'] > 0.0
                         and out['flux_coordinates_R_over_S']['sse_ratio'] <= 1.10)
    (HERE / 'd23_result.json').write_text(json.dumps(out, indent=1))
    for q in QB:
        e = out['q'][q]
        print(q, 'weights', [round(w, 3) for w in e['weights_by_fold']])
        for key in ('ratio_head_vs_direct', 'blend_vs_direct', 'blend_vs_D18_S', 'direct_vs_D18_S'):
            v = e[key]
            print(f"   {key:22s} R2 {v['r2_old']:.5f} -> {v['r2_new']:.5f}  SSE ratio {v['sse_ratio']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] P {v['P_new_better']:.3f}")
    print(json.dumps({k: out[k] for k in ('qb_gain_blend_over_direct', 'qb_gain_blend_over_D18_S', 'flux_coordinates_R_over_S', 'other_heads_proxy_R_minus_S', 'complete', 'passes')}, indent=1))


main()
