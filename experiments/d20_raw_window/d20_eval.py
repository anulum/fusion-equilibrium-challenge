# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D20-0 gate: evaluation of the raw-window arm against the sub-frame arm
"""Arm W (table + sub-frame + raw 0.5 ms windows, fitted on the host) against arm S (table + sub-frame; D18-0,
fitted on an accelerator), on the held-out frames of the five folds; and the control on fold 0, where arm S was
fitted again on the host (arm C), so that the part of any difference that is due to the machine is measured.

Per head the pooled squared-error ratio with a shot bootstrap (2000, seed 990), the same for the 64 flux
coordinates, and the D9 head proxy of the difference. Rule: protocol_d20_0_gate.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25, 'li_scored': 0.6}


def compared(new, old, where, lab, shot, targets, counts):
    """Squared-error ratios new/old and the head proxy new minus old on the frames of ``where``."""
    shots = counts.shape[1]
    per = lambda v, m: np.bincount(shot[m], weights=v, minlength=shots)
    out, delta, boot = {'frames': int(where.sum()), 'heads': {}}, {}, {}
    for j, name in enumerate(targets):
        y = lab[:, j]
        ok = where & np.isfinite(y)
        eo, en = per((old[0][ok, j] - y[ok]) ** 2, ok), per((new[0][ok, j] - y[ok]) ** 2, ok)
        cnt, t1, t2 = per(np.ones(ok.sum()), ok), per(y[ok], ok), per(y[ok] ** 2, ok)
        var, varb = t2.sum() - t1.sum() ** 2 / cnt.sum(), counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = (counts @ en) / (counts @ eo)
            boot[name] = (counts @ eo - counts @ en) / varb
        out['heads'][name] = {'r2_old': float(1 - eo.sum() / var), 'r2_new': float(1 - en.sum() / var),
                              'sse_ratio': float(en.sum() / eo.sum()),
                              'ci95': [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))],
                              'P_new_better': float(np.nanmean(ratio < 1.0))}
        delta[name] = (eo.sum() - en.sum()) / var
    co, cn = per(old[1][where], where), per(new[1][where], where)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (counts @ cn) / (counts @ co)
    out['flux_coordinates'] = {'sse_ratio': float(cn.sum() / co.sum()),
                               'ci95': [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))],
                               'P_new_better': float(np.nanmean(r < 1.0))}
    proxy = lambda d: 0.15 * np.mean([d[q] for q in QB], axis=0) + 0.20 / 7.0 * sum(s * d[k] for k, s in SHARES.items())
    b = proxy(boot)
    out['head_proxy'] = {'dS_proxy': float(proxy(delta)), 'ci95': [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))],
                         'P_positive': float(np.nanmean(b > 0))}
    return out


def main():
    z = np.load(HERE.parent / 'd9_gbm_heads' / 'nn_oof.npz')
    lab, shot = z['labels'].astype(float), z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    arm = {a: (np.full((n, len(targets)), np.nan), np.full(n, np.nan)) for a in 'swc'}
    sources = {'s': (HERE.parent / 'd18_subframe_nets', 'kaggle_out_*/d18_fold*.npz', 's'),
               'w': (HERE, 'kaggle_out_*/d20_w_fold*.npz', 'w'), 'c': (HERE, 'kaggle_out_*/d20_s_fold*.npz', 's')}
    files = {}
    for a, (root, pattern, key) in sources.items():
        files[a] = sorted(root.glob(pattern))
        for path in files[a]:
            with np.load(path, allow_pickle=False) as f:
                assert f['targets'].astype(str).tolist() == targets
                arm[a][0][f['rows']] = f[f'pred_{key}']; arm[a][1][f['rows']] = f[f'coord_sse_{key}']
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    have = {a: np.isfinite(arm[a][1]) for a in 'swc'}
    out = {'files': {a: [p.name for p in files[a]] for a in 'swc'}}
    out['w_vs_s'] = compared(arm['w'], arm['s'], have['w'] & have['s'], lab, shot, targets, counts)
    control = have['c'] & have['s'] & have['w']
    out['control_c_vs_s'] = compared(arm['c'], arm['s'], control, lab, shot, targets, counts)
    out['control_w_vs_c'] = compared(arm['w'], arm['c'], control, lab, shot, targets, counts)
    main_, machine, same = out['w_vs_s'], out['control_c_vs_s']['head_proxy']['dS_proxy'], out['control_w_vs_c']['head_proxy']['dS_proxy']
    complete = len(files['w']) == 5 and len(files['c']) == 1
    out['complete'] = complete
    out['passes'] = bool(complete and main_['head_proxy']['P_positive'] >= 0.95
                         and main_['flux_coordinates']['sse_ratio'] <= 1.0
                         and main_['head_proxy']['dS_proxy'] >= 2.0 * abs(machine) and same > 0.0)
    (HERE / 'd20_result.json').write_text(json.dumps(out, indent=1))
    for key in ('w_vs_s', 'control_c_vs_s', 'control_w_vs_c'):
        e = out[key]
        print(key, json.dumps({'frames': e['frames'], 'flux_coordinates': e['flux_coordinates'], 'head_proxy': e['head_proxy']}))
        for name, h in e['heads'].items():
            print(f"  {name:10s} old {h['r2_old']:.5f} new {h['r2_new']:.5f} SSE ratio {h['sse_ratio']:.4f} CI [{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}] P(new better) {h['P_new_better']:.3f}")
    print(json.dumps({'complete': complete, 'passes': out['passes']}))


main()
