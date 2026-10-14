# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D18-0: paired comparison of the networks with (S) and without (P) sub-frame inputs
"""Apply protocol_d18_0_gate.json: per head the pooled squared error of arm S over arm P on all held-out frames,
with a bootstrap over shots (2000, seed 990); the same for the flux coordinates; and the head proxy of the change."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = ('q95', 'beta_n')
SHARES = {'r_axis': 1.0, 'z_axis': 1.0, 'kappa': 0.5, 'volume': 0.5, 'tri_top': 0.25, 'tri_bot': 0.25, 'li_scored': 0.6}


def main():
    z = np.load(Path('$FEC_WORK/first_agent/d9_gbm_heads/nn_oof.npz'))
    lab, shot = z['labels'].astype(float), z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    n = len(shot)
    pred = {a: np.full((n, len(targets)), np.nan) for a in 'ps'}
    csse = {a: np.full(n, np.nan) for a in 'ps'}
    folds = []
    for path in sorted(HERE.glob('outputs_f*/d26_fold*.npz')):
        with np.load(path, allow_pickle=False) as f:
            assert f['targets'].astype(str).tolist() == targets
            for a in 'ps':
                pred[a][f['rows']] = f[f'pred_{a}']; csse[a][f['rows']] = f[f'coord_sse_{a}']
        folds.append(path.stem)
    assert len(folds) == 5
    have = np.isfinite(csse['p'])
    assert np.array_equal(have, z['row_fold'] >= 0)
    assert np.isfinite(pred['p'][have]).all() and np.isfinite(pred['s'][have]).all()
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)
    per = lambda v, m: np.bincount(shot[m], weights=v, minlength=shots)
    out, delta, boot = {'folds': folds, 'frames': int(have.sum()), 'heads': {}}, {}, {}
    for j, name in enumerate(targets):
        y = lab[:, j]
        ok = have & np.isfinite(y)
        ep, es = per((pred['p'][ok, j] - y[ok]) ** 2, ok), per((pred['s'][ok, j] - y[ok]) ** 2, ok)
        cnt, t1, t2 = per(np.ones(ok.sum()), ok), per(y[ok], ok), per(y[ok] ** 2, ok)
        var, varb = t2.sum() - t1.sum() ** 2 / cnt.sum(), counts @ t2 - (counts @ t1) ** 2 / (counts @ cnt)
        ratio = (counts @ es) / (counts @ ep)
        out['heads'][name] = {'r2_p': float(1 - ep.sum() / var), 'r2_s': float(1 - es.sum() / var),
                              'sse_ratio_s_over_p': float(es.sum() / ep.sum()),
                              'ci95': [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
                              'P_s_better': float(np.mean(ratio < 1.0))}
        delta[name] = (ep.sum() - es.sum()) / var; boot[name] = (counts @ ep - counts @ es) / varb
    cp, cs = per(csse['p'][have], have), per(csse['s'][have], have)
    r = (counts @ cs) / (counts @ cp)
    out['flux_coordinates'] = {'sse_ratio_s_over_p': float(cs.sum() / cp.sum()),
                               'ci95': [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], 'P_s_better': float(np.mean(r < 1.0))}
    proxy = lambda d: 0.15 * np.mean([d[q] for q in QB], axis=0) + 0.20 / 7.0 * sum(s * d[k] for k, s in SHARES.items())
    b = proxy(boot)
    out['head_proxy_s_minus_p'] = {'dS_proxy': float(proxy(delta)), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                   'P_positive': float(np.mean(b > 0))}
    zr = out['heads']['z_axis']
    out['interpretation'] = 'Raw paired hc two-member diagnostic only; geometry readout, four-member ensembles and full-scored gates still required. No admission from this result.'
    (HERE / 'raw_paired_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'heads'}, indent=1))
    for name, e in out['heads'].items():
        print(f"{name:10s} P {e['r2_p']:.5f} S {e['r2_s']:.5f} SSE ratio {e['sse_ratio_s_over_p']:.4f} CI [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}] P(S better) {e['P_s_better']:.3f}")


main()
