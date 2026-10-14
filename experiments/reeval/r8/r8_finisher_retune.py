# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — R8: finisher re-tuning on the 400 OOF shots with nested fold selection (protocol 3d11add8)
"""Build every variant on every OOF fold (the fold's own 16 networks), score every 2nd frame with the organisers'
reading (reeval/r2_stage_attribution.score_shot), then select by nested cross-validation over the folds."""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = HERE.parent.parent / 'kfold'
sys.path.insert(0, str(K))
sys.path.insert(0, str(HERE.parent))
import kfold_build as kb  # noqa: E402
import r2_stage_attribution as r2  # noqa: E402

OUT = HERE / 'results'


def variants(v25):
    b = dict(v25)
    tri = ('tri_top', 0.25), ('tri_bot', 0.25)
    return {
        'V0': b,
        'V1': dict(b, li_share=0.4), 'V2': dict(b, li_share=0.8),
        'V3': dict(b, geometric_share=0.35), 'V4': dict(b, geometric_share=0.65),
        'V5': dict(b, scalar_shares=(('kappa', 0.35), ('volume', 0.35), *tri)),
        'V6': dict(b, scalar_shares=(('kappa', 0.65), ('volume', 0.65), *tri)),
        'V7': dict(b, contour_share=0.35), 'V8': dict(b, contour_share=0.65),
        'V9': dict(b, kept_coordinates=64),
    }


def sums(rows, shots, tmean):
    """Per-shot additive sums from which r2.pooled's S is rebuilt (a missing prediction takes the pooled truth mean)."""
    idx = {s: i for i, s in enumerate(shots)}
    n = len(shots)
    z = {k: np.zeros(n) for k in ('n', 'sse', 'tsum', 'tsq', 'D', 'frames')}
    for k in ('pe', 'ts', 'tq', 'cnt'):
        z[k] = np.zeros((n, 7))
    for k in ('qe', 'qt', 'qq', 'qc'):
        z[k] = np.zeros((n, 2))
    for r in rows:
        i = idx[r['shot']]
        z['n'][i] += r['n']; z['sse'][i] += r['sse']; z['tsum'][i] += r['tsum']; z['tsq'][i] += r['tsq']
        z['D'][i] += r['D']; z['frames'][i] += 1
        t = np.asarray(r['t'], float); p = np.asarray(r['p'], float)
        ok = np.isfinite(t)
        p = np.where(np.isfinite(p), p, tmean)
        z['pe'][i][ok] += (p[ok] - t[ok]) ** 2; z['ts'][i][ok] += t[ok]; z['tq'][i][ok] += t[ok] ** 2; z['cnt'][i][ok] += 1
        q = np.asarray(r['q'], float)
        for c, j in enumerate((0, 2)):
            if np.isfinite(q[j]) and np.isfinite(q[j + 1]):
                z['qe'][i, c] += (q[j] - q[j + 1]) ** 2; z['qt'][i, c] += q[j + 1]; z['qq'][i, c] += q[j + 1] ** 2; z['qc'][i, c] += 1
    return z


def score_from(z, w):
    """Pooled S from per-shot sums weighted by bootstrap counts w."""
    s = {k: np.tensordot(w, v, axes=(0, 0)) for k, v in z.items()}
    m = s['tsum'] / s['n']
    flux = 1 - s['sse'] / (s['tsq'] - s['n'] * m * m)
    var = s['tq'] - s['ts'] ** 2 / s['cnt']
    cons = np.mean(np.maximum(0, 1 - s['pe'] / var))
    qvar = s['qq'] - s['qt'] ** 2 / s['qc']
    qb = np.mean(np.maximum(0, 1 - s['qe'] / qvar))
    d = s['D'] / s['frames']
    return 0.55 * max(0, flux) + 0.15 * qb + 0.10 * (1 - min(1, d)) + 0.20 * cons


def bootstrap(nested, base, shots, tmean, count):
    zn, zb = sums(nested, shots, tmean), sums(base, shots, tmean)
    rng = np.random.default_rng(930)
    out = np.empty(count)
    for b in range(count):
        w = np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)).astype(float)
        out[b] = score_from(zn, w) - score_from(zb, w)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, v25 = kb.v25_finish()
    rows = {}   # (variant, fold) -> list of per-frame rows with shot index
    for fold in range(5):
        receipt = json.loads((K / f'oof_fold{fold}' / 'receipt.json').read_text())
        stems, nets = receipt['stems'], [Path(p) for p in receipt['networks']]
        paths = [kb.DATA / f'{s}.parquet' for s in stems]
        for name, values in variants(v25).items():
            cache = OUT / f'rows_{name}_fold{fold}.json'
            if cache.is_file():
                rows[(name, fold)] = json.loads(cache.read_text())
                continue
            archive = K / f'oof_fold{fold}' / 'diii_d_train.npz' if name == 'V0' else OUT / f'{name}_fold{fold}.npz'
            if not archive.is_file():
                kb.vbuild.predicted_archive(paths, nets, kb.BASIS, 3, kb.vbuild.Finish(**values)).write(archive)
            with Pool(3) as pool:
                parts = pool.map(r2.score_shot, [(str(archive), i, s) for i, s in enumerate(stems)])
            rr = [dict(r, shot=f'{fold}:{i}') for i, part in enumerate(parts) for r in part]
            cache.write_text(json.dumps(rr))
            rows[(name, fold)] = rr
            print(json.dumps({'variant': name, 'fold': fold, 'S': r2.pooled(rr)['S']}), flush=True)
    names = list(variants(v25))
    per_fold = {(n, f): r2.pooled(rows[(n, f)])['S'] for n in names for f in range(5)}
    chosen = {}
    for j in range(5):
        train = {n: r2.pooled([r for f in range(5) if f != j for r in rows[(n, f)]])['S'] for n in names}
        chosen[j] = max(train, key=train.get)
    nested = [r for j in range(5) for r in rows[(chosen[j], j)]]
    base = [r for j in range(5) for r in rows[('V0', j)]]
    ds = r2.pooled(nested)['S'] - r2.pooled(base)['S']
    # shot bootstrap of the nested difference (20000, seed 930) on additive per-shot sums of the pooled score
    shots = sorted({r['shot'] for r in base})
    tmean = np.nanmean(np.asarray([r['t'] for r in base], float), axis=0)
    check = {'pooled_V0': r2.pooled(base)['S'], 'sums_V0': float(score_from(sums(base, shots, tmean), np.ones(len(shots))))}
    assert abs(check['pooled_V0'] - check['sums_V0']) < 1e-6, check
    boots = bootstrap(nested, base, shots, tmean, 20000)
    overall = {n: r2.pooled([r for f in range(5) for r in rows[(n, f)]])['S'] for n in names}
    result = {'per_variant_pooled_S_all400': overall, 'chosen_per_fold': chosen, 'nested_dS': ds,
              'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'P_positive': float((boots > 0).mean()),
              'bootstrap_resamples': len(boots), 'sum_check': check, 'chosen_on_all400': max(overall, key=overall.get)}
    result['accepted'] = bool(ds >= 2e-5 and result['P_positive'] >= 0.9)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
