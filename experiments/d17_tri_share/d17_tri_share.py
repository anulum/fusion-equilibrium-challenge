# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D17: triangularity (and elongation) shares of the finisher on the 400 OOF shots
"""Build the frozen variants on every OOF fold with the fold's own 16 networks and score them as R8 did
(protocol_d17_tri_share.json). Single stage on all 400 shots; paired shot bootstrap against the R8 V0 rows.

    PYTHONPATH=R/fec_adapter python d17_tri_share.py [workers]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R8 = HERE.parent / 'reeval' / 'r8'
K = HERE.parent / 'kfold'
for p in (K, HERE.parent / 'reeval', R8):
    sys.path.insert(0, str(p))
import kfold_build as kb  # noqa: E402
import r2_stage_attribution as r2  # noqa: E402
from r8_finisher_retune import score_from, sums  # noqa: E402

OUT = HERE / 'results'
VARIANTS = {
    'T1': (('kappa', 0.5), ('volume', 0.5), ('tri_top', 0.7), ('tri_bot', 0.7)),
    'T2': (('kappa', 0.75), ('volume', 0.5), ('tri_top', 0.7), ('tri_bot', 0.7)),
}
NAMES = ['R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li']


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    _, v25 = kb.v25_finish()
    assert tuple(tuple(x) for x in v25['scalar_shares']) == (('kappa', 0.5), ('volume', 0.5), ('tri_top', 0.25), ('tri_bot', 0.25)), v25['scalar_shares']
    rows = {name: [] for name in VARIANTS}
    base = []
    for fold in range(5):
        receipt = json.loads((K / f'oof_fold{fold}' / 'receipt.json').read_text())
        stems, nets = receipt['stems'], [Path(p) for p in receipt['networks']]
        paths = [kb.DATA / f'{s}.parquet' for s in stems]
        base += json.loads((R8 / 'results' / f'rows_V0_fold{fold}.json').read_text())
        for name, shares in VARIANTS.items():
            cache = OUT / f'rows_{name}_fold{fold}.json'
            if not cache.is_file():
                archive = OUT / f'{name}_fold{fold}.npz'
                if not archive.is_file():
                    kb.vbuild.predicted_archive(paths, nets, kb.BASIS, workers,
                                                kb.vbuild.Finish(**dict(v25, scalar_shares=shares))).write(archive)
                with Pool(workers) as pool:
                    parts = pool.map(r2.score_shot, [(str(archive), i, s) for i, s in enumerate(stems)])
                cache.write_text(json.dumps([dict(r, shot=f'{fold}:{i}') for i, part in enumerate(parts) for r in part]))
            part = json.loads(cache.read_text())
            rows[name] += part
            print(json.dumps({'variant': name, 'fold': fold, 'S': r2.pooled(part)['S']}), flush=True)
    shots = sorted({r['shot'] for r in base})
    tmean = np.nanmean(np.asarray([r['t'] for r in base], float), axis=0)
    zb = sums(base, shots, tmean)
    result = {'S_V0': r2.pooled(base)['S'], 'variants': {}}
    rng = np.random.default_rng(970)
    weights = [np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)).astype(float) for _ in range(20000)]
    for name, rr in rows.items():
        zn = sums(rr, shots, tmean)
        boots = np.array([score_from(zn, w) - score_from(zb, w) for w in weights])
        pooled, pooled0 = r2.pooled(rr), r2.pooled(base)
        one = np.ones(len(shots))
        scal = lambda z: (1 - z['pe'].sum(0) / (z['tq'].sum(0) - z['ts'].sum(0) ** 2 / z['cnt'].sum(0))).tolist()
        result['variants'][name] = {
            'S': pooled['S'], 'dS': pooled['S'] - pooled0['S'], 'ci95': np.percentile(boots, [2.5, 97.5]).tolist(),
            'P_positive': float((boots > 0).mean()),
            'terms': {k: [pooled.get(k), pooled0.get(k)] for k in pooled if k != 'S'},
            'scalars_r2': dict(zip(NAMES, scal(zn))), 'scalars_r2_V0': dict(zip(NAMES, scal(zb))),
            'D_mean': float(zn['D'].sum() / zn['frames'].sum()), 'D_mean_V0': float(zb['D'].sum() / zb['frames'].sum()),
            'per_fold_dS': [r2.pooled([r for r in rr if r['shot'].startswith(f'{f}:')])['S']
                            - r2.pooled([r for r in base if r['shot'].startswith(f'{f}:')])['S'] for f in range(5)],
            'passes': bool(pooled['S'] - pooled0['S'] >= 2e-5 and (boots > 0).mean() >= 0.9)}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
