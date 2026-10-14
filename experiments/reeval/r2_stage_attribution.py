# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D re-evaluation R2: where the pipeline gains and loses (OOF)
"""Stage attribution on out-of-fold maps (fold 0: 80 training shots the S16 networks never saw).

Builds the fold's networks with progressively more of the v25 finisher switched on, scores every
variant against the true maps with the scorer's derivation (every 2nd frame): pooled flux R2,
mean D_LCFS, the seven derived-scalar R2, q95/betaN R2, and the composite. Diagnosis only.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
K = HERE.parent / 'kfold'
sys.path.insert(0, str(K))
import kfold_build as kb  # noqa: E402

DATA = kb.DATA
OUT = HERE / 'results' / 'r2_stage_attribution_fold0'
STRIDE = 2


def variants(v25):
    base = dict(v25)
    off = dict(base, smooth_frames=0, axis_rounds=0, shape=(), x_points=False, reshape_after_x_points=False,
               contour=False, scalar_shares=())
    return {
        'raw64': dict(off, kept_coordinates=64, smooth_from=64),
        'raw45': dict(off, kept_coordinates=45, smooth_from=45),
        'smooth': dict(off, smooth_frames=base['smooth_frames'], smooth_from=base['smooth_from']),
        'smooth_axis': dict(off, smooth_frames=base['smooth_frames'], smooth_from=base['smooth_from'],
                            axis_rounds=base['axis_rounds']),
        'plus_shape': dict(base, x_points=False, reshape_after_x_points=False, contour=False),
        'plus_xpoints': dict(base, contour=False),
        'v25_full': base,
    }


def score_shot(task):
    from scipy.spatial.distance import directed_hausdorff
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    archive, index, stem = task
    reader = Reader('DIII-D')
    t = pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz', 'efit_q95', 'efit_beta_n'])
    truth = np.asarray(t['efit_psirz'][0].as_py(), dtype=np.float64)
    q95 = np.asarray(t['efit_q95'][0].as_py(), dtype=np.float64)
    bn = np.asarray(t['efit_beta_n'][0].as_py(), dtype=np.float64)
    with np.load(archive) as a:
        maps = a[f'shot_{index:04d}_psirz'].astype(np.float64)
        pq95 = a[f'shot_{index:04d}_q95'].astype(np.float64)
        pbn = a[f'shot_{index:04d}_betaN'].astype(np.float64)
    rows = []
    for f in range(0, len(maps), STRIDE):
        rt = reader.read(truth[f])
        if rt.contour is None:
            continue
        ct = np.asarray(rt.contour, float)
        rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
        r = reader.read(maps[f])
        if r.contour is None:
            d, vals = 1.0, [np.nan] * 7
        else:
            c = np.asarray(r.contour, float)
            d = max(directed_hausdorff(c, ct)[0], directed_hausdorff(ct, c)[0]) / rgeo
            vals = [r.values[n] for n in CONS_SCALARS]
        rows.append({'D': d, 'p': vals, 't': [rt.values[n] for n in CONS_SCALARS],
                     'sse': float(np.sum((maps[f] - truth[f]) ** 2)), 'tsum': float(truth[f].sum()),
                     'tsq': float(np.sum(truth[f] ** 2)), 'n': truth[f].size,
                     'q': [pq95[f], q95[f], pbn[f], bn[f]]})
    return rows


def pooled(rows):
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    n = sum(r['n'] for r in rows)
    m = sum(r['tsum'] for r in rows) / n
    flux = 1 - sum(r['sse'] for r in rows) / (sum(r['tsq'] for r in rows) - n * m * m)
    p = np.asarray([r['p'] for r in rows], float)
    t = np.asarray([r['t'] for r in rows], float)
    r2 = {}
    for j, name in enumerate(CONS_SCALARS):
        ok = np.isfinite(t[:, j])
        pj = np.where(np.isfinite(p[ok, j]), p[ok, j], np.nanmean(t[ok, j]))
        r2[name] = float(1 - np.sum((pj - t[ok, j]) ** 2) / np.sum((t[ok, j] - t[ok, j].mean()) ** 2))
    q = np.asarray([r['q'] for r in rows], float)
    qb = []
    for i in (0, 2):
        ok = np.isfinite(q[:, i + 1]) & np.isfinite(q[:, i])
        qb.append(1 - np.sum((q[ok, i] - q[ok, i + 1]) ** 2) / np.sum((q[ok, i + 1] - q[ok, i + 1].mean()) ** 2))
    d = float(np.mean([r['D'] for r in rows]))
    cons = float(np.mean([max(0, v) for v in r2.values()]))
    qbm = float(np.mean([max(0, v) for v in qb]))
    return {'flux_r2': flux, 'D': d, 'consistency': cons, 'qb': qbm, 'scalar_r2': r2,
            'S': 0.55 * max(0, flux) + 0.15 * qbm + 0.10 * (1 - min(1, d)) + 0.20 * cons}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fold = 0
    receipt = json.loads((K / f'oof_fold{fold}' / 'receipt.json').read_text())
    stems = receipt['stems']
    networks = [Path(p) for p in receipt['networks']]
    _, v25 = kb.v25_finish()
    paths = [DATA / f'{s}.parquet' for s in stems]
    results = {}
    for name, values in variants(v25).items():
        archive = K / f'oof_fold{fold}' / 'diii_d_train.npz' if name == 'v25_full' else OUT / f'{name}.npz'
        if name != 'v25_full' and not archive.is_file():
            finish = kb.vbuild.Finish(**values)
            kb.vbuild.predicted_archive(paths, networks, kb.BASIS, 6, finish).write(archive)
        with Pool(6) as pool:
            rows = [r for part in pool.map(score_shot, [(str(archive), i, s) for i, s in enumerate(stems)]) for r in part]
        results[name] = pooled(rows)
        print(json.dumps({name: {k: (round(v, 7) if isinstance(v, float) else v) for k, v in results[name].items() if k != 'scalar_r2'}}), flush=True)
    (OUT / 'summary.json').write_text(json.dumps({'fold': fold, 'shots': len(stems), 'stride': STRIDE,
                                                  'selection_or_fit': False, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
