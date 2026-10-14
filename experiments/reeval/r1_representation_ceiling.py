# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D re-evaluation R1: ceiling of the flux-basis representation
"""How much score does the PCA flux basis itself cost? (diagnosis only)

True calibration maps (every 4th frame, 372 shots) are projected onto the first k directions of
the flux basis the networks predict in (k = 32, 45, 64, 128) and read back with the scorer's
derivation. Reported per k: pooled flux R2, mean D_LCFS and the seven derived-scalar R2 against
the true map's own derivation, and the implied composite of those terms (q95/betaN excluded).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$FEC_DATA/data/diii_d_train')
BASIS = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/flux_basis.npz'
KS = (32, 45, 64, 128)
STRIDE = 4
OUT = HERE / 'results' / 'r1_representation_ceiling.json'
_B = None


def work(stem):
    global _B
    from scipy.spatial.distance import directed_hausdorff
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    if _B is None:
        with np.load(BASIS) as b:
            _B = (b['mean'].astype(np.float64), b['directions'].astype(np.float64), Reader('DIII-D'))
    mean, dirs, reader = _B
    maps = np.asarray(pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])['efit_psirz'][0].as_py(),
                      dtype=np.float64)[::STRIDE]
    rows = []
    for truth in maps:
        rt = reader.read(truth)
        if rt.contour is None:
            continue
        ct = np.asarray(rt.contour, float)
        rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
        coords = (truth.ravel() - mean) @ dirs.T
        row = {'truth_scalars': [rt.values[n] for n in CONS_SCALARS], 'truth_sq': float(np.sum(truth ** 2)),
               'truth_sum': float(truth.sum()), 'n': truth.size}
        for k in KS:
            rec = (mean + coords[:k] @ dirs[:k]).reshape(truth.shape)
            r = reader.read(rec)
            if r.contour is None:
                d = 1.0
                vals = [np.nan] * 7
            else:
                c = np.asarray(r.contour, float)
                d = max(directed_hausdorff(c, ct)[0], directed_hausdorff(ct, c)[0]) / rgeo
                vals = [r.values[n] for n in CONS_SCALARS]
            row[k] = {'D': d, 'scalars': vals, 'sse': float(np.sum((rec - truth) ** 2))}
        rows.append(row)
    return rows


def main():
    splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
    stems = sorted(splits['calibration'])
    with Pool(6) as pool:
        rows = [r for part in pool.map(work, stems, chunksize=4) for r in part]
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    truth_s = np.asarray([r['truth_scalars'] for r in rows], float)
    n_pix = sum(r['n'] for r in rows)
    mean_psi = sum(r['truth_sum'] for r in rows) / n_pix
    ss_tot = sum(r['truth_sq'] for r in rows) - n_pix * mean_psi ** 2
    out = {'frames': len(rows), 'shots': len(stems), 'stride': STRIDE, 'part': 'calibration',
           'selection_or_fit': False}
    for k in KS:
        pred_s = np.asarray([r[k]['scalars'] for r in rows], float)
        r2 = {}
        for j, name in enumerate(CONS_SCALARS):
            ok = np.isfinite(truth_s[:, j]) & np.isfinite(pred_s[:, j])
            t = truth_s[ok, j]
            r2[name] = float(1 - np.sum((pred_s[ok, j] - t) ** 2) / np.sum((t - t.mean()) ** 2))
        flux_r2 = 1 - sum(r[k]['sse'] for r in rows) / ss_tot
        d = float(np.mean([r[k]['D'] for r in rows]))
        cons = float(np.mean([max(0.0, v) for v in r2.values()]))
        out[str(k)] = {'flux_r2': flux_r2, 'D_LCFS': d, 'consistency': cons, 'scalar_r2': r2,
                       'composite_without_qb_share': 0.55 * flux_r2 + 0.10 * (1 - d) + 0.20 * cons}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
