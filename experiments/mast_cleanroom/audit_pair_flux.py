# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST clean-room Track B, step 3: on which flux surface do equal-Te points sit in our maps?
"""Label-free. For sampled frames of the submitted MAST archive: the Thomson Te profile smoothed as v7 does (3 cm
Gaussian), inner and outer radii at shares q of the peak; our map's normalised flux x = (psi_axis - psi)/(psi_axis -
psi_boundary) at those points (0 axis, 1 our LCFS; psi_boundary = mean flux on the scorer contour of our map).
If the solve honoured the pairs, x_in(q) = x_out(q); if our surfaces are shifted inboard relative to the Te profile,
x_in < x_out. Where x at the Te 2-5 % points is not near 1, our boundary disagrees with the Thomson edge.

    python audit_pair_flux.py [--every-shot 3] [--every-frame 4]
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_shot

DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
ARCHIVE = ('$LAB/agentic_group_workspace/'
           'second_agent-recovery-20260926/results/joint_submission_56cb/mast_public_test.npz')
OUT = Path(__file__).resolve().parent / 'results' / 'audit_pair_flux'
SHARES = (0.02, 0.05, 0.10, 0.25, 0.50, 0.85)
SMOOTH = 0.03
_R = None


def pairs(r, te):
    along = np.linspace(r[0], r[-1], 400)
    w = np.exp(-0.5 * ((along[:, None] - r[None, :]) / SMOOTH) ** 2)
    s = (w * te).sum(1) / w.sum(1)
    top = int(np.argmax(s))
    if top < 20 or top > 380 or s[top] < 50:
        return None
    out = {}
    for q in SHARES:
        lv = q * s[top]
        bi, bo = np.flatnonzero(s[:top] < lv), np.flatnonzero(s[top:] < lv)
        if bi.size == 0 or bo.size == 0:
            continue
        i, j = int(bi[-1]), top + int(bo[0])
        out[q] = (float(np.interp(lv, [s[i], s[i + 1]], [along[i], along[i + 1]])),
                  float(np.interp(lv, [s[j], s[j - 1]], [along[j], along[j - 1]])))
    return out


def shot(task):
    global _R
    if _R is None:
        _R = Reader('MAST')
    i, path, every = task
    s = read_shot(Path(path))
    with np.load(ARCHIVE) as a:
        psi = a[f'shot_{i:04d}_psirz'].astype(np.float64)
    th = s.thomson
    o = np.argsort(th.r)
    rr, z0 = th.r[o], float(np.median(th.z))
    rows = []
    for f in range(0, s.n_frames, every):
        te, ne = th.te[f, o], th.ne[f, o]
        ok = np.isfinite(te) & (te > 0)
        if ok.sum() < 30 or (np.isfinite(ne[ok]).any() and np.nanmedian(ne[ok]) < 2e18):
            continue
        p = pairs(rr[ok], te[ok])
        if not p:
            continue
        rd = _R.read(psi[f])
        if rd.contour is None:
            continue
        interp = RegularGridInterpolator((_R.z_grid, _R.r_grid), psi[f])
        pa = float(interp([[rd.axis[1], rd.axis[0]]])[0])
        c = np.asarray(rd.contour)
        pb = float(np.mean(interp(np.column_stack([c[:, 1], c[:, 0]]))))
        span = pa - pb
        row = {'shot': i, 'frame': f}
        for q, (ri, ro) in p.items():
            xi = float((pa - interp([[z0, ri]])[0]) / span)
            xo = float((pa - interp([[z0, ro]])[0]) / span)
            row[f'x_in_{q}'], row[f'x_out_{q}'] = xi, xo
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every-shot', type=int, default=3)
    ap.add_argument('--every-frame', type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    with Pool(4) as pool:
        rows = [r for rs in pool.map(shot, [(i, files[i], a.every_frame) for i in range(0, 1206, a.every_shot)], chunksize=2)
                for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rows.json').write_text(json.dumps(rows))
    summ = {'frames': len(rows)}
    for q in SHARES:
        xi = np.array([r[f'x_in_{q}'] for r in rows if f'x_in_{q}' in r])
        xo = np.array([r[f'x_out_{q}'] for r in rows if f'x_out_{q}' in r])
        d = np.array([r[f'x_out_{q}'] - r[f'x_in_{q}'] for r in rows if f'x_in_{q}' in r])
        summ[str(q)] = {'n': int(xi.size), 'x_in_median': float(np.median(xi)), 'x_out_median': float(np.median(xo)),
                        'x_out_minus_in_median': float(np.median(d)), 'x_out_minus_in_p10': float(np.percentile(d, 10)),
                        'x_out_minus_in_p90': float(np.percentile(d, 90)),
                        'share_out_beyond_lcfs': float(np.mean(xo > 1.0)), 'share_in_beyond_lcfs': float(np.mean(xi > 1.0))}
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
