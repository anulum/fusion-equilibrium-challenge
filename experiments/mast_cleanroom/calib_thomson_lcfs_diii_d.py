# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST clean-room Track B, step 2: what Thomson sees at the true LCFS, on DIII-D (allowed truth)
"""On DIII-D validation shots, where the true equilibrium is known (rule 3 allows other machines' equilibria), find
where the scorer's LCFS of the TRUE map crosses the vertical Thomson laser (R = 1.94 m, top crossing), and read the
Thomson ne and Te there relative to the frame's peak. Also, for fractions q, the height where ne (Te) last exceeds
q * peak going up, minus the true crossing height. A tight distribution would give a label-free boundary marker that
can be carried to MAST's radial laser; a wide one says the Thomson edge is no usable LCFS marker.

    python calib_thomson_lcfs_diii_d.py [--every-shot 2] [--every-frame 5]
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_row, aligned_shot

REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$FEC_DATA/data/diii_d_train')
OUT = Path(__file__).resolve().parent / 'results' / 'calib_thomson_lcfs_diii_d'
LASER_R = 1.94
FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
_R = None


def _retry(fn, *args):
    """sshfs gives transient EPERM under parallel reads; retry a few times."""
    for attempt in range(6):
        try:
            return fn(*args)
        except PermissionError:
            time.sleep(2 + 3 * attempt)
    return fn(*args)


def _top_crossing(contour, r0):
    c = np.asarray(contour, float)
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((r - r0) * (r2 - r0) <= 0) & (r != r2)
    t = (r0 - r[hit]) / (r2[hit] - r[hit])
    zz = z[hit] + t * (z2[hit] - z[hit])
    return float(zz.max()) if zz.size else None


def _last_above(z, v, level, z_from):
    """Highest channel height above z_from, walking up from the peak, where v still exceeds level."""
    above = z >= z_from
    zs, vs = z[above], v[above]
    if zs.size == 0:
        return None
    below = np.nonzero(vs <= level)[0]
    if below.size == 0:
        return float(zs[-1])
    k = below[0]
    if k == 0:
        return float(zs[0])
    # linear interpolation between the last channel above and the first at or below the level
    z1, z2, v1, v2 = zs[k - 1], zs[k], vs[k - 1], vs[k]
    return float(z1 + (v1 - level) / (v1 - v2) * (z2 - z1))


def shot(task):
    global _R
    if _R is None:
        _R = Reader('DIII-D')
    stem, every_frame = task
    row = _retry(read_row, DATA / f'{stem}.parquet')
    times = np.asarray(row['efit_times'], float)
    s = aligned_shot(row, stem, times)
    psi_all = np.asarray(_retry(lambda: pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])['efit_psirz'][0].as_py()),
                         dtype=np.float64)
    th = s.thomson
    core = np.abs(th.r - LASER_R) < 0.005
    zc = th.z[core]
    order = np.argsort(zc)
    zc = zc[order]
    rows = []
    for f in range(0, len(times), every_frame):
        reading = _R.read(psi_all[f])
        if reading.contour is None:
            continue
        ztop = _top_crossing(reading.contour, LASER_R)
        if ztop is None:
            continue
        te, ne = th.te[f, core][order], th.ne[f, core][order]
        ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        if ok.sum() < 8:
            continue
        z, t, n = zc[ok], te[ok], ne[ok]
        covered = bool(z.max() > ztop + 0.005 and z.min() < ztop - 0.05)
        row_out = {'stem': stem, 'frame': f, 'z_top': ztop, 'z_axis': float(reading.axis[1]), 'covered': covered,
                   'channel_z_max': float(z.max()), 'ne_peak': float(n.max()), 'te_peak': float(t.max())}
        if covered:
            row_out['ne_ratio_at_lcfs'] = float(np.interp(ztop, z, n) / n.max())
            row_out['te_ratio_at_lcfs'] = float(np.interp(ztop, z, t) / t.max())
            zpk = float(reading.axis[1])
            for q in FRACTIONS:
                zn = _last_above(z, n, q * n.max(), zpk)
                zt = _last_above(z, t, q * t.max(), zpk)
                row_out[f'dz_ne_{q}'] = None if zn is None else zn - ztop
                row_out[f'dz_te_{q}'] = None if zt is None else zt - ztop
        rows.append(row_out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every-shot', type=int, default=2)
    ap.add_argument('--every-frame', type=int, default=5)
    a = ap.parse_args()
    stems = sorted(json.loads((REPO / 'fec_adapter/splits.json').read_text())['validation'])[::a.every_shot]
    with Pool(4) as p:
        rows = [r for rs in p.map(shot, [(s, a.every_frame) for s in stems], chunksize=2) for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rows.json').write_text(json.dumps(rows))
    cov = [r for r in rows if r['covered']]
    summ = {'shots': len(stems), 'frames': len(rows), 'covered_frames': len(cov)}
    for key in ('ne_ratio_at_lcfs', 'te_ratio_at_lcfs'):
        v = np.array([r[key] for r in cov])
        summ[key] = {'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90))}
    for q in FRACTIONS:
        for kind in ('ne', 'te'):
            v = np.array([r[f'dz_{kind}_{q}'] for r in cov if r[f'dz_{kind}_{q}'] is not None])
            summ[f'dz_{kind}_{q}'] = {'n': int(v.size), 'median_cm': float(100 * np.median(v)),
                                      'iqr_cm': float(100 * (np.percentile(v, 75) - np.percentile(v, 25))),
                                      'mad_cm': float(100 * np.median(np.abs(v - np.median(v))))}
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
