# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST clean-room Track B, step 1: our boundary against the Thomson plasma edge (label-free)
"""Where does our submitted MAST boundary cross the Thomson plane, and where does the radial Thomson laser see the
plasma end? Inputs and our own predictions only; no MAST equilibrium field, demo shot or MAST score is read (rule 3).

Per sampled frame: our LCFS (organisers' reading, Reader('MAST')) crossings of the laser height -> R_in, R_out;
Thomson edge = outermost/innermost channel whose ne exceeds a fraction of the frame's peak ne (0.1 and 0.3 reported)
-> Rts_in, Rts_out; also the laser's own coverage. Output: distributions of R_ours - R_thomson per side, per shot.

    python audit_thomson_edge.py [--every-shot 3] [--every-frame 4]
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_shot

# local copy of the SAS set (sshfs gave transient EPERM under parallel reads; rsync --size-only check: 0 differences)
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
ARCHIVE = ('$LAB/agentic_group_workspace/'
           'second_agent-recovery-20260926/results/joint_submission_56cb/mast_public_test.npz')
OUT = Path(__file__).resolve().parent / 'results' / 'audit_thomson_edge'
FRACTIONS = (0.1, 0.2, 0.3)
TE_FRACTIONS = (0.03, 0.05)   # DIII-D calibration (calib_thomson_lcfs_diii_d.py): Te 5 % of peak marks the true LCFS at -1.7 cm, IQR 1.6 cm
MIN_PEAK_NE = 2e18
_R = None


def _crossings(contour, z0):
    """R values where the closed polyline crosses the height z0."""
    c = np.asarray(contour, float)
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = (z - z0) * (z2 - z0) <= 0
    hit &= z != z2
    t = (z0 - z[hit]) / (z2[hit] - z[hit])
    return r[hit] + t * (r2[hit] - r[hit])


def shot(task):
    global _R
    if _R is None:
        _R = Reader('MAST')
    i, path, every_frame = task
    s = read_shot(Path(path))
    with np.load(ARCHIVE) as a:
        psi = a[f'shot_{i:04d}_psirz']
    assert psi.shape[0] == s.n_frames, (i, psi.shape, s.n_frames)
    th = s.thomson
    order = np.argsort(th.r)
    rr, z0 = th.r[order], float(np.median(th.z))
    rows = []
    for f in range(0, s.n_frames, every_frame):
        ne = th.ne[f, order]
        ok = np.isfinite(ne) & (ne > 0)
        if ok.sum() < 10 or np.nanmax(ne[ok]) < MIN_PEAK_NE:
            continue
        reading = _R.read(psi[f].astype(np.float64))
        if reading.contour is None:
            rows.append({'shot': i, 'frame': f, 'contour': False})
            continue
        x = _crossings(reading.contour, z0)
        if x.size < 2:
            continue
        row = {'shot': i, 'frame': f, 'contour': True, 'ip_ka': float(abs(s.plasma_current[f]) / 1e3),
               'r_in': float(x.min()), 'r_out': float(x.max()), 'cover_in': float(rr[ok].min()), 'cover_out': float(rr[ok].max()),
               'axis_r': float(reading.axis[0]), 'volume': reading.values['volume'], 'kappa': reading.values['kappa']}
        peak = float(np.nanmax(ne[ok]))
        for q in FRACTIONS:
            inside = rr[ok][ne[ok] > q * peak]
            row[f'ts_in_{q}'], row[f'ts_out_{q}'] = float(inside.min()), float(inside.max())
        te = th.te[f, order]
        okt = ok & np.isfinite(te) & (te > 0)
        tpk = float(np.nanmax(te[okt]))
        for q in TE_FRACTIONS:
            inside = rr[okt][te[okt] > q * tpk]
            row[f'te_in_{q}'], row[f'te_out_{q}'] = float(inside.min()), float(inside.max())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every-shot', type=int, default=3)
    ap.add_argument('--every-frame', type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    tasks = [(i, files[i], a.every_frame) for i in range(0, len(files), a.every_shot)]
    with Pool(6) as p:
        rows = [r for rs in p.map(shot, tasks, chunksize=2) for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rows.json').write_text(json.dumps(rows))
    good = [r for r in rows if r.get('contour')]
    summ = {'frames': len(rows), 'with_contour': len(good), 'shots': len({r['shot'] for r in good})}
    for q in FRACTIONS:
        for side in ('in', 'out'):
            d = np.array([r[f'r_{side}'] - r[f'ts_{side}_{q}'] for r in good])
            summ[f'd_{side}_{q}'] = {'median': float(np.median(d)), 'p10': float(np.percentile(d, 10)),
                                     'p90': float(np.percentile(d, 90)), 'share_abs_gt_5cm': float(np.mean(np.abs(d) > 0.05)),
                                     'share_abs_gt_10cm': float(np.mean(np.abs(d) > 0.10))}
    for q in TE_FRACTIONS:
        for side in ('in', 'out'):
            d = np.array([r[f'r_{side}'] - r[f'te_{side}_{q}'] for r in good])
            summ[f'd_te_{side}_{q}'] = {'median': float(np.median(d)), 'p10': float(np.percentile(d, 10)),
                                        'p90': float(np.percentile(d, 90)), 'share_abs_gt_5cm': float(np.mean(np.abs(d) > 0.05)),
                                        'share_abs_gt_10cm': float(np.mean(np.abs(d) > 0.10))}
    # per-shot outer mismatch, to see whether a group of shots stands apart
    per = {}
    for r in good:
        per.setdefault(r['shot'], []).append(r['r_out'] - r['ts_out_0.1'])
    m = np.array([np.median(v) for v in per.values()])
    summ['per_shot_out_0.1_median_hist'] = np.histogram(m, bins=[-0.6, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.6])[0].tolist()
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
