# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST v8b: move the v7 boundary to the Thomson edge (protocol 42744aa6)
"""Post-process a MAST v7 archive: per frame, warp the map in R about its axis, separately inboard and outboard,
so the organisers' reading of the LCFS crosses the laser height at the Thomson Te 5 % marker moved by the
DIII-D-calibrated 1.7 cm outward. Inputs and our own maps only (rule 3).

    python edge_boundary.py --pilot            # the 40 pilot shots, metrics only
    python edge_boundary.py --build OUT.npz    # every shot, writes an archive
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_shot

HERE = Path(__file__).resolve().parent
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
V7 = ('$LAB/agentic_group_workspace/'
      'second_agent-recovery-20260926/results/joint_submission_56cb/mast_public_test.npz')
SHARE, OFFSET, FADE, ROUNDS = 0.05, 0.017, 0.5, 2
PILOT = list(range(0, 1206, 30))
_R = None


def markers(r, te):
    """(inner, outer) radii where the 3 cm-smoothed Te falls to SHARE of its peak; None where the laser ends first."""
    along = np.linspace(r[0], r[-1], 400)
    w = np.exp(-0.5 * ((along[:, None] - r[None, :]) / 0.03) ** 2)
    s = (w * te).sum(1) / w.sum(1)
    top = int(np.argmax(s))
    if top < 20 or top > 380 or s[top] < 50:
        return None, None
    lv = SHARE * s[top]
    bi, bo = np.flatnonzero(s[:top] < lv), np.flatnonzero(s[top:] < lv)
    inner = outer = None
    if bi.size:
        i = int(bi[-1])
        inner = float(np.interp(lv, [s[i], s[i + 1]], [along[i], along[i + 1]]))
    if bo.size:
        j = top + int(bo[0])
        outer = float(np.interp(lv, [s[j], s[j - 1]], [along[j], along[j - 1]]))
    return inner, outer


def crossings(contour, z0):
    c = np.asarray(contour, float)
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((z - z0) * (z2 - z0) <= 0) & (z != z2)
    t = (z0 - z[hit]) / (z2[hit] - z[hit])
    x = r[hit] + t * (r2[hit] - r[hit])
    return (float(x.min()), float(x.max())) if x.size >= 2 else (None, None)


def warp(reader, flux, axis, s_in, s_out, half_w, half_h):
    """Side-wise radial stretch about the axis (the new map at R is the old map at the carried point)."""
    d = np.hypot((reader.rr - axis[0]) / half_w, (reader.zz - axis[1]) / half_h)
    weight = np.exp(-(np.clip(d - 1.0, 0.0, None) / FADE) ** 2)
    scale = np.where(reader.rr < axis[0], 1.0 / s_in - 1.0, 1.0 / s_out - 1.0)
    source_r = reader.rr + weight * (reader.rr - axis[0]) * scale
    dr = reader.r_grid[1] - reader.r_grid[0]
    dz = reader.z_grid[1] - reader.z_grid[0]
    return ndimage.map_coordinates(flux, [(reader.zz - reader.z_grid[0]) / dz, (source_r - reader.r_grid[0]) / dr],
                                   order=1, mode='nearest')


def edit_shot(task):
    global _R
    if _R is None:
        _R = Reader('MAST')
    i, path = task
    shot = read_shot(Path(path))
    with np.load(V7) as a:
        maps = a[f'shot_{i:04d}_psirz'].astype(np.float64)
    th = shot.thomson
    o = np.argsort(th.r)
    rr, z0 = th.r[o], float(np.median(th.z))
    out = maps.copy()
    rows = []
    for f in range(shot.n_frames):
        te = th.te[f, o]
        ok = np.isfinite(te) & (te > 0)
        if ok.sum() < 30:
            continue
        mi, mo = markers(rr[ok], te[ok])
        if mi is None and mo is None:
            continue
        rd = _R.read(maps[f])
        if rd.contour is None:
            continue
        ci, co = crossings(rd.contour, z0)
        if ci is None:
            continue
        ti = None if mi is None else mi - OFFSET
        to = None if mo is None else mo + OFFSET
        axis = rd.axis
        c = np.asarray(rd.contour)
        half_w, half_h = 0.5 * (co - ci), 0.5 * (c[:, 1].max() - c[:, 1].min())
        flux, s_tot_in, s_tot_out, row = maps[f], 1.0, 1.0, {'shot': i, 'frame': f}
        for _ in range(ROUNDS):
            s_in = 1.0 if ti is None or ci >= axis[0] else (ti - axis[0]) / (ci - axis[0])
            s_out = 1.0 if to is None or co <= axis[0] else (to - axis[0]) / (co - axis[0])
            if not (0.5 < s_in < 2.0 and 0.5 < s_out < 2.0):
                break
            flux = warp(_R, flux, axis, s_in, s_out, half_w, half_h)
            s_tot_in *= s_in
            s_tot_out *= s_out
            new = _R.read(flux)
            if new.contour is None:
                row['lost_contour'] = True
                break
            ci, co = crossings(new.contour, z0)
            if ci is None:
                row['lost_contour'] = True
                break
            axis = new.axis
        row.update({'target_in': ti, 'target_out': to, 's_in': s_tot_in, 's_out': s_tot_out,
                    'cross_in': ci, 'cross_out': co})
        if not row.get('lost_contour'):
            out[f] = flux
        rows.append(row)
    return i, out.astype(np.float32), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--build', type=Path)
    a = ap.parse_args()
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    idx = PILOT if a.pilot else list(range(1206))
    with Pool(6) as p:
        results = p.map(edit_shot, [(i, files[i]) for i in idx], chunksize=1)
    rows = [r for _, _, rs in results for r in rs]
    OUTD = HERE / ('pilot' if a.pilot else 'build')
    OUTD.mkdir(exist_ok=True)
    (OUTD / 'rows.json').write_text(json.dumps(rows))
    edited = [r for r in rows if not r.get('lost_contour')]
    hit, sides, scales = 0, 0, []
    for r in edited:
        for side in ('in', 'out'):
            t = r[f'target_{side}']
            if t is None:
                continue
            sides += 1
            hit += abs(r[f'cross_{side}'] - t) <= 0.01
            scales.append(abs(r[f's_{side}'] - 1.0))
    summ = {'shots': len(idx), 'frames_edited': len(rows), 'lost_contour': sum(1 for r in rows if r.get('lost_contour')),
            'sides': sides, 'A1_share_within_1cm': hit / max(sides, 1), 'A3_median_abs_s_minus_1': float(np.median(scales)),
            'p90_abs_s_minus_1': float(np.percentile(scales, 90))}
    summ['A2_share_contour_kept'] = 1 - summ['lost_contour'] / max(len(rows), 1)
    summ['accepted'] = bool(summ['A1_share_within_1cm'] >= 0.8 and summ['A2_share_contour_kept'] >= 0.99
                            and summ['A3_median_abs_s_minus_1'] <= 0.15)
    (OUTD / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    if a.build:
        arrays = {}
        with np.load(V7) as v7:
            for k in v7.files:
                arrays[k] = v7[k]
        for i, maps, _ in results:
            arrays[f'shot_{i:04d}_psirz'] = maps
        np.savez_compressed(a.build, **arrays)
        print('written', a.build, flush=True)


if __name__ == '__main__':
    main()
