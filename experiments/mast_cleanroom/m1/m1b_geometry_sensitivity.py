# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M1b: sensitivity of the q95/betaN routines to the measured MAST geometry error (DIII-D only)
"""Protocol protocol_m1b_geometry_sensitivity_diii_d.json (sha a377bbd8...). DIII-D labels only (rule 3 clean).

    PYTHONPATH=R/fec_adapter python m1b_geometry_sensitivity.py
"""
import glob
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from fec_adapter.cross_machine import map_scalars
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import aligned_shot, read_row
from fec_adapter.split import bucket_of

HERE = Path(__file__).resolve().parent
DATA = '$FEC_DATA/data/diii_d_train/'
SHOTS, STRIDE, MIN_IP = 60, 4, 3.0e5
ARMS = ('A0', 'A1', 'A2', 'A3')
_R = None


def resample(reader, flux, source_r):
    dr = reader.r_grid[1] - reader.r_grid[0]
    dz = reader.z_grid[1] - reader.z_grid[0]
    return ndimage.map_coordinates(flux, [(reader.zz - reader.z_grid[0]) / dz, (source_r - reader.r_grid[0]) / dr],
                                   order=1, mode='nearest')


def perturbed(reader, flux, arm, axis_r, width):
    if arm == 'A0':
        return flux
    shift = 0.10 if arm == 'A2' else 0.05
    moved = resample(reader, flux, reader.rr + shift)
    if arm != 'A3':
        return moved
    squeeze = (width - 0.026) / width
    centre = axis_r - shift
    return resample(reader, moved, centre + (reader.rr - centre) / squeeze)


def scalars(reader, shot, frame, flux):
    reading = reader.read(flux)
    if reading.contour is None or not np.all(np.isfinite(reading.axis)):
        return float('nan'), float('nan'), None
    contour = np.asarray(reading.contour, float)
    axis = (float(reading.axis[0]), float(reading.axis[1]))
    edge_flux = float(np.mean(map_scalars._at(flux, reader.r_grid, reader.z_grid, contour[:, 0], contour[:, 1])))
    # The extremum of the grid next to the axis, as production's solved maps carry it: an interpolated axis value
    # lies inside the extremum, which leaves the axis cell at a negative normalised depth and no plasma region.
    row = int(np.argmin(np.abs(reader.z_grid - axis[1])))
    column = int(np.argmin(np.abs(reader.r_grid - axis[0])))
    nearby = flux[max(row - 1, 0):row + 2, max(column - 1, 0):column + 2]
    centre = float(flux[row, column])
    axis_flux = float(nearby.min()) if edge_flux > centre else float(nearby.max())
    f_vac = map_scalars.vacuum_f(shot.source, float(shot.tf_current[frame]))
    q95 = map_scalars.safety_factor(flux, reader.r_grid, reader.z_grid, axis, axis_flux, edge_flux, f_vac)
    beta = map_scalars.normalised_beta(flux, reader.r_grid, reader.z_grid, axis, axis_flux, edge_flux,
                                       shot.thomson.r, shot.thomson.z, shot.thomson.te[frame], shot.thomson.ne[frame],
                                       float(shot.plasma_current[frame]), f_vac)
    return q95, beta, (axis[0], float(contour[:, 0].max() - contour[:, 0].min()))


def one_shot(path):
    global _R
    if _R is None:
        _R = Reader('DIII-D')
    for attempt in range(5):
        try:
            row = read_row(Path(path))
            break
        except OSError:
            time.sleep(2.0 * (attempt + 1))
    else:
        return []
    shot = aligned_shot(row, Path(path).stem, np.asarray(row['efit_times'], dtype=np.float64))
    psi = np.asarray(row['efit_psirz'], dtype=np.float64)
    q_lab, b_lab = np.asarray(row['efit_q95'], float), np.asarray(row['efit_beta_n'], float)
    out = []
    for frame in range(0, shot.n_frames, STRIDE):
        ip = float(shot.plasma_current[frame])
        if not (np.isfinite(ip) and abs(ip) >= MIN_IP and np.isfinite(q_lab[frame]) and np.isfinite(b_lab[frame])):
            continue
        flux = psi[frame]
        q0, b0, geo = scalars(_R, shot, frame, flux)
        if geo is None:
            continue
        rec = {'shot': shot.name, 'frame': frame, 'q95_label': float(q_lab[frame]), 'betaN_label': float(b_lab[frame]),
               'A0': [q0, b0]}
        for arm in ARMS[1:]:
            q, b, _ = scalars(_R, shot, frame, perturbed(_R, flux, arm, geo[0], geo[1]))
            rec[arm] = [q, b]
        out.append(rec)
    return out


def summary(rows):
    res = {}
    for arm in ARMS:
        res[arm] = {}
        for k, name in enumerate(('q95', 'betaN')):
            y = np.array([r[f'{name}_label'] for r in rows])
            p = np.array([r[arm][k] for r in rows])
            ok = np.isfinite(p)
            yv, pv = y[ok], p[ok]
            var = float(np.var(yv))
            r2 = 1.0 - float(np.mean((pv - yv) ** 2)) / var
            ratio = float(np.median(pv / yv))
            r2_debiased = 1.0 - float(np.mean((pv / ratio - yv) ** 2)) / var
            # unsolved frames score as a zero prediction in the organisers' R2
            filled = np.where(ok, p, 0.0)
            r2_scored = 1.0 - float(np.mean((filled - y) ** 2)) / float(np.var(y))
            res[arm][name] = {'finite_share': float(ok.mean()), 'R2_finite': r2, 'R2_scored_clipped': max(r2_scored, 0.0),
                              'median_ratio': ratio, 'R2_bias_removed': r2_debiased,
                              'corr': float(np.corrcoef(pv, yv)[0, 1])}
        res[arm]['mean_clipped_qb'] = 0.5 * (res[arm]['q95']['R2_scored_clipped'] + res[arm]['betaN']['R2_scored_clipped'])
    base = res['A0']['mean_clipped_qb']
    drop = {arm: base - res[arm]['mean_clipped_qb'] for arm in ARMS[1:]}
    worst = max(drop['A1'], drop['A3'])
    verdict = ('geometry_explains_loss_M3_first' if worst >= 0.30 else
               'geometry_does_not_explain_M1c' if worst < 0.10 else 'both_geometry_first')
    return res, drop, verdict


def main():
    files = sorted(f for f in glob.glob(DATA + '*.parquet') if bucket_of(Path(f).stem) == 'validation')[:SHOTS]
    assert len(files) == SHOTS
    with Pool(2) as pool:
        rows = [r for part in pool.imap(one_shot, files, chunksize=1) for r in part]
    (HERE / 'm1b_rows.json').write_text(json.dumps(rows))
    res, drop, verdict = summary(rows)
    out = {'frames': len(rows), 'shots': len({r['shot'] for r in rows}), 'arms': res, 'drop_vs_A0': drop, 'verdict': verdict}
    (HERE / 'm1b_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
