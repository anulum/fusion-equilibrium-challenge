# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D11 exploration: q95 read off a DIII-D flux map against the network head
"""How well does q95 computed from a flux map match the label, on the true map and on our out-of-fold map?

q = F / 2π · ∮ dl / (R |∇ψ|) on the surface at 95 % of the normalised flux, F = μ0 · 144 · |I_TF| / 2π (vacuum).
The map is a bicubic spline; the boundary flux is the mean of the spline on the scorer's own LCFS contour
(scoring_kit.extract_lcfs); the 95 % surface is traced on a four times finer grid.

    PYTHONPATH=R/fec_adapter python q95_map_ceiling.py FOLD [STEP]

Measurement on DIII-D training shots of one K-fold fold (labels allowed). No fit, no selection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import RectBivariateSpline
from skimage import measure

from fec_adapter.scoring_kit.common import AXIS_SIGN
from fec_adapter.scoring_kit.lcfs import extract_lcfs
from fec_adapter.shot_reader import aligned_shot, input_columns

MU0 = 4e-7 * np.pi
HERE = Path(__file__).resolve().parent
C = HERE.parent
DATA = Path('$FEC_DATA/data/diii_d_train')
FINE = 4


def q95_of(psi, r_grid, z_grid, f_vacuum):
    """Return q at 95 % of the normalised flux from one map, or NaN."""
    sign = AXIS_SIGN['DIII-D']
    contour = extract_lcfs(psi, r_grid, z_grid, 'DIII-D', axis_sign=sign)
    if contour is None:
        return np.nan
    phi = sign * np.asarray(psi, dtype=np.float64)
    spline = RectBivariateSpline(z_grid, r_grid, phi, kx=3, ky=3)
    edge = float(np.mean(spline.ev(contour[:, 1], contour[:, 0])))
    zf = np.linspace(z_grid[0], z_grid[-1], FINE * (z_grid.size - 1) + 1)
    rf = np.linspace(r_grid[0], r_grid[-1], FINE * (r_grid.size - 1) + 1)
    fine = spline(zf, rf)
    inside_z = (zf > contour[:, 1].min()) & (zf < contour[:, 1].max())
    inside_r = (rf > contour[:, 0].min()) & (rf < contour[:, 0].max())
    window = fine[np.ix_(inside_z, inside_r)]
    iz, ir = np.unravel_index(np.argmax(window), window.shape)
    axis_z, axis_r, axis = zf[inside_z][iz], rf[inside_r][ir], float(window[iz, ir])
    level = axis + 0.95 * (edge - axis)
    best = None
    for c in measure.find_contours(fine, level):
        if not np.allclose(c[0], c[-1]):
            continue
        z = zf[0] + c[:, 0] * (zf[1] - zf[0])
        r = rf[0] + c[:, 1] * (rf[1] - rf[0])
        if r.min() < axis_r < r.max() and z.min() < axis_z < z.max() and (best is None or r.size > best[0].size):
            best = (r, z)
    if best is None:
        return np.nan
    r, z = best
    mr, mz = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
    grad = np.hypot(spline.ev(mz, mr, dx=1), spline.ev(mz, mr, dy=1))
    if np.any(grad <= 0.0):
        return np.nan
    return float(f_vacuum / (2.0 * np.pi) * np.sum(np.hypot(np.diff(r), np.diff(z)) / (mr * grad)))


def main():
    fold, step = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
    receipt = json.loads((C / f'kfold/oof_fold{fold}/receipt.json').read_text())
    archive = np.load(C / f'kfold/oof_fold{fold}/diii_d_train.npz')
    rows = []
    for index, stem in enumerate(receipt['stems']):
        path = DATA / f'{stem}.parquet'
        names = pq.read_schema(path).names
        cols = input_columns(dict.fromkeys(names))
        t = pq.read_table(path, columns=cols + ['efit_q95', 'efit_psirz', 'efit_grid_R', 'efit_grid_Z']).to_pylist()[0]
        shot = aligned_shot({k: t[k] for k in cols}, stem, np.asarray(t['efit_times'], float))
        r_grid, z_grid = np.asarray(t['efit_grid_R'], float), np.asarray(t['efit_grid_Z'], float)
        truth = np.asarray(t['efit_psirz'], float)
        ours = archive[f'shot_{index:04d}_psirz']
        head = archive[f'shot_{index:04d}_q95']
        label = np.asarray(t['efit_q95'], float)
        assert ours.shape == truth.shape, (stem, ours.shape, truth.shape)
        for i in range(0, shot.n_frames, step):
            f_vacuum = MU0 * 144.0 * abs(float(shot.tf_current[i])) / (2.0 * np.pi)
            rows.append((index, i, label[i], float(head[i]), q95_of(truth[i], r_grid, z_grid, f_vacuum),
                         q95_of(ours[i], r_grid, z_grid, f_vacuum), abs(float(shot.plasma_current[i]))))
        if index % 10 == 9:
            print(json.dumps({'shots_done': index + 1, 'frames': len(rows)}), flush=True)
    rows = np.asarray(rows, dtype=np.float64)
    np.save(HERE / f'q95_rows_fold{fold}.npy', rows)
    print('ROWS_SAVED', rows.shape, flush=True)


if __name__ == '__main__':
    main()
