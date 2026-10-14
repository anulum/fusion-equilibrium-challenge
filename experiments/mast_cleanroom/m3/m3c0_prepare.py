# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-0 step 1: sample MAST frames and export the solver inputs (inputs only)
"""Runs in the fec_adapter venv. Writes m3c0_inputs.npz (coil filaments and currents, Ip, TF current, Thomson,
grid, vessel wall contour) for the sampled frames; the FreeGS venv has neither pyarrow nor fec_adapter's
dependencies. Protocol protocol_m3c0_freegs_position_feasibility.json, amendment 1.

    PYTHONPATH=R/fec_adapter python m3c0_prepare.py
"""
import glob
import json
from pathlib import Path

from skimage.measure import find_contours
import numpy as np

from fec_adapter.shot_reader import aligned_shot, read_row
from fec_adapter.vessel import envelope, grids

HERE = Path(__file__).resolve().parent
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
SHOTS = (0, 301, 603, 904, 1205)
SHARE, PRESSURE_FACTOR, MIN_IP = 0.05, 2.25, 3.0e5
MU0, ELECTRON_VOLT, TF_TURNS = 4.0e-7 * np.pi, 1.602176634e-19, 24.0


def smoothed(r, values):
    along = np.linspace(r[0], r[-1], 400)
    w = np.exp(-0.5 * ((along[:, None] - r[None, :]) / 0.03) ** 2)
    return along, (w * values).sum(1) / w.sum(1)


def has_markers(r, te):
    along, s = smoothed(r, te)
    top = int(np.argmax(s))
    if top < 20 or top > 380 or s[top] < 50:
        return False
    level = SHARE * s[top]
    return bool(np.any(s[:top] < level) and np.any(s[top:] < level))


def wall():
    mask = envelope('MAST').astype(float)
    r_grid, z_grid = grids('MAST')
    line = max(find_contours(mask, 0.5), key=len)  # (row = Z index, column = R index), as the second agent's setup
    return (np.interp(line[:, 1], np.arange(len(r_grid)), r_grid).tolist(),
            np.interp(line[:, 0], np.arange(len(z_grid)), z_grid).tolist())


def main():
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    wall_r, wall_z = wall()
    arrays, frames = {'wall_r': np.asarray(wall_r), 'wall_z': np.asarray(wall_z)}, []
    for i in SHOTS:
        path = Path(files[i])
        row = read_row(path)
        shot = aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=np.float64))
        order = np.argsort(shot.thomson.r)
        live = [f for f in range(shot.n_frames)
                if np.isfinite(shot.plasma_current[f]) and abs(shot.plasma_current[f]) >= MIN_IP
                and np.all(np.isfinite(shot.coil_current[f]))
                and np.isfinite(shot.thomson.te[f, order]).sum() >= 30
                and has_markers(shot.thomson.r[order][np.isfinite(shot.thomson.te[f, order])],
                                shot.thomson.te[f, order][np.isfinite(shot.thomson.te[f, order])])]
        picks = sorted({live[int(q * (len(live) - 1))] for q in (0.25, 0.5, 0.75)}) if live else []
        arrays[f'shot_{i:04d}_coil_r'] = shot.coils.r
        arrays[f'shot_{i:04d}_coil_z'] = shot.coils.z
        arrays[f'shot_{i:04d}_coil_turns'] = shot.coils.turns
        arrays[f'shot_{i:04d}_coil_group'] = np.array([sorted(set(shot.coils.columns)).index(c) for c in shot.coils.columns])
        arrays[f'shot_{i:04d}_grid_r'] = shot.r_grid
        arrays[f'shot_{i:04d}_grid_z'] = shot.z_grid
        for f in picks:
            order_r = shot.thomson.r[order]
            te, ne = shot.thomson.te[f, order], shot.thomson.ne[f, order]
            ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
            _along, pe = smoothed(order_r[ok], ne[ok] * te[ok] * ELECTRON_VOLT)
            key = f'shot_{i:04d}_frame_{f:03d}'
            arrays[key + '_coil_current'] = shot.coil_current[f]
            frames.append({'shot': i, 'stem': path.stem, 'frame': f, 'ip_A': float(shot.plasma_current[f]),
                           'tf_A': float(shot.tf_current[f]), 'p_axis_Pa': PRESSURE_FACTOR * float(pe.max())})
    np.savez(HERE / 'm3c0_inputs.npz', **arrays)
    (HERE / 'm3c0_frames.json').write_text(json.dumps(frames, indent=1))
    print(len(frames), 'frames', flush=True)


if __name__ == '__main__':
    main()
