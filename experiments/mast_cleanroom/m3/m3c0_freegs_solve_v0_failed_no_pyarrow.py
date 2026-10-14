# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-0 step 1: FreeGS free-boundary solves of sampled MAST frames from inputs only
"""Runs in the isolated FreeGS venv (the second agent's checkout 9fbec205, unchanged). Writes the solved flux on the
organisers' MAST grid, (Z, R) order, to m3c0_freegs_maps.npz. Protocol protocol_m3c0_freegs_position_feasibility.json.

    PYTHONPATH=R/fec_adapter FREEGS_VENV/bin/python m3c0_freegs_solve.py
"""
import glob
import json
import time
import traceback
from pathlib import Path

import contourpy
import freegs
import numpy as np
from freegs.filament_coil import FilamentCoil

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
    lines = contourpy.contour_generator(z=mask).lines(0.5)
    line = max(lines, key=len)
    return (np.interp(line[:, 0], np.arange(len(r_grid)), r_grid).tolist(),
            np.interp(line[:, 1], np.arange(len(z_grid)), z_grid).tolist())


def solve(shot, frame, wall_rz):
    coils = shot.coils
    currents = shot.coil_current[frame]
    machine_coils = []
    for index, name in enumerate(sorted(set(coils.columns))):
        ids = [j for j, column in enumerate(coils.columns) if column == name]
        shape = [(float(coils.r[ids].min()), float(coils.z[ids].min())),
                 (float(coils.r[ids].min()), float(coils.z[ids].max())),
                 (float(coils.r[ids].max()), float(coils.z[ids].max())),
                 (float(coils.r[ids].max()), float(coils.z[ids].min()))]
        machine_coils.append((f'C{index:02d}', FilamentCoil(
            Rfil=coils.r[ids], Zfil=coils.z[ids], shape=shape, turns=float(np.sum(coils.turns[ids])),
            current=float(currents[ids[0]] / coils.turns[ids[0]]), control=False)))
    tokamak = freegs.machine.Machine(machine_coils, wall=freegs.machine.Wall(*wall_rz))
    r_grid, z_grid = shot.r_grid, shot.z_grid
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=float(r_grid[0]), Rmax=float(r_grid[-1]),
                            Zmin=float(z_grid[0]), Zmax=float(z_grid[-1]), nx=len(r_grid), ny=len(z_grid))
    order = np.argsort(shot.thomson.r)
    r = shot.thomson.r[order]
    te, ne = shot.thomson.te[frame, order], shot.thomson.ne[frame, order]
    ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    _along, pe = smoothed(r[ok], ne[ok] * te[ok] * ELECTRON_VOLT)
    p_axis = PRESSURE_FACTOR * float(pe.max())
    ip = float(shot.plasma_current[frame])
    f_vacuum = MU0 * TF_TURNS * abs(float(shot.tf_current[frame])) / (2.0 * np.pi)
    profiles = freegs.jtor.ConstrainPaxisIp(eq, p_axis, ip, f_vacuum)
    freegs.solve(eq, profiles, constrain=None, show=False, maxits=50, check_limited=True)
    flux = np.asarray(eq.psi()).T  # FreeGS stores [R, Z]; the organisers' maps are [Z, R]
    return flux, {'ip_input_A': ip, 'ip_solved_A': float(eq.plasmaCurrent()), 'p_axis_Pa': p_axis,
                  'f_vacuum_Tm': f_vacuum}


def main():
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    wall_rz = wall()
    maps, rows = {}, []
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
        for f in picks:
            began = time.time()
            record = {'shot': i, 'stem': path.stem, 'frame': f}
            try:
                flux, info = solve(shot, f, wall_rz)
                record.update(info, converged=True, finite=bool(np.isfinite(flux).all()))
                maps[f'shot_{i:04d}_frame_{f:03d}'] = flux.astype(np.float64)
            except Exception as error:  # a non-converged or failed solve is a result of this test
                record.update(converged=False, error=f'{type(error).__name__}: {error}'[:300],
                              trace=traceback.format_exc()[-600:])
            record['seconds'] = time.time() - began
            rows.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'trace'}), flush=True)
    np.savez(HERE / 'm3c0_freegs_maps.npz', **maps)
    (HERE / 'm3c0_freegs_rows.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
