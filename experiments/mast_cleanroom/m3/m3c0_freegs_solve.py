# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-0 step 2: FreeGS free-boundary solves of the sampled MAST frames
"""Runs in the isolated FreeGS venv (the second agent's checkout 9fbec205, unchanged) on m3c0_inputs.npz only. Writes the
solved flux on the organisers' MAST grid, (Z, R) order, to m3c0_freegs_maps.npz. Setup as the second agent's
run_freegs_actual_mast_vessel_20260929.py (every input coil a FilamentCoil at its measured current, vessel wall,
ConstrainPaxisIp with FreeGS default exponents, check_limited, maxits 50); p_axis = 2.25 x Thomson peak p_e, signed Ip.

    FREEGS_VENV/bin/python m3c0_freegs_solve.py
"""
import json
import time
import traceback
from pathlib import Path

import freegs
import numpy as np
from freegs.filament_coil import FilamentCoil

HERE = Path(__file__).resolve().parent
MU0, TF_TURNS = 4.0e-7 * np.pi, 24.0


def solve(data, frame):
    i = frame['shot']
    r, z = data[f'shot_{i:04d}_coil_r'], data[f'shot_{i:04d}_coil_z']
    turns, group = data[f'shot_{i:04d}_coil_turns'], data[f'shot_{i:04d}_coil_group']
    currents = data[f'shot_{i:04d}_frame_{frame["frame"]:03d}_coil_current']
    coils = []
    for g in sorted(set(group.tolist())):
        ids = np.flatnonzero(group == g)
        shape = [(float(r[ids].min()), float(z[ids].min())), (float(r[ids].min()), float(z[ids].max())),
                 (float(r[ids].max()), float(z[ids].max())), (float(r[ids].max()), float(z[ids].min()))]
        coils.append((f'C{g:02d}', FilamentCoil(Rfil=r[ids], Zfil=z[ids], shape=shape, turns=float(np.sum(turns[ids])),
                                               current=float(currents[ids[0]] / turns[ids[0]]), control=False)))
    tokamak = freegs.machine.Machine(coils, wall=freegs.machine.Wall(data['wall_r'].tolist(), data['wall_z'].tolist()))
    grid_r, grid_z = data[f'shot_{i:04d}_grid_r'], data[f'shot_{i:04d}_grid_z']
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=float(grid_r[0]), Rmax=float(grid_r[-1]),
                            Zmin=float(grid_z[0]), Zmax=float(grid_z[-1]), nx=len(grid_r), ny=len(grid_z))
    f_vacuum = MU0 * TF_TURNS * abs(frame['tf_A']) / (2.0 * np.pi)
    profiles = freegs.jtor.ConstrainPaxisIp(eq, frame['p_axis_Pa'], frame['ip_A'], f_vacuum)
    freegs.solve(eq, profiles, constrain=None, show=False, maxits=50, check_limited=True)
    return np.asarray(eq.psi()).T, float(eq.plasmaCurrent())


def main():
    frames = json.loads((HERE / 'm3c0_frames.json').read_text())
    maps, rows = {}, []
    with np.load(HERE / 'm3c0_inputs.npz') as data:
        for frame in frames:
            began, record = time.time(), dict(frame)
            try:
                flux, ip_solved = solve(data, frame)
                maps[f'shot_{frame["shot"]:04d}_frame_{frame["frame"]:03d}'] = flux
                record.update(converged=True, ip_input_A=frame['ip_A'], ip_solved_A=ip_solved,
                              finite=bool(np.isfinite(flux).all()))
            except Exception as error:  # a failed solve is a result of this test
                record.update(converged=False, error=f'{type(error).__name__}: {error}'[:300],
                              trace=traceback.format_exc()[-600:])
            record['seconds'] = time.time() - began
            rows.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'trace'}), flush=True)
    np.savez(HERE / 'm3c0_freegs_maps.npz', **maps)
    (HERE / 'm3c0_freegs_rows.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
