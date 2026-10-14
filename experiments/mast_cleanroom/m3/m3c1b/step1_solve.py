# SCPN Reactor Systems — M3c-1b step 1b (FreeGS venv): free-boundary solves per pressure arm (protocol step1, frozen)
import json, time, traceback
from pathlib import Path
import numpy as np, freegs
from freegs.filament_coil import FilamentCoil
HERE = Path(__file__).resolve().parent
MU0, TF_TURNS = 4e-7 * np.pi, 24.0
ARMS = {'s0.25': 0.25, 's0.5': 0.5, 's1.0': 1.0, 'fixed3kPa': None}
def solve(d, fr, p_axis):
    k = f"shot_{fr['shot']:04d}"
    r, z, g, at = d[k + '_coil_r'], d[k + '_coil_z'], d[k + '_group'], d[k + '_coil_at']
    coils = [(f'C{c:02d}', FilamentCoil(Rfil=r[g == c], Zfil=z[g == c], current=float(at[g == c].sum()), turns=1, control=False))
             for c in sorted(set(g.tolist()))]
    tok = freegs.machine.Machine(coils, wall=freegs.machine.Wall(d['wall_r'].tolist(), d['wall_z'].tolist()))
    gr, gz = d[k + '_grid_r'], d[k + '_grid_z']
    eq = freegs.Equilibrium(tokamak=tok, Rmin=float(gr[0]), Rmax=float(gr[-1]), Zmin=float(gz[0]), Zmax=float(gz[-1]), nx=len(gr), ny=len(gz))
    prof = freegs.jtor.ConstrainPaxisIp(eq, p_axis, fr['ip_A'], MU0 * TF_TURNS * abs(fr['tf_A']) / (2 * np.pi))
    freegs.solve(eq, prof, constrain=None, show=False, maxits=50, check_limited=True)
    return np.asarray(eq.psi()).T, float(eq.plasmaCurrent())
frames = json.loads((HERE / 'step1_frames.json').read_text())
maps, rows = {}, []
with np.load(HERE / 'step1_inputs.npz') as d:
    for arm, s in ARMS.items():
        for fr in frames:
            p_axis = 3e3 if s is None else s * fr['p_thomson_Pa']
            t0, rec = time.time(), {'arm': arm, 'shot': fr['shot'], 'frame': fr['frame'], 'p_axis_Pa': p_axis}
            try:
                psi, ip = solve(d, fr, p_axis)
                maps[f"{arm}_shot_{fr['shot']:04d}"] = psi
                rec.update(converged=True, ip_ratio=ip / fr['ip_A'])
            except Exception as e:
                rec.update(converged=False, error=f'{type(e).__name__}: {e}'[:200])
            rec['seconds'] = round(time.time() - t0, 1)
            rows.append(rec); print(json.dumps(rec), flush=True)
np.savez(HERE / 'step1_maps.npz', **maps)
(HERE / 'step1_rows.json').write_text(json.dumps(rows, indent=1))
