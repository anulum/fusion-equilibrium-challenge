# SCPN Reactor Systems — M3c-1b step 1c (FreeGS venv): step-1b solve with check_limited False (protocol step1c, frozen)
import json, time
from pathlib import Path
import numpy as np, freegs
from freegs.filament_coil import FilamentCoil
HERE = Path(__file__).resolve().parent
MU0, TF_TURNS = 4e-7 * np.pi, 24.0
CONTROL = {'magnetics_p4u_current', 'magnetics_p4l_current', 'magnetics_p5u_current', 'magnetics_p5l_current',
           'magnetics_p6u_current', 'magnetics_p6l_current'}
ARMS = {'s0.5': 0.5, 's1.0': 1.0}
frames = json.loads((HERE / 'step1b_frames.json').read_text())
maps, rows = {}, []
with np.load(HERE / 'step1_inputs.npz') as d:
    for arm, s in ARMS.items():
        for fr in frames:
            rec = {'arm': arm, 'shot': fr['shot'], 'frame': fr['frame']}
            if fr['r_in'] is None or fr['r_out'] is None:
                rec.update(converged=False, error='no markers'); rows.append(rec); print(json.dumps(rec), flush=True); continue
            k = f"shot_{fr['shot']:04d}"
            r, z, g, at = d[k + '_coil_r'], d[k + '_coil_z'], d[k + '_group'], d[k + '_coil_at']
            measured, coils = {}, []
            for c in sorted(set(g.tolist())):
                name = fr['circuits'][c]; measured[name] = float(at[g == c].sum())
                coils.append((name, FilamentCoil(Rfil=r[g == c], Zfil=z[g == c], current=measured[name], turns=1, control=name in CONTROL)))
            tok = freegs.machine.Machine(coils, wall=freegs.machine.Wall(d['wall_r'].tolist(), d['wall_z'].tolist()))
            gr, gz = d[k + '_grid_r'], d[k + '_grid_z']
            eq = freegs.Equilibrium(tokamak=tok, Rmin=float(gr[0]), Rmax=float(gr[-1]), Zmin=float(gz[0]), Zmax=float(gz[-1]), nx=len(gr), ny=len(gz))
            prof = freegs.jtor.ConstrainPaxisIp(eq, s * fr['p_thomson_Pa'], fr['ip_A'], MU0 * TF_TURNS * abs(fr['tf_A']) / (2 * np.pi))
            z0, rm = fr['z0'], 0.5 * (fr['r_in'] + fr['r_out'])
            ctrl = freegs.control.constrain(isoflux=[(fr['r_in'], z0, fr['r_out'], z0), (rm, z0 + 0.5, rm, z0 - 0.5)], gamma=1e-12)
            t0 = time.time()
            try:
                freegs.solve(eq, prof, constrain=ctrl, show=False, maxits=50, check_limited=False)
                maps[f"{arm}_shot_{fr['shot']:04d}"] = np.asarray(eq.psi()).T
                final = {name: float(coil.current) for name, coil in tok.coils}
                rec.update(converged=True, ip_ratio=float(eq.plasmaCurrent()) / fr['ip_A'],
                           dI={n: final[n] - measured[n] for n in CONTROL}, I_measured={n: measured[n] for n in CONTROL},
                           r_in=fr['r_in'], r_out=fr['r_out'], z0=z0)
            except Exception as e:
                rec.update(converged=False, error=f'{type(e).__name__}: {e}'[:200])
            rec['seconds'] = round(time.time() - t0, 1)
            rows.append(rec); print(json.dumps({k2: v for k2, v in rec.items() if k2 not in ('dI', 'I_measured')}), flush=True)
np.savez(HERE / 'step1c_maps.npz', **maps)
(HERE / 'step1c_rows.json').write_text(json.dumps(rows, indent=1))
