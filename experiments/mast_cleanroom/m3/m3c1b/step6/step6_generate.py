# SCPN Reactor Systems — M3c-1b step 6 (FreeGS venv): step-3 truth set plus a hidden vessel current (protocol step6, frozen)
import json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, freegs
from freegs.filament_coil import FilamentCoil
HERE = Path(__file__).resolve().parent
SRC = Path(__import__('os').environ.get('STEP_SRC', str(HERE.parent)))
MU0, TF_TURNS = 4e-7 * np.pi, 24.0
CONTROL = {'magnetics_p4u_current', 'magnetics_p4l_current', 'magnetics_p5u_current', 'magnetics_p5l_current',
           'magnetics_p6u_current', 'magnetics_p6l_current'}
_D = None
def work(fr):
    global _D
    if _D is None:
        _D = dict(np.load(SRC / 'step3_inputs.npz'))
    d = _D; k = f"shot_{fr['shot']:04d}"
    rng = np.random.default_rng(1000 * fr['shot'] + fr['frame'])
    s, am, an = rng.uniform(0.3, 1.0), rng.uniform(1.5, 3.0), rng.uniform(0.6, 1.6)
    rec = {'shot': fr['shot'], 'frame': fr['frame'], 's': s, 'alpha_m': am, 'alpha_n': an}
    r, z, g = d[k + '_coil_r'], d[k + '_coil_z'], d[k + '_group']; at = d[f"{k}_frame_{fr['frame']:03d}_coil_at"]
    measured, coils = {}, []
    for c in sorted(set(g.tolist())):
        name = fr['circuits'][c]; measured[name] = float(at[g == c].sum())
        coils.append((name, FilamentCoil(Rfil=r[g == c], Zfil=z[g == c], current=measured[name], turns=1, control=name in CONTROL)))
    wr, wz = d['wall_r'], d['wall_z']
    step = np.hypot(np.diff(wr, append=wr[0]), np.diff(wz, append=wz[0]))
    keep = np.arange(0, len(wr), 4)
    share = np.add.reduceat(step, keep) / wr[keep]
    share = share / share.sum()
    u = np.random.default_rng(1000 * fr['shot'] + fr['frame'] + 13).uniform(-0.1, 0.4)
    rec['vessel_share_of_ip'] = u
    for n_, (vr, vz, w) in enumerate(zip(wr[keep], wz[keep], share)):
        coils.append((f'V{n_:02d}', freegs.machine.Coil(float(vr), float(vz), current=float(u * fr['ip_A'] * w), turns=1, control=False)))
    tok = freegs.machine.Machine(coils, wall=freegs.machine.Wall(d['wall_r'].tolist(), d['wall_z'].tolist()))
    gr, gz = d[k + '_grid_r'], d[k + '_grid_z']
    eq = freegs.Equilibrium(tokamak=tok, Rmin=float(gr[0]), Rmax=float(gr[-1]), Zmin=float(gz[0]), Zmax=float(gz[-1]), nx=len(gr), ny=len(gz))
    prof = freegs.jtor.ConstrainPaxisIp(eq, s * fr['p_thomson_Pa'], fr['ip_A'], MU0 * TF_TURNS * abs(fr['tf_A']) / (2 * np.pi), alpha_m=am, alpha_n=an)
    z0, rm = fr['z0'], 0.5 * (fr['r_in'] + fr['r_out'])
    ctrl = freegs.control.constrain(isoflux=[(fr['r_in'], z0, fr['r_out'], z0), (rm, z0 + 0.5, rm, z0 - 0.5)], gamma=1e-12)
    t0 = time.time()
    try:
        freegs.solve(eq, prof, constrain=ctrl, show=False, maxits=50, check_limited=False)
        final = {name: float(coil.current) for name, coil in tok.coils}
        rec.update(converged=True, ip_ratio=float(eq.plasmaCurrent()) / fr['ip_A'], dI={n: final[n] - measured[n] for n in CONTROL},
                   I_measured={n: measured[n] for n in CONTROL}, psi=np.asarray(eq.psi()).T)
    except Exception as e:
        rec.update(converged=False, error=f'{type(e).__name__}: {e}'[:200])
    rec['seconds'] = round(time.time() - t0, 1)
    return rec
if __name__ == '__main__':
    frames = json.loads((SRC / 'step3_frames.json').read_text())
    with Pool(4) as p:
        rows = p.map(work, frames, chunksize=4)
    maps = {f"shot_{r['shot']:04d}_frame_{r['frame']:03d}": r.pop('psi') for r in rows if 'psi' in r}
    np.savez(HERE / 'step6_truth.npz', **maps)
    (HERE / 'step6_rows.json').write_text(json.dumps(rows, indent=1))
    print(sum(r['converged'] for r in rows), 'of', len(rows), 'converged', flush=True)
