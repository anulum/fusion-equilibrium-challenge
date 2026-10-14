# SCPN Reactor Systems — M3c-1b step 0 (infrastructure): FreeGS vacuum flux of the same coils vs the house vacuum
import numpy as np, freegs
from pathlib import Path
from freegs.filament_coil import FilamentCoil
d = np.load(Path(__file__).with_name('step0_frame.npz'))
r, z, I, g = d['r'], d['z'], d['I'], d['group']
gr, gz = d['grid_r'], d['grid_z']
RR, ZZ = np.meshgrid(gr, gz)  # (Z, R) order as the house map
point = freegs.machine.Machine([(f'F{k}', freegs.machine.Coil(float(r[k]), float(z[k]), current=float(I[k]), turns=1, control=False)) for k in range(len(r))])
psi_point = point.psi(RR, ZZ)
grouped = []
for k in sorted(set(g.tolist())):
    ids = np.flatnonzero(g == k)
    grouped.append((f'C{k}', FilamentCoil(Rfil=r[ids], Zfil=z[ids], current=float(I[ids].sum()), turns=1, control=False)))
psi_group = freegs.machine.Machine(grouped).psi(RR, ZZ)
vac = d['vac']; big = np.abs(vac) > 1e-3
for name, p in (('point', psi_point), ('grouped_sum', psi_group)):
    print(name, 'max|diff|', float(np.abs(p - vac).max()), 'median ratio', float(np.median(p[big] / vac[big])),
          'corr', float(np.corrcoef(p.ravel(), vac.ravel())[0, 1]))
