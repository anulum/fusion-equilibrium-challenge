# M3c-1b step 0b (feasibility, inputs only): vertical field at R0 from PF coils vs from the solenoid, and the Shafranov need
import glob, numpy as np
from pathlib import Path
from fec_adapter.shot_reader import read_shot
from fec_adapter.cross_machine.thomson_isoflux import green
MU0 = 4e-7 * np.pi
files = sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet'))
for i in (0, 301, 603, 904, 1205):
    shot = read_shot(Path(files[i])); f = shot.n_frames // 2
    I = np.nan_to_num(shot.coil_current[f]); sol = np.asarray(shot.coils.columns) == 'magnetics_sol_current'
    R0, a, h = 0.85, 0.55, 1e-4
    def bz(mask):
        psi = lambda R: float(sum(I[k] * green(R, 0.0, shot.coils.r[k], shot.coils.z[k]) for k in np.flatnonzero(mask)))
        return (psi(R0 + h) - psi(R0 - h)) / (2 * h) / R0
    ip = float(shot.plasma_current[f])
    need = MU0 * ip / (4 * np.pi * R0) * (np.log(8 * R0 / a) + 0.3 + 1.0 / 2 - 1.5)  # betap .3, li 1.0 assumed
    print(i, 'Ip kA', round(ip / 1e3), 'Bz PF', round(bz(~sol), 4), 'Bz sol(0.5)', round(bz(sol), 4), 'Shafranov |Bv| need ~', round(need, 4))
