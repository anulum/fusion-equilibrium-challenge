# SCPN Reactor Systems — M3c-1b step 0 (infrastructure): export one frame's coils and the house vacuum flux
import glob, numpy as np
from pathlib import Path
from fec_adapter.shot_reader import read_shot
from fec_adapter.cross_machine.thomson_isoflux import Geometry
files = sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet'))
shot = read_shot(Path(files[301])); f = 16
geo = Geometry.of(shot)
I = np.nan_to_num(shot.coil_current[f])
vac = np.einsum('f,fzr->zr', I, geo.coil_flux)
np.savez(Path(__file__).with_name('step0_frame.npz'), r=shot.coils.r, z=shot.coils.z, I=I,
         group=np.array([sorted(set(shot.coils.columns)).index(c) for c in shot.coils.columns]),
         grid_r=shot.r_grid, grid_z=shot.z_grid, vac=vac, ip=float(shot.plasma_current[f]))
print('exported', I.shape, vac.shape, float(vac.min()), float(vac.max()))
