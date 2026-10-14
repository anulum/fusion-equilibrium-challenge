# SCPN Reactor Systems — M3c-1b step 1a (fec venv): export the pilot frames' solver inputs (inputs only)
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m3c0_prepare import smoothed, has_markers, wall, MIN_IP, ELECTRON_VOLT, PRESSURE_FACTOR
from fec_adapter.shot_reader import aligned_shot, read_row
HERE = Path(__file__).resolve().parent
files = sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet'))
assert len(files) == 1206
wall_r, wall_z = wall()
arrays, frames = {'wall_r': np.asarray(wall_r), 'wall_z': np.asarray(wall_z)}, []
for i in range(0, 1206, 60):
    path = Path(files[i]); row = read_row(path)
    shot = aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=np.float64))
    o = np.argsort(shot.thomson.r)
    live = [f for f in range(shot.n_frames) if np.isfinite(shot.plasma_current[f]) and abs(shot.plasma_current[f]) >= MIN_IP
            and np.all(np.isfinite(shot.coil_current[f])) and np.isfinite(shot.thomson.te[f, o]).sum() >= 30
            and has_markers(shot.thomson.r[o][np.isfinite(shot.thomson.te[f, o])], shot.thomson.te[f, o][np.isfinite(shot.thomson.te[f, o])])]
    if not live:
        continue
    f = live[int(0.5 * (len(live) - 1))]
    cols = sorted(set(shot.coils.columns)); group = np.array([cols.index(c) for c in shot.coils.columns])
    te, ne = shot.thomson.te[f, o], shot.thomson.ne[f, o]
    ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    _a, pe = smoothed(shot.thomson.r[o][ok], ne[ok] * te[ok] * ELECTRON_VOLT)
    key = f'shot_{i:04d}'
    arrays[key + '_coil_r'], arrays[key + '_coil_z'], arrays[key + '_group'] = shot.coils.r, shot.coils.z, group
    arrays[key + '_coil_at'] = shot.coil_current[f]
    arrays[key + '_grid_r'], arrays[key + '_grid_z'] = shot.r_grid, shot.z_grid
    frames.append({'shot': i, 'stem': path.stem, 'frame': f, 'ip_A': float(shot.plasma_current[f]), 'tf_A': float(shot.tf_current[f]),
                   'p_thomson_Pa': PRESSURE_FACTOR * float(pe.max())})
np.savez(HERE / 'step1_inputs.npz', **arrays)
(HERE / 'step1_frames.json').write_text(json.dumps(frames, indent=1))
print(len(frames), 'frames', flush=True)
