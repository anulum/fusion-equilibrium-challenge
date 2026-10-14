# SCPN Reactor Systems — M3c-1b step 3a (fec venv): export inputs + Te-5 % markers of the large testbed frames (inputs only)
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent.parent.parent / 'v8b'))
from m3c0_prepare import smoothed, has_markers, wall, MIN_IP, ELECTRON_VOLT, PRESSURE_FACTOR
from edge_boundary import OFFSET, markers
from fec_adapter.shot_reader import aligned_shot, read_row
files = sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet'))
assert len(files) == 1206
wall_r, wall_z = wall()
arrays, frames = {'wall_r': np.asarray(wall_r), 'wall_z': np.asarray(wall_z)}, []
for i in range(3, 1206, 6):
    path = Path(files[i]); row = read_row(path)
    shot = aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=np.float64))
    o = np.argsort(shot.thomson.r)
    live = [f for f in range(shot.n_frames) if np.isfinite(shot.plasma_current[f]) and abs(shot.plasma_current[f]) >= MIN_IP
            and np.all(np.isfinite(shot.coil_current[f])) and np.isfinite(shot.thomson.te[f, o]).sum() >= 30
            and has_markers(shot.thomson.r[o][np.isfinite(shot.thomson.te[f, o])], shot.thomson.te[f, o][np.isfinite(shot.thomson.te[f, o])])]
    if not live:
        continue
    cols = sorted(set(shot.coils.columns)); group = np.array([cols.index(c) for c in shot.coils.columns])
    key = f'shot_{i:04d}'
    arrays[key + '_coil_r'], arrays[key + '_coil_z'], arrays[key + '_group'] = shot.coils.r, shot.coils.z, group
    arrays[key + '_grid_r'], arrays[key + '_grid_z'] = shot.r_grid, shot.z_grid
    for f in sorted({live[int(q * (len(live) - 1))] for q in (0.33, 0.66)}):
        r, te, ne = shot.thomson.r[o], shot.thomson.te[f, o], shot.thomson.ne[f, o]
        ok = np.isfinite(te) & (te > 0)
        m_in, m_out = markers(r[ok], te[ok])
        if m_in is None or m_out is None:
            continue
        okp = ok & np.isfinite(ne) & (ne > 0)
        _a, pe = smoothed(r[okp], ne[okp] * te[okp] * ELECTRON_VOLT)
        arrays[f'{key}_frame_{f:03d}_coil_at'] = shot.coil_current[f]
        frames.append({'shot': i, 'frame': f, 'ip_A': float(shot.plasma_current[f]), 'tf_A': float(shot.tf_current[f]),
                       'p_thomson_Pa': PRESSURE_FACTOR * float(pe.max()), 'z0': float(np.median(shot.thomson.z)),
                       'r_in': float(m_in - OFFSET), 'r_out': float(m_out + OFFSET), 'circuits': cols,
                       'split': 'TUNE' if (i // 6) % 2 == 0 else 'TEST'})
np.savez(HERE / 'step3_inputs.npz', **arrays)
(HERE / 'step3_frames.json').write_text(json.dumps(frames, indent=1))
print(len(frames), 'frames', flush=True)
