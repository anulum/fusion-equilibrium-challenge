# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M2: physical consistency of the v7 and v8b MAST maps (protocol_m2_final_mast_map_criterion.json)
"""Current closure on the measured plasma current and current outside the boundary, from the maps themselves.
MAST inputs and our own maps only (rule 3).

    PYTHONPATH=R/fec_adapter python m2_physical_consistency.py
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import points_in_poly

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_shot
from fec_adapter.vessel import envelope

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'm1'))
from equilibrium_beta import current_density  # noqa: E402

DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
V7 = ('$LAB/agentic_group_workspace/'
      'second_agent-recovery-20260926/results/joint_submission_56cb/mast_public_test.npz')
V8B = HERE.parent / 'v8b' / 'mast_public_test_v8b.npz'
PILOT = list(range(0, 1206, 30))
MIN_IP = 1.0e5
_R = None


def allowed_region(shot):
    """The thomson_isoflux Geometry interior: vessel envelope without coil cells, eroded one cell."""
    rr, zz = np.meshgrid(shot.r_grid, shot.z_grid)
    step_r, step_z = shot.r_grid[1] - shot.r_grid[0], shot.z_grid[1] - shot.z_grid[0]
    allowed = envelope(shot.source).copy()
    for coil_r, coil_z in zip(shot.coils.r, shot.coils.z, strict=True):
        allowed &= ~((np.abs(rr - coil_r) < 1.5 * step_r) & (np.abs(zz - coil_z) < 1.5 * step_z))
    return ndimage.binary_erosion(allowed, iterations=1)


def measures(reader, shot, region, flux, ip):
    reading = reader.read(flux)
    if reading.contour is None:
        return None
    current = current_density(flux, shot.r_grid, shot.z_grid)
    area = (shot.r_grid[1] - shot.r_grid[0]) * (shot.z_grid[1] - shot.z_grid[0])
    rr, zz = np.meshgrid(shot.r_grid, shot.z_grid)
    c = np.asarray(reading.contour, float)
    inside = points_in_poly(np.column_stack([rr.ravel(), zz.ravel()]), c).reshape(rr.shape)
    total = float(np.sum(current[region]) * area)
    magnitude = float(np.sum(np.abs(current[region])) * area)
    outside = float(np.sum(np.abs(current[region & ~inside])) * area)
    return {'closure': abs(total) / abs(ip), 'outside_share': outside / magnitude if magnitude else float('nan')}


def one_shot(task):
    global _R
    if _R is None:
        _R = Reader('MAST')
    i, path = task
    shot = read_shot(Path(path))
    region = allowed_region(shot)
    with np.load(V7) as a:
        v7 = a[f'shot_{i:04d}_psirz'].astype(np.float64)
    with np.load(V8B) as a:
        v8b = a[f'shot_{i:04d}_psirz'].astype(np.float64)
    rows = []
    for f in range(shot.n_frames):
        ip = float(np.nan_to_num(shot.plasma_current[f]))
        if abs(ip) < MIN_IP or np.array_equal(v7[f], v8b[f]):
            continue
        a7, a8 = measures(_R, shot, region, v7[f], ip), measures(_R, shot, region, v8b[f], ip)
        if a7 is None or a8 is None:
            continue
        rows.append({'shot': i, 'frame': f, 'v7': a7, 'v8b': a8})
    return rows


def main():
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    with Pool(3) as pool:
        rows = [r for part in pool.map(one_shot, [(i, files[i]) for i in PILOT], chunksize=1) for r in part]
    (HERE / 'm2_rows.json').write_text(json.dumps(rows))
    out = {'frames': len(rows), 'shots': len({r['shot'] for r in rows})}
    for arm in ('v7', 'v8b'):
        cl = np.array([r[arm]['closure'] for r in rows])
        os_ = np.array([r[arm]['outside_share'] for r in rows])
        out[arm] = {'median_abs_closure_minus_1': float(np.median(np.abs(cl - 1.0))),
                    'closure_pct_10_50_90': [float(v) for v in np.percentile(cl, [10, 50, 90])],
                    'median_outside_share': float(np.median(os_)),
                    'outside_share_pct_10_50_90': [float(v) for v in np.percentile(os_, [10, 50, 90])]}
    ok_closure = out['v8b']['median_abs_closure_minus_1'] <= out['v7']['median_abs_closure_minus_1'] + 0.01
    ok_outside = out['v8b']['median_outside_share'] <= out['v7']['median_outside_share'] + 0.02
    out['closure_ok'], out['outside_ok'] = bool(ok_closure), bool(ok_outside)
    out['final_mast_map_set'] = 'v8b' if ok_closure and ok_outside else 'v7'
    (HERE / 'm2_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
