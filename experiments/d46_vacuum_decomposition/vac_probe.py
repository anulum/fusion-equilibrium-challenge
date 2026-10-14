# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D46 probe: is the plasma part of the flux map easier to represent than the whole map?
"""Subtract the coils' exact flux from true DIII-D maps of training shots and compare truncated bases.

Truth only, training shots only (none of the 358 or the 357). Nothing is fitted to inputs here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SAS = Path('$FEC_RUNS')
X = SAS / 'ml350/runs/d39_finisher_20261011'
PKG = SAS / 'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
DATA = Path('$FEC_DATA/data/diii_d_train')
for p in (PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter', PKG / 'entrypoints', Path.home() / 'fusion-equilibrium-challenge/fec_harness'):
    sys.path.insert(0, str(p))
import predict_diii_d as pd
import run_parallel_score as scorer
from fec_adapter.cross_machine.thomson_isoflux import green
from fec_adapter.diii_d_accuracy import x_point
from fec_adapter.diii_d_accuracy.shape_edit import Reader

OUT = Path(sys.argv[1])
cohort = json.loads((X / 'd38/cohort.json').read_text())
train = sorted(cohort['training'])
fit_stems, test_stems = train[::75][:80], train[37::300][:20]
assert not (set(fit_stems) | set(test_stems)) & (set(cohort['development']) | set(cohort['reserved_confirmation']))


def read(stem, step):
    path = DATA / (stem + '.parquet')
    columns = pd.input_columns(dict.fromkeys(pq.read_schema(path).names))
    table = pq.read_table(path, columns=columns + ['efit_psirz'], use_threads=False)
    row = {}
    for name in columns:
        value = table[name][0]
        flat = name.startswith('magnetics_') and str(table.schema.field(name).type) in ('list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
    shot = pd.aligned_shot(row, stem, np.asarray(row['efit_times'], dtype=float))
    psi = np.asarray(table['efit_psirz'][0].as_py(), dtype=float)[::step]
    return shot, psi, shot.coil_current[::step], shot.plasma_current[::step]


shot0, _, _, _ = read(fit_stems[0], 50)
rr, zz = np.meshgrid(shot0.r_grid, shot0.z_grid)
G = np.stack([green(rr, zz, float(r), float(z)) for r, z in zip(shot0.coils.r, shot0.coils.z)])   # filaments x 65 x 65
print('filaments', len(shot0.coils), 'grid', rr.shape, flush=True)


def gather(stems, step):
    maps, vac, ip, entrant = [], [], [], []
    for k, stem in enumerate(stems):
        shot, psi, current, plasma = read(stem, step)
        assert np.allclose(shot.coils.r, shot0.coils.r) and np.allclose(shot.r_grid, shot0.r_grid)
        maps.append(psi); vac.append(np.einsum('tf,fzr->tzr', current, G)); ip.append(plasma); entrant += [k] * len(psi)
    return np.concatenate(maps), np.concatenate(vac), np.concatenate(ip), np.asarray(entrant)


fit_maps, fit_vac, fit_ip, _ = gather(fit_stems, 3)
print('fit frames', len(fit_maps), flush=True)
# convention: one scalar c with psi_true = c * psi_vac + plasma part; estimated on the grid corners, plasma part taken
# as linear in Ip per pixel (far from the plasma its flux is close to that of a current at the axis)
corner = np.zeros((65, 65), bool); corner[:6, :6] = corner[:6, -6:] = corner[-6:, :6] = corner[-6:, -6:] = True
y, v = fit_maps[:, corner], fit_vac[:, corner]
design_ip = np.column_stack([fit_ip, np.ones(len(fit_ip))])
resid = lambda a: a - design_ip @ np.linalg.lstsq(design_ip, a, rcond=None)[0]
c = float(np.sum(resid(v) * resid(y)) / np.sum(resid(v) ** 2))
r2 = 1 - np.sum((resid(y) - c * resid(v)) ** 2) / np.sum(resid(y) ** 2)
print(json.dumps({'vacuum_scale_c': c, 'c_times_2pi': c * 2 * np.pi, 'corner_R2_after_Ip': float(r2)}), flush=True)
test_maps, test_vac, _, entrant = gather(test_stems, 4)
print('test frames', len(test_maps), flush=True)


def basis(values):
    flat = values.reshape(len(values), -1); mean = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt, s


radius, height, coarse, mask = scorer._grid()
local = scorer.local_score
reader = Reader(str(shot0.source))


def boundary(pred, true):
    out, xs = [], []
    for p, t in zip(pred, true):
        a, _ = local.extract_lcfs_with_sign(p.astype(np.float32), radius, height, local.MACHINE, coarse, mask)
        b, _ = local.extract_lcfs_with_sign(t.astype(np.float32), radius, height, local.MACHINE, coarse, mask)
        if a is None or b is None:
            continue
        out.append(local.symmetric_hausdorff(a, np.asarray(b, dtype=float)) / local.major_radius(np.asarray(b, dtype=float)) if hasattr(local, 'major_radius') else local.symmetric_hausdorff(a, np.asarray(b, dtype=float)) / 1.69)
        xp, xt = x_point.x_point_positions(reader, p), x_point.x_point_positions(reader, t)
        d = [np.hypot(xp[i] - xt[i], xp[i + 1] - xt[i + 1]) for i in (0, 2) if np.isfinite(xt[i]) and np.isfinite(xp[i])]
        xs += d
    return float(np.mean(out)), float(np.mean(xs) * 1000) if xs else float('nan'), len(out)


result = {'vacuum_scale_c': c, 'fit_frames': int(len(fit_maps)), 'test_frames': int(len(test_maps)), 'rows': []}
total = basis(fit_maps); plasma = basis(fit_maps - c * fit_vac)
energy = lambda s, k: float(1 - np.sum(s[:k] ** 2) / np.sum(s ** 2))
for k in (8, 16, 32, 45, 64):
    row = {'K': k}
    for name, (mean, vt, s), target, add in (('whole_map', total, test_maps, 0.0), ('plasma_part', plasma, test_maps - c * test_vac, c * test_vac)):
        flat = target.reshape(len(target), -1) - mean
        rebuilt = (flat @ vt[:k].T @ vt[:k] + mean).reshape(target.shape) + add
        err = rebuilt - test_maps
        d, xmm, n = boundary(rebuilt, test_maps)
        row[name] = {'flux_rmse': float(np.sqrt(np.mean(err ** 2))), 'boundary_D': d, 'x_point_mm': xmm, 'frames_with_lcfs': n,
                     'variance_left_in_fit': energy(s, k)}
    result['rows'].append(row)
    print(json.dumps(row), flush=True)
OUT.write_text(json.dumps(result, indent=1))
