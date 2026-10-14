# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M1c-1: equilibrium against kinetic betaN on true DIII-D maps (protocol_m1c1_*.json)
import glob
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import equilibrium_beta as eb
import m1b_geometry_sensitivity as m
from fec_adapter.cross_machine import map_scalars
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import aligned_shot, read_row
from fec_adapter.split import bucket_of

HERE = Path(__file__).resolve().parent
_R = None


def one_shot(path):
    global _R
    if _R is None:
        _R = Reader('DIII-D')
    for attempt in range(5):
        try:
            row = read_row(Path(path))
            break
        except OSError:
            time.sleep(2.0 * (attempt + 1))
    else:
        return []
    shot = aligned_shot(row, Path(path).stem, np.asarray(row['efit_times'], dtype=np.float64))
    psi = np.asarray(row['efit_psirz'], dtype=np.float64)
    label = np.asarray(row['efit_beta_n'], float)
    out = []
    for frame in range(0, shot.n_frames, m.STRIDE):
        ip = float(shot.plasma_current[frame])
        if not (np.isfinite(ip) and abs(ip) >= m.MIN_IP and np.isfinite(label[frame])):
            continue
        flux = psi[frame]
        reading = _R.read(flux)
        if reading.contour is None or not np.all(np.isfinite(reading.axis)):
            continue
        contour = np.asarray(reading.contour, float)
        axis = (float(reading.axis[0]), float(reading.axis[1]))
        edge = float(np.mean(map_scalars._at(flux, _R.r_grid, _R.z_grid, contour[:, 0], contour[:, 1])))
        r_ = int(np.argmin(np.abs(_R.z_grid - axis[1])))
        c_ = int(np.argmin(np.abs(_R.r_grid - axis[0])))
        near = flux[max(r_ - 1, 0):r_ + 2, max(c_ - 1, 0):c_ + 2]
        axis_flux = float(near.min()) if edge > float(flux[r_, c_]) else float(near.max())
        f_vac = map_scalars.vacuum_f(shot.source, float(shot.tf_current[frame]))
        kin = map_scalars.normalised_beta(flux, _R.r_grid, _R.z_grid, axis, axis_flux, edge, shot.thomson.r,
                                          shot.thomson.z, shot.thomson.te[frame], shot.thomson.ne[frame], ip, f_vac)
        eq, share = eb.equilibrium_beta(flux, _R.r_grid, _R.z_grid, axis, axis_flux, edge, ip, f_vac, axis[0])
        out.append({'shot': shot.name, 'frame': frame, 'label': float(label[frame]), 'kinetic': kin,
                    'equilibrium': eq, 'pressure_share': share})
    return out


def stats(y, p):
    var = float(np.var(y))
    ratio = float(np.median(p / y))
    return {'R2': 1.0 - float(np.mean((p - y) ** 2)) / var, 'median_ratio': ratio,
            'R2_bias_removed': 1.0 - float(np.mean((p / ratio - y) ** 2)) / var, 'corr': float(np.corrcoef(p, y)[0, 1])}


def main():
    files = sorted(f for f in glob.glob(m.DATA + '*.parquet') if bucket_of(Path(f).stem) == 'validation')[:m.SHOTS]
    with Pool(2) as pool:
        rows = [r for part in pool.imap(one_shot, files, chunksize=1) for r in part]
    (HERE / 'm1c1_rows.json').write_text(json.dumps(rows))
    y = np.array([r['label'] for r in rows])
    k = np.array([r['kinetic'] for r in rows])
    e = np.array([r['equilibrium'] for r in rows])
    both = np.isfinite(k) & np.isfinite(e)
    res = {'frames': len(rows), 'frames_both_finite': int(both.sum()), 'finite_share_kinetic': float(np.isfinite(k).mean()),
           'finite_share_equilibrium': float(np.isfinite(e).mean()),
           'kinetic': stats(y[both], k[both]), 'equilibrium': stats(y[both], e[both]),
           'pressure_share_pct_10_50_90': [float(v) for v in np.nanpercentile([r['pressure_share'] for r in rows], [10, 50, 90])]}
    res['candidate'] = bool(res['equilibrium']['R2'] >= res['kinetic']['R2']
                            and res['equilibrium']['R2_bias_removed'] >= res['kinetic']['R2_bias_removed'])
    (HERE / 'm1c1_result.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
