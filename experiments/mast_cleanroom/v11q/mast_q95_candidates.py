# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST q95 candidates C1 (boundary formula) and C2 (blend) on the v11 maps
"""Read q95 off each v11 map a second way: the ITER/Uckan formula on the map's own last closed surface
(the scorer's extract_lcfs), times the DIII-D calibration CAL (label over formula with true DIII-D
boundaries, q95_formula_diiid_check.py: median 0.8085 over 3449 frames of 60 training shots).
B0 = mu0 * 24 * I_TF / (2 pi R0) (TF turns as in v11), Ip from the inputs. Frames without a boundary or
with |Ip| < 1e5 A keep the v11 value. The series is filled in time and given the same 5-frame running
median as v11. C1 = that series; C2 = sqrt(v11 * C1). Flux maps and betaN are copied from v11 bit for bit.

    python mast_q95_candidates.py --v11 ARCHIVE --data-root DIR --out DIR [--workers N]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage

MU0 = 4e-7 * np.pi
CAL = 0.8084701495893001
TF_TURNS = 24.0
MEDIAN_FRAMES = 5
MIN_CURRENT = 1.0e5
_STATE = {}


def formula(a, r0, kappa, delta, b0, ip_ma):
    eps = a / r0
    return (5.0 * a**2 * b0 / (r0 * ip_ma) * (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
            * (1.17 - 0.65 * eps) / (1.0 - eps**2) ** 2)


def steadied(series):
    known = np.isfinite(series)
    if not known.any():
        return series
    frames = np.arange(series.size)
    filled = np.interp(frames, frames[known], series[known])
    return ndimage.median_filter(filled, size=MEDIAN_FRAMES, mode='nearest')


def _init(harness):
    sys.path.insert(0, str(Path(harness) / 'fusion_scoring'))
    from lcfs import extract_lcfs
    _STATE['lcfs'] = extract_lcfs


def shot_q95(task):
    index, parquet, psi, q_v11 = task
    from fec_adapter.shot_reader import aligned_shot, input_columns
    names = pq.read_schema(parquet).names
    cols = input_columns(dict.fromkeys(names))
    row = pq.read_table(parquet, columns=cols).to_pylist()[0]
    shot = aligned_shot(row, Path(parquet).stem, np.asarray(row['efit_times'], float))
    r_grid, z_grid = shot.r_grid, shot.z_grid
    raw = np.full(psi.shape[0], np.nan)
    for t in range(psi.shape[0]):
        ip, itf = float(shot.plasma_current[t]), float(shot.tf_current[t])
        if not (np.isfinite(ip) and np.isfinite(itf)) or abs(ip) < MIN_CURRENT:
            continue
        # the v11 solve puts the axis at the maximum of sign(Ip) * flux
        c = _STATE['lcfs'](psi[t].astype(np.float64), r_grid, z_grid, 'MAST', axis_sign=float(np.sign(ip)))
        if c is None or len(c) < 8:
            continue
        r, z = c[:, 0], c[:, 1]
        rmax, rmin = r.max(), r.min()
        a, r0 = 0.5 * (rmax - rmin), 0.5 * (rmax + rmin)
        if a <= 0 or a >= r0:
            continue
        kappa = (z.max() - z.min()) / (rmax - rmin)
        delta = 0.5 * ((r0 - r[np.argmax(z)]) / a + (r0 - r[np.argmin(z)]) / a)
        b0 = MU0 * TF_TURNS * abs(itf) / (2 * np.pi * r0)
        raw[t] = CAL * formula(a, r0, kappa, delta, b0, abs(ip) / 1e6)
    read = int(np.isfinite(raw).sum())
    raw = np.where(np.isfinite(raw), raw, np.nan)
    c1 = steadied(raw) if read else q_v11.astype(np.float64)
    c1 = np.where(np.isfinite(c1), c1, q_v11)
    c2 = np.sqrt(np.clip(q_v11.astype(np.float64), 1e-6, None) * np.clip(c1, 1e-6, None))
    return index, c1.astype(np.float32), c2.astype(np.float32), read, psi.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v11', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    src = np.load(a.v11, allow_pickle=False)
    keys = list(src.files)
    files = sorted((a.data_root / 'mast_public_test').glob('*.parquet'))
    n = len(keys) // 3
    assert n == len(files) == 1206
    tasks = [(i, str(files[i]), src[f'shot_{i:04d}_psirz'], src[f'shot_{i:04d}_q95']) for i in range(n)]
    c1, c2, read, frames = {}, {}, 0, 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(str(a.harness),)) as pool:
        for i, q1, q2, r, f in pool.map(shot_q95, tasks, chunksize=4):
            c1[i], c2[i] = q1, q2
            read += r
            frames += f
    a.out.mkdir(parents=True, exist_ok=True)
    for name, q in (('C1', c1), ('C2', c2)):
        out = {k: (q[int(k[5:9])] if k.endswith('_q95') else src[k]) for k in keys}
        (a.out / name).mkdir(exist_ok=True)
        np.savez_compressed(a.out / name / 'mast_public_test.npz', allow_pickle=False, **out)
    all_v11 = np.concatenate([src[f'shot_{i:04d}_q95'] for i in range(n)]).astype(float)
    all_c1 = np.concatenate([c1[i] for i in range(n)]).astype(float)
    report = {'frames': frames, 'formula_read_frames': read, 'CAL': CAL, 'TF_TURNS': TF_TURNS,
              'corr_v11_C1': float(np.corrcoef(all_v11, all_c1)[0, 1]),
              'median_v11': float(np.median(all_v11)), 'median_C1': float(np.median(all_c1)),
              'pct_C1': np.percentile(all_c1, [1, 5, 25, 50, 75, 95, 99]).tolist()}
    (a.out / 'candidates_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
