# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — sanity check of the ITER/Uckan q95 formula on DIII-D labels (allowed data)
"""q95 = 5 a^2 B0 / (R0 Ip[MA]) * (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)) / 2 * (1.17 - 0.65 eps) / (1 - eps^2)^2
(Uckan and the ITER Physics Basis, Nucl. Fusion 39 (1999) 2137, eq. for q95 with the low-aspect-ratio factor),
with a, R0, kappa, delta = (delta_top + delta_bot)/2 from the EFIT boundary, B0 = mu0 N I_TF / (2 pi R0).
Reported against efit_q95 on DIII-D training shots, every 4th frame. DIII-D only; nothing here reads MAST.

    PYTHONPATH=R/fec_adapter python q95_formula_diiid_check.py [n_shots]
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fec_adapter.shot_reader import aligned_shot, input_columns

MU0 = 4e-7 * np.pi
DATA = '$FEC_DATA/data/diii_d_train'


def formula(a, r0, kappa, delta, b0, ip_ma):
    eps = a / r0
    return (5.0 * a**2 * b0 / (r0 * ip_ma) * (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
            * (1.17 - 0.65 * eps) / (1.0 - eps**2) ** 2)


def geometry(r, z):
    rmax, rmin = r.max(), r.min()
    a, r0 = 0.5 * (rmax - rmin), 0.5 * (rmax + rmin)
    kappa = (z.max() - z.min()) / (rmax - rmin)
    delta = 0.5 * ((r0 - r[np.argmax(z)]) / a + (r0 - r[np.argmin(z)]) / a)
    return a, r0, kappa, delta


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    files = sorted(glob.glob(f'{DATA}/*.parquet'), key=lambda f: hashlib.sha256(Path(f).stem.encode()).hexdigest())[:n]
    pred, true = [], []
    for f in files:
        names = pq.read_schema(f).names
        cols = input_columns(dict.fromkeys(names))
        t = pq.read_table(f, columns=cols + ['efit_q95', 'efit_lcfs_n', 'efit_lcfs_r', 'efit_lcfs_z']).to_pylist()[0]
        shot = aligned_shot({k: t[k] for k in cols}, Path(f).stem, np.asarray(t['efit_times'], float))
        q = np.asarray(t['efit_q95'], float)
        for i in range(0, shot.n_frames, 4):
            k = int(t['efit_lcfs_n'][i])
            r = np.asarray(t['efit_lcfs_r'][i][:k], float)
            z = np.asarray(t['efit_lcfs_z'][i][:k], float)
            ip, itf = float(shot.plasma_current[i]), float(shot.tf_current[i])
            if k < 8 or not np.isfinite(q[i]) or abs(ip) < 2e5 or not np.isfinite(itf):
                continue
            a, r0, kappa, delta = geometry(r, z)
            b0 = MU0 * 144.0 * abs(itf) / (2 * np.pi * r0)
            pred.append(formula(a, r0, kappa, delta, b0, abs(ip) / 1e6))
            true.append(abs(q[i]))
    p, y = np.asarray(pred), np.asarray(true)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    out = {'shots': n, 'frames': int(y.size), 'r2': float(r2), 'corr': float(np.corrcoef(p, y)[0, 1]),
           'median_ratio_label_over_formula': float(np.median(y / p)),
           'quartiles_ratio': np.percentile(y / p, [25, 75]).tolist()}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
