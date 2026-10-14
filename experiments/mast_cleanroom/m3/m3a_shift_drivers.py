# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3a: what drives the inboard shift of the v7 MAST plasma (protocol_m3a_shift_drivers.json)
"""MAST inputs and our own v7 maps only (rule 3). Audit rows: ../results/audit_thomson_edge/rows.json."""
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from fec_adapter import machine

HERE = Path(__file__).resolve().parent
ROWS = HERE.parent / 'results' / 'audit_thomson_edge' / 'rows.json'
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
OFFSET, MARGIN, HALF_WINDOW_MS = 0.017, 0.005, 5.0
COLUMNS = ('source', 'efit_times', 'magnetics_time', 'magnetics_sol_current', 'magnetics_plasma_current',
           'magnetics_plasma_current_times')


def drivers_of(path):
    names = pq.read_schema(path).names
    table = pq.read_table(path, columns=[c for c in COLUMNS if c in names])
    row = {c: table.column(c)[0].as_py() for c in table.schema.names}
    frames = np.asarray(row['efit_times'], float)
    t_mag = np.asarray(row['magnetics_time'], float)
    sol = np.asarray(row['magnetics_sol_current'], float) * machine.KILO
    t_ip = machine.plasma_current_times(row)
    ip = np.asarray(row['magnetics_plasma_current'], float) * machine.KILO

    def at(t, base, series):
        ok = np.isfinite(series)
        return np.interp(t, base[ok], series[ok])

    rate = lambda base, series: (at(frames + HALF_WINDOW_MS, base, series)
                                 - at(frames - HALF_WINDOW_MS, base, series)) / (2e-3 * HALF_WINDOW_MS)
    ip_frames = np.abs(at(frames, t_ip, ip))
    live = np.flatnonzero(ip_frames >= 1e5)
    start = frames[live[0]] if live.size else frames[0]
    return {'dsol_dt': rate(t_mag, sol), 'dip_dt': rate(t_ip, np.abs(ip)), 'sol': at(frames, t_mag, sol),
            'ip': ip_frames, 'since_start_ms': frames - start}


def grouped_cv_r2(X, y, groups, alpha=1.0, folds=5):
    keys = np.array([hash(g) % folds for g in groups])
    pred = np.empty_like(y)
    for k in range(folds):
        tr, te = keys != k, keys == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A = (X[tr] - mu) / sd
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
        pred[te] = y[tr].mean() + ((X[te] - mu) / sd) @ w
    return 1.0 - float(np.mean((pred - y) ** 2)) / float(np.var(y))


def main():
    rows = json.loads(ROWS.read_text())
    files = sorted(glob.glob(DATA + '*.parquet'))
    assert len(files) == 1206
    cache, kept = {}, []
    for r in rows:
        if not r.get('contour') or r.get('r_in') is None or r.get('te_in_0.05') is None or r.get('te_out_0.05') is None:
            continue
        ti, to = r['te_in_0.05'], r['te_out_0.05']
        if not (ti > r['cover_in'] + MARGIN and to < r['cover_out'] - MARGIN):
            continue
        if r['shot'] not in cache:
            cache[r['shot']] = drivers_of(files[r['shot']])
        d = cache[r['shot']]
        f = r['frame']
        est_in, est_out = ti - OFFSET, to + OFFSET
        kept.append({'shot': r['shot'], 'frame': f,
                     'shift': 0.5 * ((r['r_in'] - est_in) + (r['r_out'] - est_out)),
                     'width_error': (r['r_out'] - r['r_in']) - (est_out - est_in),
                     **{k: float(v[f]) for k, v in d.items()}})
    names = ('dsol_dt', 'dip_dt', 'sol', 'ip', 'since_start_ms')
    X = np.array([[k[n] for n in names] for k in kept])
    groups = [k['shot'] for k in kept]
    out = {'frames_kept': len(kept), 'shots': len(set(groups)), 'frames_in_audit': len(rows)}
    for target in ('shift', 'width_error'):
        y = np.array([k[target] for k in kept])
        res = {'pct_10_50_90_m': [float(v) for v in np.percentile(y, [10, 50, 90])],
               'spearman': {n: float(spearmanr(X[:, i], y).statistic) for i, n in enumerate(names)},
               'cv_r2_single': {n: grouped_cv_r2(X[:, [i]], y, groups) for i, n in enumerate(names)},
               'cv_r2_ramp_rates': grouped_cv_r2(X[:, [0, 1]], y, groups),
               'cv_r2_joint': grouped_cv_r2(X, y, groups)}
        out[target] = res
    s = out['shift']
    out['verdict'] = ('passive_eddy_model' if s['cv_r2_ramp_rates'] >= 0.30 else
                      'solenoid_treatment' if s['cv_r2_single']['sol'] >= 0.30 else
                      'static_or_profile' if s['cv_r2_joint'] < 0.15 else 'mixed')
    (HERE / 'm3a_rows.json').write_text(json.dumps(kept))
    (HERE / 'm3a_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
