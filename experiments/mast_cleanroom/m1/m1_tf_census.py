# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M1 label-free census: TF current, vacuum field and plasma current on MAST test inputs
"""Inputs only (the canonical machine.*_si converters on the frame time base; no efit_* label value is read). Vacuum B0 at R = 0.85 m from TF_TURNS[MAST] = 24."""
import glob, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, '$HOME/fusion-equilibrium-challenge/fec_adapter')
from fec_adapter.cross_machine.map_scalars import vacuum_f
from fec_adapter import machine

rows = []
for path in sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet')):
    names = pq.read_schema(path).names
    cols = [c for c in ('source', 'efit_times', 'magnetics_time', 'magnetics_tf_current', 'magnetics_plasma_current',
                        'magnetics_plasma_current_times') if c in names]
    table = pq.read_table(path, columns=cols)
    row = {c: table.column(c)[0].as_py() for c in cols}
    times = np.asarray(row['efit_times'], float)  # the frame time base only; no label value is read
    tf = np.abs(machine.toroidal_current_si(row, times)); ip = np.abs(machine.plasma_current_si(row, times))
    live = np.isfinite(ip) & (ip > 1e5)
    tf_live = tf[live & np.isfinite(tf)]
    rows.append({'shot': path.rsplit('/', 1)[1][:-8], 'frames_live': int(live.sum()),
                 'tf_median': float(np.median(tf_live)) if tf_live.size else float('nan'),
                 'tf_rel_spread': float(np.ptp(tf_live) / np.median(tf_live)) if tf_live.size and np.median(tf_live) > 0 else float('nan'),
                 'ip_max_kA': float(np.nanmax(ip) / 1e3) if np.isfinite(ip).any() else float('nan')})
tf = np.array([r['tf_median'] for r in rows]); ok = np.isfinite(tf)
b0 = np.array([vacuum_f('MAST', x) / 0.85 if np.isfinite(x) else np.nan for x in tf])
pct = lambda a: [round(float(v), 4) for v in np.nanpercentile(a, [0, 1, 10, 50, 90, 99, 100])]
out = {'shots': len(rows), 'tf_finite_shots': int(ok.sum()), 'shots_without_live_frames': int(sum(r['frames_live'] == 0 for r in rows)),
       'tf_median_A_pct_0_1_10_50_90_99_100': pct(tf), 'B0_at_0p85m_T_pct': pct(b0),
       'tf_rel_spread_pct': pct([r['tf_rel_spread'] for r in rows]), 'ip_max_kA_pct': pct([r['ip_max_kA'] for r in rows]),
       'shots_B0_below_0p3T': [r['shot'] for r, b in zip(rows, b0) if b < 0.3], 'shots_B0_above_0p7T': [r['shot'] for r, b in zip(rows, b0) if b > 0.7]}
json.dump({'summary': out, 'rows': rows}, open('m1_tf_census.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
