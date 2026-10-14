# SCPN Reactor Systems — M3c-1b step 1b-c (fec venv): sanity, control size and yield per arm (protocol step1b, frozen)
import json
from pathlib import Path
import numpy as np
from fec_adapter.diii_d_accuracy.shape_edit import Reader
HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / 'step1b_rows.json').read_text())
rd = Reader('MAST'); out = {}
def crossings(c, z0):
    r, z = c[:, 0], c[:, 1]; r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((z - z0) * (z2 - z0) <= 0) & (z != z2); t = (z0 - z[hit]) / (z2[hit] - z[hit])
    x = r[hit] + t * (r2[hit] - r[hit]); return (x.min(), x.max()) if x.size >= 2 else (None, None)
with np.load(HERE / 'step1b_maps.npz') as maps:
    for r in rows:
        key = f"{r['arm']}_shot_{r['shot']:04d}"; sane = False
        if r.get('converged') and abs(r['ip_ratio'] - 1) <= 0.01 and key in maps.files:
            reading = rd.read(maps[key].astype(np.float64))
            if reading.contour is not None:
                c = np.asarray(reading.contour); ax = reading.axis; xi, xo = crossings(c, r['z0'])
                r.update(axis_r=float(ax[0]), axis_z=float(ax[1]), rmin=float(c[:, 0].min()), rmax=float(c[:, 0].max()),
                         zmin=float(c[:, 1].min()), zmax=float(c[:, 1].max()), x_in=xi, x_out=xo)
                total = sum(abs(r['I_measured'][n]) for n in r['I_measured'] if 'p4' in n or 'p5' in n)
                r['control_rel'] = max(abs(v) for v in r['dI'].values()) / total
                sane = (0.5 <= ax[0] <= 1.2 and abs(ax[1]) <= 0.3 and c[:, 0].min() >= 0.19 and c[:, 0].max() <= 1.6
                        and xi is not None and abs(xi - r['r_in']) <= 0.03 and abs(xo - r['r_out']) <= 0.03)
        r['sane'] = bool(sane); out.setdefault(r['arm'], []).append(r)
summary = {}
for arm, v in out.items():
    s = [x for x in v if x['sane']]
    summary[arm] = {'frames': len(v), 'converged': sum(bool(x.get('converged')) for x in v), 'sane': len(s),
                    'sane_yield': len(s) / len(v),
                    'median_control_rel': float(np.median([x['control_rel'] for x in s])) if s else None}
viable = [a for a, s in summary.items() if s['sane_yield'] >= 0.6 and s['median_control_rel'] is not None and s['median_control_rel'] <= 0.10]
res = {'summary': summary, 'viable_arms': viable, 'verdict': 'generator_viable' if viable else 'not_viable', 'rows': rows}
(HERE / 'step1b_result.json').write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))
