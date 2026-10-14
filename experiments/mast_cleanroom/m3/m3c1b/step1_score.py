# SCPN Reactor Systems — M3c-1b step 1c (fec venv): sanity per frame and yield per arm (protocol step1, frozen)
import json
from pathlib import Path
import numpy as np
from fec_adapter.diii_d_accuracy.shape_edit import Reader
HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / 'step1_rows.json').read_text())
rd = Reader('MAST'); out = {}
with np.load(HERE / 'step1_maps.npz') as maps:
    for r in rows:
        key = f"{r['arm']}_shot_{r['shot']:04d}"
        sane = False
        if r['converged'] and abs(r['ip_ratio'] - 1) <= 0.01 and key in maps.files:
            reading = rd.read(maps[key].astype(np.float64))
            if reading.contour is not None:
                c = np.asarray(reading.contour); ax = reading.axis
                r.update(axis_r=float(ax[0]), axis_z=float(ax[1]), rmin=float(c[:, 0].min()), rmax=float(c[:, 0].max()),
                         zmin=float(c[:, 1].min()), zmax=float(c[:, 1].max()))
                sane = 0.5 <= ax[0] <= 1.2 and abs(ax[1]) <= 0.3 and c[:, 0].min() >= 0.19 and c[:, 0].max() <= 1.6
        r['sane'] = bool(sane)
        out.setdefault(r['arm'], []).append(r)
summary = {arm: {'frames': len(v), 'converged': sum(x['converged'] for x in v), 'sane': sum(x['sane'] for x in v),
                 'sane_yield': sum(x['sane'] for x in v) / len(v)} for arm, v in out.items()}
viable = [a for a, s in summary.items() if s['sane_yield'] >= 0.6]
result = {'summary': summary, 'viable_arms': viable, 'verdict': 'generator_viable' if viable else 'step1b_vertical_control', 'rows': rows}
(HERE / 'step1_result.json').write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1))
