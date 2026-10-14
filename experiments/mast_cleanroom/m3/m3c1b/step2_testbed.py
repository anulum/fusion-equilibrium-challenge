# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-1b step 2 (fec venv): v7 on synthetic MAST inputs vs the FreeGS synthetic truth
"""Protocol protocol_m3c1b_step2_testbed.json (frozen). Inputs only plus our own synthetic truth.

    PYTHONPATH=R/fec_adapter python step2_testbed.py
"""
import dataclasses
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import directed_hausdorff

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.scoring_kit.common import CONS_SCALARS
from fec_adapter.shot_reader import select_frames

HERE = Path(__file__).resolve().parent
V7 = HERE.parent.parent / 'v8' / 'entrypoints_v7_copy'
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
OBSERVED_PUBLIC_V7 = {'flux_r2': 0.9805, 'D': 0.1884, 'consistency': 0.2732}


def load(name):
    sys.path.insert(0, str(V7))
    spec = importlib.util.spec_from_file_location(name, V7 / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def crossings(c, z0):
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((z - z0) * (z2 - z0) <= 0) & (z != z2)
    t = (z0 - z[hit]) / (z2[hit] - z[hit])
    x = r[hit] + t * (r2[hit] - r[hit])
    return (float(x.min()), float(x.max())) if x.size >= 2 else (None, None)


def main():
    ti, pm = load('thomson_isoflux'), load('predict_mast')
    rows1c = json.loads((HERE / 'step1c_rows.json').read_text())
    frames = {f['shot']: f for f in json.loads((HERE / 'step1b_frames.json').read_text())}
    files = sorted(glob.glob(DATA + '*.parquet'))
    reader = Reader('MAST')
    out, scored = [], []
    with np.load(HERE / 'step1c_maps.npz') as maps:
        for r in rows1c:
            if r['arm'] != 's0.5':
                continue
            rec = {'shot': r['shot'], 'frame': r['frame']}
            key = f"s0.5_shot_{r['shot']:04d}"
            fr = frames[r['shot']]
            # inclusion filter (protocol): plausible diverted truth with small control
            ok = r.get('converged') and abs(r['ip_ratio'] - 1) <= 0.01 and key in maps.files
            truth = maps[key].astype(np.float64) if ok else None
            rt = reader.read(truth) if ok else None
            if ok and rt.contour is not None:
                ct = np.asarray(rt.contour, float)
                xi, xo = crossings(ct, fr['z0'])
                total = sum(abs(v) for n, v in r['I_measured'].items() if 'p4' in n or 'p5' in n)
                ok = (0.5 <= rt.axis[0] <= 1.2 and abs(rt.axis[1]) <= 0.3 and xi is not None
                      and 0.12 <= xi <= fr['r_in'] + 0.01 and fr['r_out'] - 0.01 <= xo <= fr['r_out'] + 0.20
                      and max(abs(v) for v in r['dI'].values()) / total <= 0.10)
            else:
                ok = False
            rec['included'] = bool(ok)
            if not ok:
                out.append(rec)
                continue
            shot = select_frames(pm.read_shot(Path(files[r['shot']])), np.array([r['frame']]))
            coils = shot.coil_current.copy()
            columns = np.asarray(shot.coils.columns)
            for name, change in r['dI'].items():
                mine = columns == name
                coils[0, mine] += change / mine.sum()
            # synthetic Thomson from the truth (fixed profile, protocol)
            interp = RegularGridInterpolator((reader.z_grid, reader.r_grid), truth)
            pa = float(interp([[rt.axis[1], rt.axis[0]]])[0])
            pb = float(np.mean(interp(np.column_stack([ct[:, 1], ct[:, 0]]))))
            ch_r, ch_z = shot.thomson.r, shot.thomson.z
            psin = (pa - interp(np.column_stack([ch_z, ch_r]))) / (pa - pb)
            real_te = shot.thomson.te[0]
            fired = np.isfinite(real_te) & (real_te > 0)
            order = np.argsort(ch_r[fired])
            along = np.linspace(ch_r[fired][order][0], ch_r[fired][order][-1], 400)
            w = np.exp(-0.5 * ((along[:, None] - ch_r[fired][order][None, :]) / 0.03) ** 2)
            te0 = float(((w * real_te[fired][order]).sum(1) / w.sum(1)).max())
            inside = psin < 1.0
            te = np.where(inside, te0 * np.clip(1 - psin ** 2, 0, None) ** 1.5, te0 * 0.02 * np.exp(-(psin - 1) / 0.03))
            ne = np.where(inside, 3e19 * (1 - 0.8 * psin ** 2), 0.6e19)
            thomson = dataclasses.replace(shot.thomson, te=te[None, :], ne=ne[None, :])
            synthetic = dataclasses.replace(shot, coil_current=coils, thomson=thomson)
            geometry = ti.Geometry.of(synthetic)
            result, solved = ti.reconstruct(synthetic, geometry, None)
            pred = result[0].astype(np.float64)
            rec['solved'] = solved[0] is not None
            rp = reader.read(pred)
            rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
            if rp.contour is None:
                d, vals = 1.0, [np.nan] * 7
            else:
                c = np.asarray(rp.contour, float)
                d = max(directed_hausdorff(c, ct)[0], directed_hausdorff(ct, c)[0]) / rgeo
                vals = [rp.values[n] for n in CONS_SCALARS]
                rec.update(pred_axis=list(rp.axis), pred_rmin=float(c[:, 0].min()), pred_rmax=float(c[:, 0].max()),
                           pred_zmin=float(c[:, 1].min()), pred_zmax=float(c[:, 1].max()))
            rec.update(D=d, p=vals, t=[rt.values[n] for n in CONS_SCALARS], truth_axis=list(rt.axis),
                       truth_rmin=float(ct[:, 0].min()), truth_rmax=float(ct[:, 0].max()),
                       truth_zmin=float(ct[:, 1].min()), truth_zmax=float(ct[:, 1].max()),
                       sse=float(np.sum((pred - truth) ** 2)), tsum=float(truth.sum()), tsq=float(np.sum(truth ** 2)),
                       n=truth.size)
            out.append(rec)
            scored.append(rec)
            print(json.dumps({k: rec[k] for k in ('shot', 'solved', 'D')}), flush=True)
    n = sum(r['n'] for r in scored)
    m = sum(r['tsum'] for r in scored) / n
    flux = 1 - sum(r['sse'] for r in scored) / (sum(r['tsq'] for r in scored) - n * m * m)
    p, t = np.asarray([r['p'] for r in scored], float), np.asarray([r['t'] for r in scored], float)
    r2 = {}
    for j, name in enumerate(CONS_SCALARS):
        good = np.isfinite(t[:, j])
        pj = np.where(np.isfinite(p[good, j]), p[good, j], np.nanmean(t[good, j]))
        r2[name] = float(1 - np.sum((pj - t[good, j]) ** 2) / np.sum((t[good, j] - t[good, j].mean()) ** 2))
    summary = {'frames_included': len(scored), 'frames_total': len(out),
               'solved': int(sum(r['solved'] for r in scored)), 'flux_r2': float(flux),
               'D_mean': float(np.mean([r['D'] for r in scored])), 'D_median': float(np.median([r['D'] for r in scored])),
               'consistency': float(np.mean([max(0.0, v) for v in r2.values()])), 'cons_r2': r2,
               'observed_public_v7_for_description_only': OBSERVED_PUBLIC_V7}
    (HERE / 'step2_result.json').write_text(json.dumps({'summary': summary, 'rows': out}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
