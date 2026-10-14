# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST v8 pilot (protocol_v8_pilot_20260930.json): v7 vs v8 on 40 shots, label-free metrics
"""Both arms reconstruct the same 40 public MAST shots from inputs only; metrics use only inputs and our own maps.

    python pilot_v8.py --arm v7|v8
    python pilot_v8.py --compare
"""
import argparse
import glob
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
OUT = HERE / 'pilot'
SHOTS = list(range(0, 1206, 30))
SHARES = (0.85, 0.5, 0.25, 0.10, 0.05)
_MOD = None
_READER = None


def _load(arm):
    d = HERE / ('entrypoints_v7_copy' if arm == 'v7' else 'entrypoints_v8')
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('thomson_isoflux', d / 'thomson_isoflux.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['thomson_isoflux'] = mod
    spec.loader.exec_module(mod)
    pm = importlib.util.spec_from_file_location('predict_mast', d / 'predict_mast.py')
    pmod = importlib.util.module_from_spec(pm)
    sys.modules['predict_mast'] = pmod
    pm.loader.exec_module(pmod)
    return mod, pmod


def _pairs(r, te):
    along = np.linspace(r[0], r[-1], 400)
    w = np.exp(-0.5 * ((along[:, None] - r[None, :]) / 0.03) ** 2)
    s = (w * te).sum(1) / w.sum(1)
    top = int(np.argmax(s))
    if top < 20 or top > 380 or s[top] < 50:
        return {}
    out = {}
    for q in SHARES:
        lv = q * s[top]
        bi, bo = np.flatnonzero(s[:top] < lv), np.flatnonzero(s[top:] < lv)
        if bi.size and bo.size:
            i, j = int(bi[-1]), top + int(bo[0])
            out[q] = (float(np.interp(lv, [s[i], s[i + 1]], [along[i], along[i + 1]])),
                      float(np.interp(lv, [s[j], s[j - 1]], [along[j], along[j - 1]])))
    return out


def _crossings(contour, z0):
    c = np.asarray(contour, float)
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((z - z0) * (z2 - z0) <= 0) & (z != z2)
    t = (z0 - z[hit]) / (z2[hit] - z[hit])
    return r[hit] + t * (r2[hit] - r[hit])


def run_shot(task):
    global _MOD, _READER
    arm, i, path = task
    if _MOD is None:
        _MOD = _load(arm)
        from fec_adapter.diii_d_accuracy.shape_edit import Reader
        _READER = Reader('MAST')
    ti, pm = _MOD
    shot = pm.read_shot(Path(path))
    geometry = ti.Geometry.of(shot)
    maps, solved = ti.reconstruct(shot, geometry, None)
    th = shot.thomson
    o = np.argsort(th.r)
    rr, z0 = th.r[o], float(np.median(th.z))
    live = np.abs(np.nan_to_num(shot.plasma_current)) >= ti.MIN_CURRENT
    rows = []
    for f in range(shot.n_frames):
        res = solved[f]
        row = {'shot': i, 'frame': f, 'live': bool(live[f]), 'solved': res is not None}
        if res is not None:
            row['field'] = float(getattr(res, 'field', 0.0))
            vac = np.einsum('f,fzr->zr', np.nan_to_num(shot.coil_current[f]), geometry.coil_flux)
            dpsi = RegularGridInterpolator((shot.z_grid, shot.r_grid), np.gradient(vac, shot.r_grid, axis=1))
            bz = float(dpsi([[z0, geometry.reference_radius]])[0]) / geometry.reference_radius
            row['field_ratio'] = float(2 * row['field'] / bz) if bz else None
        te = th.te[f, o]
        ok = np.isfinite(te) & (te > 0)
        rd = _READER.read(maps[f].astype(np.float64)) if live[f] else None
        if rd is None or rd.contour is None or ok.sum() < 30:
            rows.append(row)
            continue
        interp = RegularGridInterpolator((_READER.z_grid, _READER.r_grid), maps[f].astype(np.float64))
        pa = float(interp([[rd.axis[1], rd.axis[0]]])[0])
        c = np.asarray(rd.contour)
        pb = float(np.mean(interp(np.column_stack([c[:, 1], c[:, 0]]))))
        span = pa - pb
        for q, (ri, ro) in _pairs(rr[ok], te[ok]).items():
            row[f'x_in_{q}'] = float((pa - interp([[z0, ri]])[0]) / span)
            row[f'x_out_{q}'] = float((pa - interp([[z0, ro]])[0]) / span)
            if q == 0.05:
                x = _crossings(c, z0)
                if x.size >= 2:
                    row['d_out_te5'], row['d_in_te5'] = float(x.max() - ro), float(x.min() - ri)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('v7', 'v8'))
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.arm:
        files = sorted(glob.glob(DATA + '*.parquet'))
        assert len(files) == 1206
        with Pool(6) as p:
            rows = [r for rs in p.map(run_shot, [(a.arm, i, files[i]) for i in SHOTS], chunksize=1) for r in rs]
        (OUT / f'rows_{a.arm}.json').write_text(json.dumps(rows))
        print(a.arm, len(rows), 'frames', flush=True)
    if a.compare:
        A = {(r['shot'], r['frame']): r for r in json.loads((OUT / 'rows_v7.json').read_text())}
        B = {(r['shot'], r['frame']): r for r in json.loads((OUT / 'rows_v8.json').read_text())}
        live = [k for k in A if A[k]['live']]
        res = {'frames_live': len(live),
               'solved_share': {arm: float(np.mean([D[k]['solved'] for k in live])) for arm, D in (('v7', A), ('v8', B))}}
        for q in SHARES:
            ks = [k for k in live if f'x_in_{q}' in A[k] and f'x_in_{q}' in B[k]]
            da = np.array([abs(A[k][f'x_out_{q}'] - A[k][f'x_in_{q}']) for k in ks])
            db = np.array([abs(B[k][f'x_out_{q}'] - B[k][f'x_in_{q}']) for k in ks])
            res[f'abs_pair_gap_{q}'] = {'n': len(ks), 'v7_median': float(np.median(da)), 'v8_median': float(np.median(db)),
                                        'share_v8_lower': float(np.mean(db < da))}
        for side in ('out', 'in'):
            ks = [k for k in live if f'd_{side}_te5' in A[k] and f'd_{side}_te5' in B[k]]
            res[f'd_{side}_te5_median_cm'] = {'v7': 100 * float(np.median([A[k][f'd_{side}_te5'] for k in ks])),
                                             'v8': 100 * float(np.median([B[k][f'd_{side}_te5'] for k in ks]))}
        fr = np.array([abs(B[k]['field_ratio']) for k in live if B[k].get('field_ratio') is not None])
        res['field_ratio_abs'] = {'median': float(np.median(fr)), 'p90': float(np.percentile(fr, 90))}
        g = res
        res['P1'] = (g['abs_pair_gap_0.25']['v8_median'] < g['abs_pair_gap_0.25']['v7_median']
                     and g['abs_pair_gap_0.5']['v8_median'] < g['abs_pair_gap_0.5']['v7_median']
                     and g['abs_pair_gap_0.85']['v8_median'] <= g['abs_pair_gap_0.85']['v7_median'] + 0.005)
        res['P2'] = g['solved_share']['v8'] >= g['solved_share']['v7'] - 0.05
        res['P3'] = g['field_ratio_abs']['median'] <= 0.2
        res['accepted'] = bool(res['P1'] and res['P2'] and res['P3'])
        (OUT / 'compare.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
