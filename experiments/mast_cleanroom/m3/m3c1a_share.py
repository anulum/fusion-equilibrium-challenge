# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-1a (protocol_m3c1a_solenoid_share_label_free.json): solenoid row share arms, label-free
"""Each arm reconstructs the same 41 public MAST shots from inputs only with the v7 entrypoint copy, the solenoid
filaments' currents scaled by share/0.5; metrics use only inputs and our own maps. Derived from pilot_m3b.py.

    python m3c1a_share.py --arm 0.5      (parity gate against ../v8/pilot/rows_v7.json)
    python m3c1a_share.py --compare
"""
import argparse
import dataclasses
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
OUT = HERE / 'm3c1a'
BASELINE = HERE.parent / 'v8' / 'pilot' / 'rows_v7.json'
ARMS = (0.25, 0.5, 0.75, 1.0)
CURRENT_SHARE = 0.5  # the share fec_adapter.machine applies; arms rescale from it
SHOTS = list(range(0, 1206, 30))
SHARES = (0.85, 0.5, 0.25, 0.10, 0.05)
_MOD = None
_READER = None


def _load(arm):
    d = HERE.parent / 'v8' / 'entrypoints_v7_copy'
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
    sol = np.asarray(shot.coils.columns) == 'magnetics_sol_current'
    assert sol.sum() == 656, sol.sum()
    scaled = shot.coil_current.copy()
    scaled[:, sol] *= float(arm) / CURRENT_SHARE
    shot = dataclasses.replace(shot, coil_current=scaled)
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
        row['sol_at'] = float(np.nansum(np.abs(shot.coil_current[f, sol])))
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


def _score(rows):
    return {(r['shot'], r['frame']): r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', type=float, choices=ARMS)
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.arm is not None:
        files = sorted(glob.glob(DATA + '*.parquet'))
        assert len(files) == 1206
        with Pool(4) as p:
            rows = [r for rs in p.map(run_shot, [(a.arm, i, files[i]) for i in SHOTS], chunksize=1) for r in rs]
        (OUT / f'rows_{a.arm}.json').write_text(json.dumps(rows))
        print(a.arm, len(rows), 'frames', flush=True)
        if a.arm == CURRENT_SHARE:
            base, new = _score(json.loads(BASELINE.read_text())), _score(rows)
            assert base.keys() == new.keys(), 'parity: frame sets differ'
            for k in base:
                for f in set(base[k]) | set(new[k]):
                    if f in ('field', 'field_ratio', 'sol_at'):
                        continue
                    x, y = base[k].get(f), new[k].get(f)
                    assert (x == y) or (isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-9), ('parity', k, f, x, y)
            print('parity with rows_v7.json: exact', flush=True)
    if a.compare:
        arms = {s: _score(json.loads((OUT / f'rows_{s}.json').read_text())) for s in ARMS}
        keys = [k for k in arms[0.5] if all(k in arms[s] and arms[s][k]['live'] for s in ARMS)]
        res = {'frames_live': len(keys),
               'solved_share': {str(s): float(np.mean([arms[s][k]['solved'] for k in keys])) for s in ARMS}}
        both = [k for k in keys if all(arms[s][k]['solved'] for s in ARMS)]

        def primary(r):
            gaps = [abs(r[f'x_out_{q}'] - r[f'x_in_{q}']) for q in SHARES if f'x_in_{q}' in r]
            return float(np.mean(gaps)) if gaps else None
        per = {s: {k: primary(arms[s][k]) for k in both} for s in ARMS}
        used = [k for k in both if all(per[s][k] is not None for s in ARMS)]
        shots = sorted({k[0] for k in used})
        by_shot = {s: np.array([np.median([per[s][k] for k in used if k[0] == i]) for i in shots]) for s in ARMS}
        res.update(frames_compared=len(used), shots_compared=len(shots),
                   score={str(s): float(np.median(by_shot[s])) for s in ARMS})
        rng = np.random.default_rng(931)
        draws = rng.integers(0, len(shots), size=(20000, len(shots)))

        def p_lower(a_, b_):
            return float(np.mean(np.median(by_shot[a_][draws], axis=1) < np.median(by_shot[b_][draws], axis=1)))
        res['P_lower'] = {f'{x}<{y}': p_lower(x, y) for x in ARMS for y in ARMS if x != y}
        for q in SHARES:
            ks = [k for k in both if all(f'x_in_{q}' in arms[s][k] for s in ARMS)]
            res[f'median_gap_{q}'] = {str(s): float(np.median([abs(arms[s][k][f'x_out_{q}'] - arms[s][k][f'x_in_{q}']) for k in ks])) for s in ARMS}
        cut = np.median([arms[0.5][k]['sol_at'] for k in used])
        hi = [k for k in used if arms[0.5][k]['sol_at'] >= cut]
        res['diagnostic_upper_half_solenoid'] = {'frames': len(hi), 'median_primary': {
            str(s): float(np.median([per[s][k] for k in hi])) for s in ARMS}}
        best = min(ARMS, key=lambda s: res['score'][str(s)])
        guard = {str(s): res['solved_share'][str(s)] >= res['solved_share']['0.5'] - 0.10 for s in ARMS}
        P = res['P_lower']
        if best == 0.5 and P['0.5<1.0'] >= 0.9 and P['0.5<0.25'] >= 0.8 and P['0.5<0.75'] >= 0.8:
            verdict = 'supports_0.5'
        elif best == 1.0 and P['1.0<0.5'] >= 0.9 and guard['1.0']:
            verdict = 'supports_1.0'
        else:
            verdict = 'inconclusive'
        res.update(best=best, guard=guard, verdict=verdict)
        (OUT / 'compare.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
