# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-1b step 8 C (fec venv): real-MAST label-free guards, v7 vs VP_s0.1r
"""Protocol protocol_m3c1b_step6_vessel_mode.json (frozen), step 7 guards = step 5 guards + vessel amplitude guard.

    PYTHONPATH=R/fec_adapter python step7_real_check.py
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / 'm2'))
import step34_testbed as tb  # noqa: E402
from m2_physical_consistency import allowed_region, measures  # noqa: E402
from pilot_m3b import SHARES, _pairs  # noqa: E402

DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
SHOTS = list(range(0, 1206, 30))
_W = {}


def init():
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    _W['pm'] = tb._module('predict_mast', tb.V7, 'predict_mast')
    free = tb._module('thomson_isoflux', HERE / 'entrypoints_v7_vessel', 'ti_vp')
    free.FREE_CIRCUITS = (("magnetics_p4u_current", "magnetics_p4l_current"), ("magnetics_p5u_current", "magnetics_p5l_current"))
    free.FREE_SCALE, free.FREE_FLOOR_SHARE, free.FREE_BOUND = 0.03, 0.1, 0.10
    free.VESSEL_SCALE, free.VESSEL_BOUND, free.SHIFT_RELAXATION = 0.1, 0.5, 0.3
    _W['arms'] = {'v7': tb._module('thomson_isoflux', tb.V7, 'ti_v7'), 'free': free}
    _W['reader'] = Reader('MAST')


def gaps(reader, flux, rr, te, z0):
    from scipy.interpolate import RegularGridInterpolator
    rd = reader.read(flux)
    if rd.contour is None:
        return None
    interp = RegularGridInterpolator((reader.z_grid, reader.r_grid), flux)
    pa = float(interp([[rd.axis[1], rd.axis[0]]])[0])
    c = np.asarray(rd.contour)
    pb = float(np.mean(interp(np.column_stack([c[:, 1], c[:, 0]]))))
    span = pa - pb
    out = {}
    for q, (ri, ro) in _pairs(rr, te).items():
        out[str(q)] = abs(float((interp([[z0, ro]])[0] - interp([[z0, ri]])[0]) / span))
    return out


def one(i):
    if not _W:
        init()
    files = sorted(glob.glob(DATA + '*.parquet'))
    shot = _W['pm'].read_shot(Path(files[i]))
    region = allowed_region(shot)
    columns = np.asarray(shot.coils.columns)
    p45 = np.isin(columns, ["magnetics_p4u_current", "magnetics_p4l_current", "magnetics_p5u_current", "magnetics_p5l_current"])
    results = {}
    for arm, ti in _W['arms'].items():
        geometry = ti.Geometry.of(shot)
        results[arm] = ti.reconstruct(shot, geometry, None)
    o = np.argsort(shot.thomson.r)
    z0 = float(np.median(shot.thomson.z))
    rows = []
    for f in range(shot.n_frames):
        ip = float(np.nan_to_num(shot.plasma_current[f]))
        if abs(ip) < _W['arms']['v7'].MIN_CURRENT:
            continue
        row = {'shot': i, 'frame': f, 'solved': {a: results[a][1][f] is not None for a in results}}
        if all(row['solved'].values()):
            te = shot.thomson.te[f, o]
            ok = np.isfinite(te) & (te > 0)
            for arm in results:
                flux = results[arm][0][f].astype(np.float64)
                row[arm] = {'m2': measures(_W['reader'], shot, region, flux, ip),
                            'gaps': gaps(_W['reader'], flux, shot.thomson.r[o][ok], te[ok], z0) if ok.sum() >= 30 else None}
            shifts = results['free'][1][f].shifts
            scale = abs(float(np.nan_to_num(shot.coil_current[f][p45]).sum())) or 1.0
            row['change_rel'] = float(max(abs(s) for s in shifts[:-1]) / scale) if len(shifts) > 1 else 0.0
            row['vessel_rel'] = float(abs(shifts[-1]) / abs(ip)) if shifts else 0.0
            row['vessel_signed'] = float(shifts[-1] / ip) if shifts else 0.0
            sizes = [abs(float(np.nan_to_num(shot.coil_current[f][np.isin(columns, pair)]).sum())) for pair in _W['arms']['free'].FREE_CIRCUITS]
            sizes = [max(v, 1.0, 0.1 * sizes[0]) for v in sizes]
            row['at_bound'] = bool(any(abs(s) >= 0.0999 * v for s, v in zip(shifts[:-1], sizes))) if shifts else False
            row['vessel_at_bound'] = bool(abs(shifts[-1]) >= 0.499 * abs(ip)) if shifts else False
        rows.append(row)
    return rows


def main():
    with Pool(4, initializer=init) as pool:
        rows = [r for rs in pool.map(one, SHOTS, chunksize=1) for r in rs]
    both = [r for r in rows if all(r['solved'].values()) and r['v7']['m2'] and r['free']['m2']]
    res = {'live_frames': len(rows), 'both_solved_and_read': len(both),
           'solved_share': {a: float(np.mean([r['solved'][a] for r in rows])) for a in ('v7', 'free')}}
    for a in ('v7', 'free'):
        res[f'{a}_closure_err_median'] = float(np.median([abs(r[a]['m2']['closure'] - 1) for r in both]))
        res[f'{a}_outside_median'] = float(np.nanmedian([r[a]['m2']['outside_share'] for r in both]))
        for q in SHARES:
            vals = [r[a]['gaps'][str(q)] for r in both if r[a]['gaps'] and str(q) in r[a]['gaps']]
            res[f'{a}_gap_{q}_median'] = float(np.median(vals)) if vals else None
    ch = [r['change_rel'] for r in both]
    res['change_rel_median'] = float(np.median(ch))
    res['change_at_bound_share'] = float(np.mean([r['at_bound'] for r in both]))
    res['vessel_rel_median'] = float(np.median([r['vessel_rel'] for r in both]))
    res['vessel_signed_median'] = float(np.median([r['vessel_signed'] for r in both]))
    res['vessel_at_bound_share'] = float(np.mean([r['vessel_at_bound'] for r in both]))
    g = {'G1': res['solved_share']['free'] >= res['solved_share']['v7'] - 0.05,
         'G2': res['free_closure_err_median'] <= res['v7_closure_err_median'] + 0.01,
         'G3': res['free_outside_median'] <= res['v7_outside_median'] + 0.01,
         'G4': res['change_rel_median'] <= 0.05 and res['change_at_bound_share'] <= 0.20,
         'G5': res['free_gap_0.05_median'] <= res['v7_gap_0.05_median'] + 0.01,
         'G6': res['vessel_rel_median'] <= 0.3 and res['vessel_at_bound_share'] <= 0.20}
    res['guards'] = {k: bool(v) for k, v in g.items()}
    res['verdict'] = 'recommend_v10_to_entrant' if all(g.values()) else 'not_recommended'
    (HERE / 'step8_real_result.json').write_text(json.dumps({'summary': res, 'rows': rows}, indent=1, default=float))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
