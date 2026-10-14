# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-1b steps 3 (v7 baseline) and 4 (v7_coilfree) on the large synthetic testbed (fec venv)
"""Protocols protocol_m3c1b_step3_large_testbed.json and protocol_m3c1b_step4_coilfree.json (both frozen).

    PYTHONPATH=R/fec_adapter python step34_testbed.py
"""
import dataclasses
import glob
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import directed_hausdorff

HERE = Path(__file__).resolve().parent
V7 = HERE.parent.parent / 'v8' / 'entrypoints_v7_copy'
FREE = HERE / 'entrypoints_v7_coilfree'
DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'
_W = {}


def _module(name, directory, alias):
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(alias, directory / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    sys.path.remove(str(directory))
    return mod


def _init():
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    _W['pm'] = _module('predict_mast', V7, 'predict_mast')
    _W['arms'] = {'v7': _module('thomson_isoflux', V7, 'ti_v7'), 'v7_coilfree': _module('thomson_isoflux', FREE, 'ti_free')}
    _W['reader'] = Reader('MAST')
    _W['files'] = sorted(glob.glob(DATA + '*.parquet'))


def crossings(c, z0):
    r, z = c[:, 0], c[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    hit = ((z - z0) * (z2 - z0) <= 0) & (z != z2)
    t = (z0 - z[hit]) / (z2[hit] - z[hit])
    x = r[hit] + t * (r2[hit] - r[hit])
    return (float(x.min()), float(x.max())) if x.size >= 2 else (None, None)


def run(task):
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    from fec_adapter.shot_reader import select_frames
    if not _W:
        _init()
    row, fr, truth = task
    reader = _W['reader']
    rec = {'shot': fr['shot'], 'frame': fr['frame'], 'split': fr['split'], 'included': False}
    if not row.get('converged') or abs(row['ip_ratio'] - 1) > 0.01 or truth is None:
        return rec
    rt = reader.read(truth)
    if rt.contour is None:
        return rec
    ct = np.asarray(rt.contour, float)
    xi, xo = crossings(ct, fr['z0'])
    total = sum(abs(v) for n, v in row['I_measured'].items() if 'p4' in n or 'p5' in n)
    li = rt.values.get('li', np.nan)
    if not (0.5 <= rt.axis[0] <= 1.2 and abs(rt.axis[1]) <= 0.3 and xi is not None
            and 0.12 <= xi <= fr['r_in'] + 0.01 and fr['r_out'] - 0.01 <= xo <= fr['r_out'] + 0.20
            and max(abs(v) for v in row['dI'].values()) / total <= 0.10 and np.isfinite(li) and 0.8 <= li <= 1.3):
        rec['truth_li'] = float(li) if np.isfinite(li) else None
        return rec
    rec.update(included=True, truth_li=float(li), control_rel=max(abs(v) for v in row['dI'].values()) / total)
    shot = select_frames(_W['pm'].read_shot(Path(_W['files'][fr['shot']])), np.array([fr['frame']]))
    columns = np.asarray(shot.coils.columns)
    exact = shot.coil_current.copy()
    for name, change in row['dI'].items():
        mine = columns == name
        exact[0, mine] += change / mine.sum()
    interp = RegularGridInterpolator((reader.z_grid, reader.r_grid), truth)
    pa = float(interp([[rt.axis[1], rt.axis[0]]])[0])
    pb = float(np.mean(interp(np.column_stack([ct[:, 1], ct[:, 0]]))))
    psin = (pa - interp(np.column_stack([shot.thomson.z, shot.thomson.r]))) / (pa - pb)
    real = shot.thomson.te[0]
    fired = np.isfinite(real) & (real > 0)
    o = np.argsort(shot.thomson.r[fired])
    rr, tt = shot.thomson.r[fired][o], real[fired][o]
    along = np.linspace(rr[0], rr[-1], 400)
    w = np.exp(-0.5 * ((along[:, None] - rr[None, :]) / 0.03) ** 2)
    te0 = float(((w * tt).sum(1) / w.sum(1)).max())
    b = np.random.default_rng(1000 * fr['shot'] + fr['frame'] + 7).uniform(1.0, 2.0)
    inside = psin < 1.0
    te = np.where(inside, te0 * np.clip(1 - psin ** 2, 0, None) ** b, te0 * 0.02 * np.exp(-(psin - 1) / 0.03))
    ne = np.where(inside, 3e19 * (1 - 0.8 * psin ** 2), 0.6e19)
    thomson = dataclasses.replace(shot.thomson, te=te[None, :], ne=ne[None, :])
    rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
    rec['truth'] = {'t': [rt.values[n] for n in CONS_SCALARS], 'tsum': float(truth.sum()), 'tsq': float(np.sum(truth ** 2)),
                    'n': truth.size, 'x_in': xi, 'x_out': xo}
    for variant, coils in (('exact', exact), ('measured', shot.coil_current)):
        synthetic = dataclasses.replace(shot, coil_current=coils, thomson=thomson)
        for arm, ti in _W['arms'].items():
            geometry = ti.Geometry.of(synthetic)
            maps, solved = ti.reconstruct(synthetic, geometry, None)
            pred = maps[0].astype(np.float64)
            rp = reader.read(pred)
            res = {'solved': solved[0] is not None, 'sse': float(np.sum((pred - truth) ** 2))}
            if rp.contour is None:
                res.update(D=1.0, p=[np.nan] * 7)
            else:
                c = np.asarray(rp.contour, float)
                pi_, po = crossings(c, fr['z0'])
                res.update(D=max(directed_hausdorff(c, ct)[0], directed_hausdorff(ct, c)[0]) / rgeo,
                           p=[rp.values[n] for n in CONS_SCALARS], x_in=pi_, x_out=po)
            rec[f'{variant}/{arm}'] = res
    return rec


def summarise(recs, variant, arm):
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    use = [r for r in recs if r['included']]
    if not use:
        return None
    n = sum(r['truth']['n'] for r in use)
    m = sum(r['truth']['tsum'] for r in use) / n
    flux = 1 - sum(r[f'{variant}/{arm}']['sse'] for r in use) / (sum(r['truth']['tsq'] for r in use) - n * m * m)
    p = np.asarray([r[f'{variant}/{arm}']['p'] for r in use], float)
    t = np.asarray([r['truth']['t'] for r in use], float)
    r2, bias = {}, {}
    for j, name in enumerate(CONS_SCALARS):
        ok = np.isfinite(t[:, j])
        pj = np.where(np.isfinite(p[ok, j]), p[ok, j], np.nanmean(t[ok, j]))
        r2[name] = float(1 - np.sum((pj - t[ok, j]) ** 2) / np.sum((t[ok, j] - t[ok, j].mean()) ** 2))
        bias[name] = float(np.nanmean(p[ok, j] - t[ok, j]))
    outs = [r[f'{variant}/{arm}'].get('x_out') for r in use]
    return {'frames': len(use), 'solved': int(sum(r[f'{variant}/{arm}']['solved'] for r in use)), 'flux_r2': float(flux),
            'D_mean': float(np.mean([r[f'{variant}/{arm}']['D'] for r in use])),
            'consistency': float(np.mean([max(0.0, v) for v in r2.values()])), 'cons_r2': r2, 'bias': bias,
            'x_out_err_median_cm': 100 * float(np.median([o - r['truth']['x_out'] for o, r in zip(outs, use) if o is not None]))}


def main():
    frames = json.loads((HERE / 'step3_frames.json').read_text())
    rows = {(r['shot'], r['frame']): r for r in json.loads((HERE / 'step3_rows.json').read_text())}
    with np.load(HERE / 'step3_truth.npz') as z:
        tasks = []
        for fr in frames:
            key = f"shot_{fr['shot']:04d}_frame_{fr['frame']:03d}"
            tasks.append((rows[(fr['shot'], fr['frame'])], fr, z[key].astype(np.float64) if key in z.files else None))
    with Pool(4, initializer=_init) as pool:
        recs = pool.map(run, tasks, chunksize=2)
    out = {'frames_total': len(recs), 'included': int(sum(r['included'] for r in recs)),
           'included_by_split': {s: int(sum(r['included'] and r['split'] == s for r in recs)) for s in ('TUNE', 'TEST')}}
    for split in ('TUNE', 'TEST', 'ALL'):
        sel = [r for r in recs if split == 'ALL' or r['split'] == split]
        for variant in ('exact', 'measured'):
            for arm in ('v7', 'v7_coilfree'):
                out[f'{split}/{variant}/{arm}'] = summarise(sel, variant, arm)
    test = [r for r in recs if r['included'] and r['split'] == 'TEST']
    shots = sorted({r['shot'] for r in test})
    idx = {s: i for i, s in enumerate(shots)}
    d_v7 = np.zeros(len(shots)); d_free = np.zeros(len(shots)); cnt = np.zeros(len(shots))
    for r in test:
        i = idx[r['shot']]
        d_v7[i] += r['measured/v7']['D']; d_free[i] += r['measured/v7_coilfree']['D']; cnt[i] += 1
    rng = np.random.default_rng(933)
    wts = np.stack([np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)) for _ in range(20000)])
    boot = (wts @ d_free) / (wts @ cnt) - (wts @ d_v7) / (wts @ cnt)
    tm, te_ = out['TEST/measured/v7'], out['TEST/measured/v7_coilfree']
    xe, xf = out['TEST/exact/v7'], out['TEST/exact/v7_coilfree']
    rule = {'D_rel_change_measured': te_['D_mean'] / tm['D_mean'] - 1, 'P_D_lower_measured': float(np.mean(boot < 0)),
            'kappa_bias_measured': [tm['bias']['kappa'], te_['bias']['kappa']],
            'D_rel_change_exact': xf['D_mean'] / xe['D_mean'] - 1, 'cons_change_exact': xf['consistency'] - xe['consistency']}
    rule['passes'] = bool(rule['D_rel_change_measured'] <= -0.10 and rule['P_D_lower_measured'] >= 0.9
                          and abs(te_['bias']['kappa']) <= abs(tm['bias']['kappa'])
                          and rule['D_rel_change_exact'] <= 0.05 and rule['cons_change_exact'] >= -0.02)
    out['step4_rule'] = rule
    out['step3_usable'] = out['included'] >= 150
    (HERE / 'step34_result.json').write_text(json.dumps({'summary': out, 'rows': recs}, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


if __name__ == '__main__':
    main()
