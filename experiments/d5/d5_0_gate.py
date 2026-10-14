# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D5-0: does the discharge history predict the OOF residuals? (protocol_d5_0, frozen)
"""Nested by fold over the 5 OOF folds: same-frame model M0 against history model M1, per residual target.

    OMP_NUM_THREADS=6 python d5_0_gate.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
K = HERE.parent / 'kfold'
R8 = HERE.parent / 'reeval' / 'r8' / 'results'
NAMES = ('R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li', 'q95', 'betaN')
LAGS = (-8, -4, -2, -1, 1, 2, 4, 8)
SEED = 932


def load(k):
    z = np.load(K / f'features_fold{k}.npz')
    rows = json.loads((R8 / f'rows_V0_fold{k}.json').read_text())
    p = np.array([r['p'] for r in rows], float)
    t = np.array([r['t'] for r in rows], float)
    q = np.array([r['q'] for r in rows], float)
    target = np.column_stack([t - p, q[:, 1] - q[:, 0], q[:, 3] - q[:, 2]])
    shot = np.array([f'{k}:{s}' for s in z['shot_of_row']])
    return {'inputs': z['inputs'].astype(np.float64), 'map': z['map_features'], 'p': p, 'target': target,
            'shot': shot, 'frame': z['frame']}


def history(pc, p, shot, frame):
    base = np.column_stack([pc, p])
    out = [np.zeros((len(base), 0))]
    starts = {}
    for i, s in enumerate(shot):
        starts.setdefault(s, []).append(i)
    lagged = np.full((len(base), len(LAGS) * base.shape[1]), np.nan)
    shot_mean = np.empty_like(base)
    run_mean = np.empty_like(base)
    position = np.empty(len(base))
    for s, idx in starts.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(frame[idx])]
        b = base[idx]
        for j, lag in enumerate(LAGS):
            sh = np.full_like(b, np.nan)
            if lag > 0:
                sh[:-lag] = b[lag:]
            else:
                sh[-lag:] = b[:lag]
            lagged[idx, j * base.shape[1]:(j + 1) * base.shape[1]] = sh
        shot_mean[idx] = np.nanmean(b, axis=0)
        finite = np.where(np.isfinite(b), b, 0.0)
        count = np.cumsum(np.isfinite(b), axis=0)
        run_mean[idx] = np.cumsum(finite, axis=0) / np.maximum(count, 1)
        position[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
    out = [lagged, shot_mean, run_mean, position[:, None]]
    return np.column_stack(out)


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)


def main():
    folds = [load(k) for k in range(5)]
    pred = {m: [np.full(f['target'].shape, np.nan) for f in folds] for m in ('M0', 'M1')}
    for k in range(5):
        train = [folds[j] for j in range(5) if j != k]
        x_in = np.concatenate([f['inputs'] for f in train])
        mu, sd = x_in.mean(0), x_in.std(0) + 1e-12
        pca = PCA(n_components=32, random_state=0).fit((x_in - mu) / sd)
        sets = {}
        for name, fs in (('train', train), ('test', [folds[k]])):
            m0, m1, y = [], [], []
            for f in fs:
                pc = pca.transform((f['inputs'] - mu) / sd)
                same = np.column_stack([pc, f['map']])
                m0.append(same)
                m1.append(np.column_stack([same, history(pc, f['p'], f['shot'], f['frame'])]))
                y.append(f['target'])
            sets[name] = (np.concatenate(m0), np.concatenate(m1), np.concatenate(y))
        for t in range(len(NAMES)):
            ok = np.isfinite(sets['train'][2][:, t])
            for m, col in (('M0', 0), ('M1', 1)):
                reg = model().fit(sets['train'][col][ok], sets['train'][2][ok, t])
                pred[m][k][:, t] = reg.predict(sets['test'][col])
            print(json.dumps({'fold': k, 'target': NAMES[t]}), flush=True)
    target = np.concatenate([f['target'] for f in folds])
    shots = np.concatenate([f['shot'] for f in folds])
    p0, p1 = np.concatenate(pred['M0']), np.concatenate(pred['M1'])
    ids, inv = np.unique(shots, return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(20000)])
    result = {'protocol_sha256_prefix': 'ecbd00a5', 'frames': int(len(target)), 'shots': int(len(ids)), 'targets': {}}
    passed = 0
    for t, name in enumerate(NAMES):
        ok = np.isfinite(target[:, t])
        r = target[ok, t]
        sse = {m: np.bincount(inv[ok], (r - p[ok, t]) ** 2, minlength=len(ids)) for m, p in (('M0', p0), ('M1', p1))}
        ref = np.bincount(inv[ok], r ** 2, minlength=len(ids))
        r2 = {m: float(1 - sse[m].sum() / ref.sum()) for m in sse}
        boot = 1 - (weights @ sse['M1']) / (weights @ ref) - (1 - (weights @ sse['M0']) / (weights @ ref))
        gain = r2['M1'] - r2['M0']
        p_pos = float(np.mean(boot > 0))
        ok_rule = bool(gain >= 0.05 and p_pos >= 0.9 and r2['M1'] > 0)
        passed += ok_rule
        result['targets'][name] = {'R2_M0': r2['M0'], 'R2_M1': r2['M1'], 'gain': gain,
                                   'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                   'P_gain_positive': p_pos, 'meets_rule': ok_rule}
    result['targets_meeting_rule'] = passed
    result['verdict'] = 'D5_justified' if passed >= 2 else 'D5_closed'
    (HERE / 'd5_0_result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
