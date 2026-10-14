# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — R6: do physics features explain the network's li / kappa error? (protocol d068122a)
"""Arms A (367 inputs + 7 map-read scalars) and B (A + physics block P) predict the per-frame residual
truth - OOF map of li and kappa, leave-one-fold-out over the 5 K-fold folds; shot bootstrap of B - A."""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import r6_features as rf

HERE = Path(__file__).resolve().parent
K = HERE.parent.parent / 'kfold'
TABLE = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_training.npz')
CACHE = HERE / 'physics_block.npz'
OUT = HERE / 'r6_result.json'
TARGETS = ('li', 'kappa')
TRUTH = {'li': 'li_scored', 'kappa': 'kappa'}


def main():
    folds = [np.load(K / f'features_fold{k}.npz') for k in range(5)]
    stems = [str(s) for f in folds for s in f['stems']]
    if CACHE.is_file():
        with np.load(CACHE) as c:
            block = {s: c[s] for s in c.files}
    else:
        with Pool(6) as pool:
            block = dict(zip(stems, pool.map(rf.features, stems, chunksize=4)))
        np.savez(CACHE, **block)
    with np.load(TABLE) as t:
        tst = t['stems'].astype(str)
        start = np.concatenate(([0], np.cumsum(np.bincount(t['shot_of_row'], minlength=len(tst)))))
        where = {s: i for i, s in enumerate(tst)}
        names = t['label_names'].astype(str).tolist()
        cols = [names.index(TRUTH[x]) for x in TARGETS]
        rows, parts = [], []
        for k, f in enumerate(folds):
            fs = f['stems'].astype(str)
            r = np.array([start[where[fs[s]]] + fr for s, fr in zip(f['shot_of_row'], f['frame'])])
            rows.append(r)
        truth = t['labels'][np.concatenate(rows)][:, cols].astype(np.float64)
    fold_id = np.concatenate([np.full(len(f['frame']), k) for k, f in enumerate(folds)])
    shot_key = np.concatenate([np.array([f'{f["stems"][s]}' for s in f['shot_of_row']]) for f in folds])
    mc = [str(c) for c in folds[0]['map_columns']]
    mapsc = np.concatenate([f['map_features'][:, [mc.index(c) for c in ('R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li')]] for f in folds])
    mapped = np.column_stack([mapsc[:, 6], mapsc[:, 2]])
    inputs = np.concatenate([f['inputs'] for f in folds]).astype(np.float32)
    phys = np.concatenate([np.stack([block[str(f['stems'][s])][fr] for s, fr in zip(f['shot_of_row'], f['frame'])])
                           for f in folds])
    XA = np.column_stack([inputs, mapsc]).astype(np.float32)
    XB = np.column_stack([XA, phys]).astype(np.float32)
    result = {'frames': int(len(fold_id)), 'shots': len(set(shot_key)), 'physics_names': list(rf.NAMES),
              'physics_finite_share': np.isfinite(phys).mean(0).round(3).tolist()}
    for c, target in enumerate(TARGETS):
        res = truth[:, c] - mapped[:, c]
        ok = np.isfinite(res) & np.isfinite(mapped[:, c])
        preds = {}
        for arm, X in (('A', XA), ('B', XB)):
            p = np.full(len(res), np.nan)
            for k in range(5):
                fit, held = ok & (fold_id != k), ok & (fold_id == k)
                m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                  early_stopping=False, random_state=0)
                p[held] = m.fit(X[fit], res[fit]).predict(X[held])
            preds[arm] = p
        r = res[ok]
        sse0 = float(np.sum(r ** 2))
        out = {'residual_rms': float(np.sqrt(np.mean(r ** 2))), 'truth_var': float(np.var(truth[ok, c]))}
        for arm in ('A', 'B'):
            e = r - preds[arm][ok]
            out[f'{arm}_residual_R2'] = 1 - float(np.sum(e ** 2)) / sse0
        # shot bootstrap of the B - A difference in residual R2
        keys = shot_key[ok]
        uniq, inv = np.unique(keys, return_inverse=True)
        s0 = np.bincount(inv, weights=r ** 2)
        sa = np.bincount(inv, weights=(r - preds['A'][ok]) ** 2)
        sb = np.bincount(inv, weights=(r - preds['B'][ok]) ** 2)
        rng = np.random.default_rng(930)
        diffs = []
        for _ in range(20000):
            i = rng.integers(0, len(uniq), len(uniq))
            diffs.append((sa[i].sum() - sb[i].sum()) / s0[i].sum())
        diffs = np.array(diffs)
        out.update({'B_minus_A': out['B_residual_R2'] - out['A_residual_R2'], 'ci95': np.percentile(diffs, [2.5, 97.5]).tolist(),
                    'P_B_better': float((diffs > 0).mean()),
                    'truth_R2_of_map': 1 - sse0 / float(np.sum((truth[ok, c] - truth[ok, c].mean()) ** 2))})
        result[target] = out
        print(target, json.dumps(out), flush=True)
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
