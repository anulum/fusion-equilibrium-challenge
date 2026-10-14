# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D59: do slow-memory features explain the persistent left-over of the map and the heads?
"""Paired, cross-validated over five groups of whole shots, 358 development shots, boosted trees:
control = 582 features + 64 answered coordinates; treatment = control + history_features. Targets: the three leading
left-over directions of the deployed map (directions fitted on the fit groups) and the residual of five deployed
heads against the scorer's true-map scalars (r_axis, z_axis, kappa, volume, li). Reports the cross-validated share
of each target's left-over energy removed by each arm and the paired difference with a shot bootstrap."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xgboost as xgb

import d39_common as c
import history_features as h

KEPT, GROUPS, TOP, RESAMPLES, SEED = 45, 5, 3, 4000, 5901
HEADS = ('r_axis', 'z_axis', 'kappa', 'volume', 'li_scored')
PARAMS = {'tree_method': 'hist', 'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.5,
          'min_child_weight': 50, 'lambda': 5.0, 'nthread': 20, 'seed': 59}


def extract(case):
    return h.features_of(case['path'])


def main():
    cases = c.cases()
    with ProcessPoolExecutor(16) as pool:
        history = np.concatenate(list(pool.map(extract, cases, chunksize=4)))
    z = np.load(c.X / 'reports/d58_coordinates_358.npz')
    truth, answer, shot = z['truth'].astype(float), z['answer'].astype(float), z['shot']
    assert len(history) == len(truth)
    left = truth.copy()
    left[:, :KEPT] -= answer[:, :KEPT]
    features, head_residual = [], []
    order = ('R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li')
    for case in cases:
        cache = c.load_cache(case['stem'])
        names = cache.label_names.astype(str).tolist(); tree_names = cache.tree_names.astype(str).tolist()
        features.append(cache.features.astype(np.float32))
        reference = np.asarray(c.reference_of(case['stem'])[1], dtype=float)
        block = []
        for name, column in zip(HEADS, (0, 1, 2, 5, 6)):
            base = cache.labels[:, names.index(name)].astype(float)
            if name in c.TREE_WEIGHTS:
                base = base + c.TREE_WEIGHTS[name] * (cache.tree[:, tree_names.index(name)] - base)
            block.append(reference[:, column] - base)
        head_residual.append(np.column_stack(block))
    features, head_residual = np.concatenate(features), np.concatenate(head_residual)
    np.savez_compressed(c.X / 'reports/d59_history_features_358.npz', history=history, shot=shot)
    shots = int(shot.max()) + 1
    group = shot % GROUPS
    arms = {'control': np.concatenate([features, answer.astype(np.float32)], axis=1)}
    arms['treatment'] = np.concatenate([arms['control'], history], axis=1)
    names = [f'map_direction_{k + 1}' for k in range(TOP)] + list(HEADS)
    target = np.zeros((len(left), len(names)))
    predicted = {arm: np.zeros_like(target) for arm in arms}
    gains = {}
    for g in range(GROUPS):
        fit, hold = group != g, group == g
        mean = left[fit].mean(axis=0)
        _, _, vt = np.linalg.svd(left[fit] - mean, full_matrices=False)
        y_fit = np.column_stack([(left[fit] - mean) @ vt[:TOP].T, head_residual[fit]])
        target[hold] = np.column_stack([(left[hold] - mean) @ vt[:TOP].T, head_residual[hold]])
        for arm, x in arms.items():
            dfit = xgb.QuantileDMatrix(x[fit], nthread=20)
            dhold = xgb.QuantileDMatrix(x[hold], ref=dfit, nthread=20)
            for j in range(len(names)):
                ok = np.isfinite(y_fit[:, j])
                dfit.set_label(np.where(ok, y_fit[:, j], 0.0)); dfit.set_weight(ok.astype(np.float32))
                booster = xgb.train(PARAMS, dfit, 400)
                predicted[arm][hold, j] = booster.predict(dhold)
                if arm == 'treatment' and g == 0:
                    score = booster.get_score(importance_type='total_gain')
                    total = sum(score.values())
                    gains[names[j]] = round(sum(v for k, v in score.items() if int(k[1:]) >= arms['control'].shape[1]) / total, 4)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(p, minlength=shots) for p in rng.integers(0, shots, size=(RESAMPLES, shots))]).astype(float)
    report = {'frames': int(len(left)), 'history_columns': int(history.shape[1]), 'targets': {}}
    for j, name in enumerate(names):
        y = target[:, j]
        ok = np.isfinite(y)
        per = lambda v: np.bincount(shot[ok], weights=v[ok], minlength=shots)
        base = per(np.nan_to_num(y) ** 2)
        e = {arm: per((np.nan_to_num(y) - predicted[arm][:, j]) ** 2) for arm in arms}
        removed = {arm: float(1 - e[arm].sum() / base.sum()) for arm in arms}
        boot = (counts @ e['control'] - counts @ e['treatment']) / (counts @ base)
        report['targets'][name] = {'share_removed_control': removed['control'], 'share_removed_treatment': removed['treatment'],
                                   'treatment_minus_control': removed['treatment'] - removed['control'],
                                   'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                   'P_positive': float(np.mean(boot > 0)), 'gain_share_of_history_columns_fold0': gains.get(name)}
    (c.X / 'reports/d59_history_test_358.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    sys.exit(main())
