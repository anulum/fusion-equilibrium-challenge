# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D58b: can the leading directions of the left-over map be estimated from the inputs?
"""Measurement only. Leading principal directions of the deployed left-over (inside the 128-direction basis) on the
358 development shots; for each: share of the left-over energy, per-shot offset share, lag-1 autocorrelation, and the
cross-validated (five groups of whole shots) share explained by (a) ridge on the 582 features, (b) boosted trees on
the 582 features and the 64 answered coordinates. Directions are fitted on the fit groups only."""
import json

import numpy as np
import xgboost as xgb

import d39_common as c

KEPT, GROUPS, TOP = 45, 5, 6


def main():
    z = np.load(c.X / 'reports/d58_coordinates_358.npz')
    truth, answer, shot = z['truth'].astype(float), z['answer'].astype(float), z['shot']
    left = truth.copy()
    left[:, :KEPT] -= answer[:, :KEPT]
    cases = c.cases()
    features = np.concatenate([c.load_cache(case['stem']).features.astype(np.float32) for case in cases])
    assert len(features) == len(left)
    shots = int(shot.max()) + 1
    group = shot % GROUPS
    total = np.sum(left ** 2)
    scores = np.zeros((len(left), TOP)); ridge = np.zeros_like(scores); trees = np.zeros_like(scores)
    for g in range(GROUPS):
        fit, hold = group != g, group == g
        mean = left[fit].mean(axis=0)
        _, _, vt = np.linalg.svd(left[fit] - mean, full_matrices=False)
        d = vt[:TOP]
        yf, yh = (left[fit] - mean) @ d.T, (left[hold] - mean) @ d.T
        scores[hold] = yh
        mu, sd = features[fit].mean(axis=0), features[fit].std(axis=0) + 1e-9
        xf, xh = np.clip((features[fit] - mu) / sd, -8, 8).astype(np.float64), np.clip((features[hold] - mu) / sd, -8, 8).astype(np.float64)
        gram = xf.T @ xf + 1e-1 * len(xf) * np.eye(xf.shape[1])
        ridge[hold] = xh @ np.linalg.solve(gram, xf.T @ yf)
        wide_f = np.concatenate([features[fit], answer[fit].astype(np.float32)], axis=1)
        wide_h = np.concatenate([features[hold], answer[hold].astype(np.float32)], axis=1)
        dfit, dhold = xgb.QuantileDMatrix(wide_f, nthread=16), None
        dhold = xgb.QuantileDMatrix(wide_h, ref=dfit, nthread=16)
        for k in range(TOP):
            dfit.set_label(yf[:, k])
            booster = xgb.train({'tree_method': 'hist', 'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.5,
                                 'min_child_weight': 50, 'lambda': 5.0, 'nthread': 16, 'seed': 58}, dfit, 300)
            trees[hold, k] = booster.predict(dhold)
    report = {'frames': int(len(left)), 'directions': []}
    for k in range(TOP):
        y = scores[:, k]
        count = np.bincount(shot, minlength=shots).astype(float)
        mean = np.bincount(shot, weights=y, minlength=shots) / count
        centred = y - mean[shot]
        same = shot[1:] == shot[:-1]
        report['directions'].append({
            'share_of_left_over_energy': float(np.sum(y ** 2) / total),
            'per_shot_offset_share': float(np.sum(count * mean ** 2) / np.sum(y ** 2)),
            'lag1_autocorrelation_after_offset': float(np.sum(centred[1:][same] * centred[:-1][same]) / np.sum(centred ** 2)),
            'explained_by_ridge_on_features': float(1 - np.sum((y - ridge[:, k]) ** 2) / np.sum(y ** 2)),
            'explained_by_trees_on_features_and_answer': float(1 - np.sum((y - trees[:, k]) ** 2) / np.sum(y ** 2))})
    report['left_over_energy_removed_by_trees_top6'] = float(np.sum(scores ** 2 - (scores - trees) ** 2) / total)
    report['left_over_energy_removed_by_ridge_top6'] = float(np.sum(scores ** 2 - (scores - ridge) ** 2) / total)
    (c.X / 'reports/d58b_residual_predictability_358.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
