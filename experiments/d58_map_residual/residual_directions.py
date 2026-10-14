# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D58c: what the leading left-over directions look like and what they follow inside a shot
"""Measurement only. Spatial pattern of the three leading left-over directions (all 358 shots), and for each the
share of its within-shot variation explained by one input signal at a time (per shot: score = a + b x signal), for
the 19 coil circuits, the plasma current and the toroidal-field current, pooled over shots."""
import json

import numpy as np

import d39_common as c
import predict_diii_d as pd

KEPT, TOP = 45, 3


def main():
    z = np.load(c.X / 'reports/d58_coordinates_358.npz')
    truth, answer, shot = z['truth'].astype(float), z['answer'].astype(float), z['shot']
    left = truth.copy()
    left[:, :KEPT] -= answer[:, :KEPT]
    basis = pd.FluxBasis.load(c.PKG / 'models/flux_basis.npz').truncated(128)
    _, s, vt = np.linalg.svd(left - left.mean(axis=0), full_matrices=False)
    scores = (left - left.mean(axis=0)) @ vt[:TOP].T
    features = np.concatenate([c.load_cache(case['stem']).features.astype(np.float64) for case in c.cases()])
    shots = int(shot.max()) + 1
    report = {'mean_left_over_coordinate_1': float(left[:, 0].mean()), 'directions': []}
    for k in range(TOP):
        pattern = (vt[k] @ basis.directions).reshape(65, 65)
        d = {'share_of_left_over_variance': float(s[k] ** 2 / np.sum(s ** 2)),
             'weight_on_basis_coordinates_1_to_6': np.round(vt[k][:6], 3).tolist(),
             'pattern_mean_over_rms': float(pattern.mean() / np.sqrt(np.mean(pattern ** 2))),
             'pattern_relative_spread': float(pattern.std() / abs(pattern.mean())),
             'pattern_corner_values_over_centre': np.round(np.array([pattern[0, 0], pattern[0, -1], pattern[-1, 0], pattern[-1, -1]]) / pattern[32, 32], 3).tolist()}
        best = []
        for column in range(21):
            explained = total = 0.0
            for index in range(shots):
                pick = shot == index
                y, x = scores[pick, k], features[pick, column]
                y, x = y - y.mean(), x - x.mean()
                total += np.sum(y ** 2)
                if np.sum(x ** 2) > 0:
                    explained += np.sum(x * y) ** 2 / np.sum(x ** 2)
            best.append((explained / total, column))
        best.sort(reverse=True)
        d['within_shot_share_explained_by_one_signal_best5'] = [(int(col), round(float(v), 3)) for v, col in best[:5]]
        # the same with two signals: best single plus plasma current (column 19)
        report['directions'].append(d)
    report['feature_columns_note'] = 'columns 0-18 coil circuits (oriented, MA-turns), 19 plasma current, 20 toroidal-field current'
    (c.X / 'reports/d58c_residual_directions_358.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
