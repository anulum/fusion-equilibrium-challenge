# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fold-safe adaptive total-pressure factors
"""Fit five pressure factors without evaluating comparative metrics."""

import json
from pathlib import Path

import numpy as np

from campaign_common import HERE, PARTS, base_maps, digest, extra_maps, metadata, reserved, fold_of, write_json
from train_batch import Moments


def main():
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    assert neural['status'] == 'all_50_neural_tasks_terminal' and len(neural['tasks']) == 50
    base, extra = base_maps(), extra_maps()
    metas = {part: metadata(part) for part in PARTS}
    held = reserved()
    out = HERE / 'pressure_factors'
    out.mkdir(exist_ok=False)
    receipts = []
    for fold in range(5):
        moments = Moments(8)
        selected = []
        for part in ('training', 'calibration'):
            meta = metas[part]
            for index, stem in enumerate(meta['stems']):
                if stem in held or (part == 'training' and fold_of(stem) == fold):
                    continue
                start, stop = meta['bounds'][index:index + 2]
                selected.append((part, stem, start, stop))
                moments.add(extra[part]['factor_features'][start:stop].astype(float))
        scale = moments.scale()
        xx, xy = np.zeros((9, 9)), np.zeros(9)
        shots, rows = 0, 0
        for part, stem, start, stop in selected:
            beta_e = extra[part]['beta_e_truth'][start:stop].astype(float)
            beta_truth = base[part]['labels'][start:stop, metas[part]['label_names'].index('beta_n')].astype(float)
            valid = np.isfinite(beta_e) & np.isfinite(beta_truth) & (beta_e > 0) & (beta_truth > 0)
            if not valid.any():
                continue
            values = extra[part]['factor_features'][start:stop][valid].astype(float)
            z = np.clip((values - moments.mean) / scale, -8, 8)
            z = np.column_stack([np.ones(len(z)), z])
            y = np.log(beta_truth[valid] / beta_e[valid])
            xx += z.T @ z / len(z)
            xy += z.T @ y / len(z)
            rows += len(z)
            shots += 1
        assert rows > 0 and shots > 0
        xx /= shots
        xy /= shots
        penalty = np.eye(9) * .01
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(xx + penalty, xy)
        assert np.isfinite(coefficients).all()
        file = out / f'factor_fold{fold}.npz'
        np.savez(file, coefficients=coefficients, feature_mean=moments.mean, feature_scale=scale,
                 factor_bounds=np.array([.5, 8.]), fold=np.asarray(fold))
        receipt = {'fold': fold, 'status': 'factor_fit_complete', 'rows': rows, 'shots': shots,
                   'sha256': digest(file), 'ridge': .01, 'equal_shot_weighting': True,
                   'MAST_targets_used': False, 'evaluation_metrics_computed': False}
        receipts.append(receipt)
        print(json.dumps({'factor_fit_complete': fold, 'rows': rows, 'shots': shots}), flush=True)
    write_json(out / 'receipt.json', {'status': 'all_five_factor_fits_complete', 'models': receipts,
                                    'comparative_evaluation_deferred': True})


def predict_factor(path, features):
    with np.load(path, allow_pickle=False) as stored:
        values = np.clip((features - stored['feature_mean']) / stored['feature_scale'], -8, 8)
        logits = np.column_stack([np.ones(len(values)), values]) @ stored['coefficients']
        bounds = stored['factor_bounds']
        return np.exp(np.clip(logits, np.log(bounds[0]), np.log(bounds[1])))


if __name__ == '__main__':
    main()
