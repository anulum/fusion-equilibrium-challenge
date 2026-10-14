# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible shared confirmation tree heads
"""Reproduce original tree hyperparameters while excluding both evaluation cohorts."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from campaign_common import HERE, digest, write_json
from training_data import initialize

D32 = HERE.parent / 'D32_DIIID_separate_cohort_confirmation_2026-10-10T144836Z'
sys.path.insert(0, str(D32 / 'vendor'))
import xgboost as xgb

TARGETS = ('li_scored', 'r_axis', 'kappa', 'x_lower_r')
COMMON = {'tree_method': 'hist', 'objective': 'reg:squarederror', 'eval_metric': 'rmse',
          'max_depth': 8, 'subsample': .8, 'colsample_bytree': .5, 'min_child_weight': 20,
          'lambda': 1., 'max_bin': 256, 'seed': 9, 'nthread': 4, 'device': 'cpu'}
SETTINGS = {'B': {'eta': .1, 'rounds': 6000, 'patience': 200},
            'A': {'eta': .05, 'rounds': 3000, 'patience': 100}}


class FeatureIterator(xgb.DataIter):
    """Stream eligible original367 predictor features into quantile matrices."""

    def __init__(self, data: dict, part: str, rows: np.ndarray) -> None:
        super().__init__(release_data=True)
        self.source = data['base'][part]['features']
        self.rows = rows
        self.at = 0

    def reset(self) -> None:
        """Restart the source pass required by quantile construction."""
        self.at = 0

    def next(self, input_data) -> bool:
        """Supply one bounded block without allocating the whole input corpus."""
        if self.at == len(self.rows):
            return False
        chosen = self.rows[self.at:self.at + 8192]
        values = np.asarray(self.source[chosen, :367], dtype=np.float32, order='C')
        input_data(data=values)
        self.at += len(chosen)
        return True


def labels(data: dict, part: str, rows: np.ndarray) -> np.ndarray:
    """Use the unchanged original label definitions for all four shared heads."""
    names = data['metas'][part]['label_names']
    wanted = [names.index(n) for n in TARGETS]
    return np.asarray(data['base'][part]['labels'][rows][:, wanted], dtype=np.float32)


def lookup(booster, best: int, path: Path) -> dict:
    """Record the frozen float32 leaf accumulation used by the production pipeline."""
    learner = json.loads(booster.save_config())['learner']
    raw = learner['learner_model_param']['base_score']
    decoded = json.loads(raw)
    base = np.float32(decoded[0] if isinstance(decoded, list) else decoded)
    leaves = []
    for raw_tree in booster.get_dump(dump_format='json')[:best + 1]:
        stack, values = [json.loads(raw_tree)], {}
        while stack:
            node = stack.pop()
            if 'leaf' in node:
                values[node['nodeid']] = np.float32(node['leaf'])
            else:
                stack.extend(node['children'])
        leaves.append(values)
    matrix = np.full((len(leaves), max(max(tree) for tree in leaves) + 1), np.nan, dtype=np.float32)
    for index, tree in enumerate(leaves):
        for node, value in tree.items():
            matrix[index, node] = value
    np.savez(path, leaf_values=matrix, base_score=base)
    return {'lookup_sha256': digest(path), 'base_score': float(base), 'tree_count': len(leaves)}


def fit_all(qualify: bool = False) -> None:
    """Keep calibration-only stopping and never read held-out targets during fits."""
    data = initialize()
    rows = {part: data['rows'][part][:2048] if qualify else data['rows'][part] for part in ('training', 'calibration')}
    out = HERE / ('confirmation_tree_qualification' if qualify else 'confirmation_trees')
    out.mkdir(exist_ok=True)
    train = xgb.QuantileDMatrix(FeatureIterator(data, 'training', rows['training']), max_bin=256, nthread=4)
    calibration = xgb.QuantileDMatrix(FeatureIterator(data, 'calibration', rows['calibration']), ref=train, nthread=4)
    targets = labels(data, 'training', rows['training'])
    cal_targets = labels(data, 'calibration', rows['calibration'])
    reports = []
    for index, name in enumerate(TARGETS):
        destination = out / name
        destination.mkdir(exist_ok=True)
        selected = SETTINGS['A' if name == 'x_lower_r' else 'B']
        finite, cal_finite = np.isfinite(targets[:, index]), np.isfinite(cal_targets[:, index])
        train.set_label(np.where(finite, targets[:, index], 0.)); train.set_weight(finite.astype(np.float32))
        calibration.set_label(np.where(cal_finite, cal_targets[:, index], 0.)); calibration.set_weight(cal_finite.astype(np.float32))
        params = dict(COMMON, eta=selected['eta'], base_score=float(targets[finite, index].mean()))
        started = time.monotonic()
        booster = xgb.train(params, train, 2 if qualify else selected['rounds'], evals=[(calibration, 'cal')],
                            early_stopping_rounds=selected['patience'], verbose_eval=False)
        best = int(booster.best_iteration)
        booster.save_model(destination / 'tree.ubj')
        values = booster.predict(calibration, iteration_range=(0, best + 1))
        restored = xgb.Booster(); restored.load_model(destination / 'tree.ubj'); restored.set_param({'nthread': 4, 'device': 'cpu'})
        assert np.array_equal(values, restored.predict(calibration, iteration_range=(0, best + 1)))
        assert np.isfinite(values).all()
        leaf_receipt = lookup(booster, best, destination / 'lookup.npz')
        report = {'target': name, 'status': 'eligible_tree_complete', 'best_iteration': best,
                  'fit_frames': len(rows['training']), 'calibration_stop_frames': len(rows['calibration']),
                  'finite_fit_targets': int(finite.sum()), 'calibration_rmse': float(booster.best_score),
                  'model_sha256': digest(destination / 'tree.ubj'), 'params': params, 'settings': selected,
                  'seconds': time.monotonic() - started, 'export_reload_predictions_bitexact': True,
                  'both358_development_and357_confirmation_excluded': True, **leaf_receipt}
        write_json(destination / 'fit_report.json', report)
        reports.append(report)
        print(json.dumps({'eligible_tree_finished': name, 'seconds': report['seconds'], 'best_iteration': best}), flush=True)
        del booster, restored
        gc.collect()
    write_json(out / 'receipt.json', {'status': 'all_four_eligible_shared_confirmation_trees_complete' if not qualify else 'real_eligible_tree_input_export_path_qualified',
               'reports': reports, 'confirmation_labels_read': False, 'training': 'original5954 only; calibration372 only early stopping',
               'fit_corpus_includes_calibration': False, 'original_hyperparameters_preserved_CPU_device': True})


def main() -> None:
    """Qualify tree input/export or fit the shared confirmation-only tree models."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    if not args.qualify:
        assert json.loads((HERE / 'confirmation_tree_qualification/receipt.json').read_text())['status'] == 'real_eligible_tree_input_export_path_qualified'
        assert json.loads((HERE / 'evaluation/development_analysis.json').read_text())['frozen_confirmation_candidate'] is not None
    fit_all(args.qualify)


if __name__ == '__main__':
    main()
