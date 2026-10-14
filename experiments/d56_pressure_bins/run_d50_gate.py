# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D50 gate: residual correctors with and without the thermal features, paired
"""Two arms of the D10-0 cross-fitted residual corrector on a compact input (protocol_d50_gate.json).

control    13 out-of-fold network heads + plasma current, toroidal-field current, their ratio and 1 / current
treatment  the same + the thermal features (thermal_features.py)

For outer fold k the corrector is fitted on three other folds, stopped on fold (k + 1) % 5 and predicted on fold k.
Output per arm and fold in the layout d9_gate_eval.py reads.

    python run_d50_gate.py NN_OOF.npz THERMAL.npz OUT_DIR THREADS
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

PARAMS = {'tree_method': 'hist', 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 6,
          'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 50, 'lambda': 5.0,
          'max_bin': 256, 'seed': 10}
ROUNDS, PATIENCE, FOLDS = 4000, 150, 5


def main():
    heads_file, thermal_file, out, threads = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    with np.load(heads_file, allow_pickle=False) as z:
        nn, labels, row_fold, shot = z['nn'], z['labels'], z['row_fold'], z['shot_of_row']
        targets = z['targets'].astype(str).tolist()
    with np.load(thermal_file, allow_pickle=False) as z:
        thermal, basics = z['features'], z['basics']
        assert np.array_equal(z['shot_of_row'], shot)
    assert len(thermal) == len(nn) == len(basics) and np.isfinite(thermal).all() and np.isfinite(basics).all()
    heads = np.where(np.isfinite(nn), nn, 0.0).astype(np.float32)
    arms = {'control': np.concatenate([heads, basics], axis=1),
            'treatment': np.concatenate([heads, basics, thermal], axis=1)}
    print(json.dumps({'rows': int(len(nn)), 'widths': {k: int(v.shape[1]) for k, v in arms.items()}, 'targets': targets,
                      'rows_per_fold': [int((row_fold == k).sum()) for k in range(-1, FOLDS)],
                      'xgboost': xgb.__version__}), flush=True)
    for arm, X in arms.items():
        (out / arm).mkdir(parents=True, exist_ok=True)
        for k in range(FOLDS):
            target_file = out / arm / f'gbm_oof_fold{k}.npz'
            if target_file.exists():
                continue
            started = time.time()
            stop_fold = (k + 1) % FOLDS
            train = (row_fold >= 0) & (row_fold != k) & (row_fold != stop_fold)
            stop, held = row_fold == stop_fold, row_fold == k
            dtrain = xgb.QuantileDMatrix(X[train], max_bin=PARAMS['max_bin'], nthread=threads)
            dstop = xgb.QuantileDMatrix(X[stop], ref=dtrain, nthread=threads)
            dheld = xgb.QuantileDMatrix(X[held], ref=dtrain, nthread=threads)
            corrected = np.full((int(held.sum()), len(targets)), np.nan, dtype=np.float32)
            info = []
            for j, name in enumerate(targets):
                t0 = time.time()
                r, rs = labels[train, j] - nn[train, j], labels[stop, j] - nn[stop, j]
                ok, oks = np.isfinite(r), np.isfinite(rs)
                dtrain.set_label(np.where(ok, r, 0.0)); dtrain.set_weight(ok.astype(np.float32))
                dstop.set_label(np.where(oks, rs, 0.0)); dstop.set_weight(oks.astype(np.float32))
                params = dict(PARAMS, nthread=threads, base_score=float(r[ok].mean()))
                booster = xgb.train(params, dtrain, ROUNDS, evals=[(dstop, 'stop')], early_stopping_rounds=PATIENCE,
                                    verbose_eval=False)
                best = int(booster.best_iteration)
                predicted = booster.predict(dheld, iteration_range=(0, best + 1))
                corrected[:, j] = nn[held, j] + predicted
                truth = labels[held, j]; fin = np.isfinite(truth)
                sse_nn = float(np.sum((truth[fin] - nn[held, j][fin]) ** 2))
                sse_corrected = float(np.sum((truth[fin] - corrected[fin, j]) ** 2))
                gain = booster.get_score(importance_type='total_gain')
                top = sorted(gain.items(), key=lambda item: -item[1])[:8]
                info.append({'arm': arm, 'fold': k, 'target': name, 'best_iteration': best,
                             'stop_rmse': float(booster.best_score),
                             'stop_rmse_of_zero': float(np.sqrt(np.mean(rs[oks] ** 2))),
                             'held_error_variance_ratio': sse_corrected / sse_nn,
                             'top_gain_columns': [(name_, round(value / sum(gain.values()), 3)) for name_, value in top],
                             'seconds': round(time.time() - t0)})
                print(json.dumps(info[-1]), flush=True)
            temporary = target_file.with_name(f'partial_fold{k}.npz')
            np.savez(temporary, pred=corrected, rows=np.flatnonzero(held), targets=np.asarray(targets))
            temporary.replace(target_file)
            (out / arm / f'gbm_oof_fold{k}.json').write_text(json.dumps(
                {'fold': k, 'stop_fold': stop_fold, 'params': dict(PARAMS, nthread=threads), 'rounds': ROUNDS,
                 'patience': PATIENCE, 'fits': info, 'seconds': round(time.time() - started)}, indent=1))
            del dtrain, dstop, dheld
    print('ALL_D50_GATE_DONE', flush=True)


if __name__ == '__main__':
    main()
