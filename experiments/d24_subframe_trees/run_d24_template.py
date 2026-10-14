# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D24-0 gate: boosted-tree heads fitted on the table features plus the sub-frame features
"""Out-of-fold boosted trees for eight heads (four per kernel) on 582 features (d24_subframe_trees/protocol_d24_0_gate.json).

The D9-0 gate with two changes: the feature matrix is the 367 table features plus the 215 sub-frame features (as
the networks of D18-0 read them), and the fit runs on the host processor (hist, 4 threads) because the accelerator
quota is exhausted. Tree parameters, rounds, patience, folds, reserved stems and calibration early stopping are
those of D9 setting A. One fold per kernel.
"""
import glob
import hashlib
import json
import os
import time

FOLD = __FOLD__
TARGETS = __TARGETS__
PART = __PART__
PARAMS = {'tree_method': 'hist', 'objective': 'reg:squarederror', 'eval_metric': 'rmse',
          'max_depth': 8, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 20,
          'lambda': 1.0, 'max_bin': 256, 'seed': 9, 'device': 'cpu', 'nthread': 4}
ROUNDS, PATIENCE, FOLDS = 3000, 100, 5
OUT = '/kaggle/working'


def find(pattern):
    return sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import xgboost as xgb
    lcfs = os.path.dirname(find('table_training.npz')[0])
    sums = dict(reversed(line.split()) for line in open(os.path.join(lcfs, 'SHA256SUMS')).read().splitlines())
    for name in ('table_training.npz', 'table_calibration.npz'):
        assert digest(os.path.join(lcfs, name)) == sums[name], name
    kdir = os.path.dirname([p for p in find('splits.json') if '/kfold/' in p][0])
    reserved = set(json.load(open(os.path.join(kdir, 'reserved_for_evaluation.json')))['stems'])
    extra = {part: find(f'subframe_{part}.npz')[0] for part in ('training', 'calibration')}
    print(json.dumps({part: digest(path)[:16] for part, path in extra.items()}), flush=True)
    with np.load(os.path.join(lcfs, 'table_training.npz'), allow_pickle=False) as t:
        names = t['label_names'].astype(str).tolist()
        columns = [names.index(n) for n in TARGETS]
        base, L = t['features'], t['labels'][:, columns]
        stems, shot_of_row = t['stems'].astype(str), t['shot_of_row']
    with np.load(extra['training'], allow_pickle=False) as z:
        assert np.array_equal(z['shot_of_row'], shot_of_row)
        more = z['features']  # read once
    F = np.concatenate([base, more], axis=1); del base, more
    stem_fold = np.array([-1 if s in reserved else int(hashlib.sha256(s.encode()).hexdigest(), 16) % FOLDS for s in stems])
    row_fold = stem_fold[shot_of_row]
    with np.load(os.path.join(lcfs, 'table_calibration.npz'), allow_pickle=False) as t:
        assert t['label_names'].astype(str).tolist() == names
        cal_shot = t['shot_of_row']
        cal_keep = np.array([s not in reserved for s in t['stems'].astype(str)])[cal_shot]
        base_c, Lc = t['features'], t['labels'][:, columns]
    with np.load(extra['calibration'], allow_pickle=False) as z:
        assert np.array_equal(z['shot_of_row'], cal_shot)
        Fc = np.concatenate([base_c, z['features']], axis=1)[cal_keep]
    Lc = Lc[cal_keep]; del base_c
    print(json.dumps({'training_rows': int(len(F)), 'features': int(F.shape[1]), 'calibration_rows': int(len(Fc)),
                      'fold': FOLD, 'xgboost': xgb.__version__}), flush=True)
    started = time.time()
    train, held = (row_fold != FOLD) & (row_fold >= 0), row_fold == FOLD
    dtrain = xgb.QuantileDMatrix(F[train], max_bin=PARAMS['max_bin'], nthread=4)
    dcal = xgb.QuantileDMatrix(Fc, ref=dtrain, nthread=4)
    dheld = xgb.QuantileDMatrix(F[held], ref=dtrain, nthread=4)
    del F
    pred = np.full((int(held.sum()), len(TARGETS)), np.nan, dtype=np.float32)
    info = []
    for j, name in enumerate(TARGETS):
        t0 = time.time()
        y, yc = L[train, j], Lc[:, j]
        ok, okc = np.isfinite(y), np.isfinite(yc)
        dtrain.set_label(np.where(ok, y, 0.0)); dtrain.set_weight(ok.astype(np.float32))
        dcal.set_label(np.where(okc, yc, 0.0)); dcal.set_weight(okc.astype(np.float32))
        booster = xgb.train(dict(PARAMS, base_score=float(y[ok].mean())), dtrain, ROUNDS, evals=[(dcal, 'cal')],
                            early_stopping_rounds=PATIENCE, verbose_eval=False)
        best = int(booster.best_iteration)
        pred[:, j] = booster.predict(dheld, iteration_range=(0, best + 1))
        truth = L[held, j]; fin = np.isfinite(truth)
        r2 = float(1 - np.sum((truth[fin] - pred[fin, j]) ** 2) / np.sum((truth[fin] - truth[fin].mean()) ** 2))
        info.append({'fold': FOLD, 'target': name, 'best_iteration': best, 'cal_rmse': float(booster.best_score),
                     'held_rows': int(fin.sum()), 'held_r2': r2, 'seconds': round(time.time() - t0)})
        print(json.dumps(info[-1]), flush=True)
        np.savez(os.path.join(OUT, f'gbm_sub_oof_fold{FOLD}{PART}.npz'), pred=pred, rows=np.flatnonzero(held),
                 targets=np.asarray(TARGETS), done=np.asarray(j + 1))
    json.dump({'fold': FOLD, 'params': PARAMS, 'rounds': ROUNDS, 'patience': PATIENCE, 'fits': info,
               'seconds': round(time.time() - started)}, open(os.path.join(OUT, f'gbm_sub_oof_fold{FOLD}{PART}.json'), 'w'), indent=1)
    print('D24_FOLD_DONE', FOLD, flush=True)


main()
