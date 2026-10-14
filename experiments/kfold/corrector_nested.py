# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — map-aware X-point corrector, stages B and C: nested cross-fit and arms
"""Fit and evaluate the corrector exactly as protocol_mapaware_xpoint_corrector_20260929 fixes it.

Inputs: the five per-fold feature files from corrector_features.py and the five OOF archives.
For each outer fold j the corrector is fitted on the other four folds (its uncertainty model on
an inner 4-fold cross-fit of those), predicts fold j, and the arms are scored on fold j maps.

    python corrector_nested.py --kfold-dir DIR --out DIR [--workers 6]
"""
import argparse
import hashlib
import json
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA = Path('$FEC_DATA/data/diii_d_train')
PROTOCOL = HERE / 'protocol_mapaware_xpoint_corrector_20260929.json'
FOLDS = 5
ENDS = ('lower', 'upper')
ARMS = ('base', 'gated_full', 'gated_half', 'ungated_full', 'oracle_truth')
MIN_MOVE, SNR = 0.001, 2.0


def mean_model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)


def sigma_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)


def digest(path):
    with Path(path).open('rb') as s:
        return hashlib.file_digest(s, 'sha256').hexdigest()


def design(d):
    return np.column_stack([d['map_features'], d['inputs']]).astype(np.float32)


def targets(d, e):
    """Residual target (true saddle minus map saddle) for end e, NaN where either is missing."""
    mf = d['map_features']
    return d['target'][:, 2 * e:2 * e + 2] - mf[:, [7 * e, 7 * e + 1]]


def fit_predict(train, test):
    """Return predicted residual (n,4) and sigma (n,2) for test from train dicts (lists)."""
    x = np.concatenate([design(d) for d in train])
    xt = design(test)
    group = np.concatenate([np.full(len(d['frame']), i) for i, d in enumerate(train)])
    mu, sig = np.full((len(xt), 4), np.nan), np.full((len(xt), 2), np.nan)
    for e in range(2):
        y = np.concatenate([targets(d, e) for d in train])
        have = np.isfinite(y).all(axis=1)
        inner = np.full(y.shape, np.nan)
        for g in range(len(train)):
            fit, held = have & (group != g), have & (group == g)
            for c in range(2):
                inner[held, c] = mean_model().fit(x[fit], y[fit, c]).predict(x[held])
        err = np.hypot(*(inner[have] - y[have]).T)
        smod = sigma_model().fit(x[have], np.log(err + 1e-4))
        for c in range(2):
            mu[:, 2 * e + c] = mean_model().fit(x[have], y[have, c]).predict(xt)
        sig[:, e] = np.exp(smod.predict(xt))
    return mu, sig


def shot_arms(task):
    from scipy.spatial.distance import directed_hausdorff
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    from fec_adapter.diii_d_accuracy.x_point import with_x_points
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    archive, index, stem, frames, map_xy, mu, sig, truth_xy = task
    reader = Reader('DIII-D')
    with np.load(archive, allow_pickle=False) as a:
        maps = a[f'shot_{index:04d}_psirz']
    truth_maps = np.asarray(pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])
                            ['efit_psirz'][0].as_py(), dtype=np.float64)

    def terms(rt, ct, rgeo, flux, truth):
        r = reader.read(flux)
        if r.contour is None:
            d = np.nan
        else:
            c = np.asarray(r.contour, float)
            d = max(directed_hausdorff(c, ct)[0], directed_hausdorff(ct, c)[0]) / rgeo
        return [d, *[(r.values[n] - rt.values[n]) ** 2 for n in CONS_SCALARS], float(np.sum((flux - truth) ** 2))]

    rows = []
    for i, f in enumerate(frames):
        truth, pred = truth_maps[f], np.asarray(maps[f], np.float64)
        rt = reader.read(truth)
        if rt.contour is None:
            continue
        ct = np.asarray(rt.contour, float)
        rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
        wanted_gate, wanted_all, wanted_truth, gated = {}, {}, {}, []
        for e, end in enumerate(ENDS):
            corr = mu[i, 2 * e:2 * e + 2]
            size = float(np.hypot(*corr)) if np.isfinite(corr).all() else 0.0
            ok = np.isfinite(map_xy[i, 2 * e]) and size >= MIN_MOVE and size >= SNR * sig[i, e]
            wanted_gate[end] = map_xy[i, 2 * e:2 * e + 2] + corr if ok else np.full(2, np.nan)
            wanted_all[end] = map_xy[i, 2 * e:2 * e + 2] + corr if np.isfinite(corr).all() else np.full(2, np.nan)
            wanted_truth[end] = truth_xy[i, 2 * e:2 * e + 2]
            gated.append(bool(ok))
        base = terms(rt, ct, rgeo, pred, truth)
        rec = {'stem': stem, 'frame': int(f), 'gated': gated, 'base': base}
        rec['gated_full'] = terms(rt, ct, rgeo, with_x_points(reader, pred, wanted_gate)[0], truth) if any(gated) else base
        rec['gated_half'] = terms(rt, ct, rgeo, with_x_points(reader, pred, wanted_gate, share=0.5)[0], truth) if any(gated) else base
        rec['ungated_full'] = terms(rt, ct, rgeo, with_x_points(reader, pred, wanted_all)[0], truth)
        rec['oracle_truth'] = terms(rt, ct, rgeo, with_x_points(reader, pred, wanted_truth)[0], truth)
        rows.append(rec)
    return rows


def summarise(rows, rng, gates):
    from fec_adapter.scoring_kit.common import CONS_SCALARS
    names = ['D', *CONS_SCALARS, 'flux_sse']
    arr = {a: np.asarray([r[a] for r in rows], float) for a in ARMS}
    shots = np.asarray([r['stem'] for r in rows])
    us = np.unique(shots)
    out = {'frames': len(rows), 'shots': int(len(us)), 'gated_frame_share': float(np.mean([any(r['gated']) for r in rows]))}
    for arm in ARMS[1:]:
        a, b = arr[arm], arr['base']
        per = np.asarray([np.nanmean(a[shots == s, 0]) - np.nanmean(b[shots == s, 0]) for s in us])
        changed = np.asarray([np.any(a[shots == s, 0] != b[shots == s, 0]) for s in us])
        ci = np.quantile(per[rng.integers(0, len(us), size=(20000, len(us)))].mean(axis=1), [.025, .975]).tolist()
        ratios = {n: float(np.nansum(a[:, i]) / np.nansum(b[:, i])) for i, n in enumerate(names) if i > 0}
        wins = float(np.mean(per[changed] < 0)) if changed.any() else None
        e = {'mean_D_base': float(np.nanmean(b[:, 0])), 'mean_D_arm': float(np.nanmean(a[:, 0])),
             'mean_shot_D_change': float(per.mean()), 'shot_bootstrap95': ci, 'changed_shots': int(changed.sum()),
             'win_fraction_changed_shots': wins,
             'contour_losses': int(np.sum(np.isnan(a[:, 0]) & ~np.isnan(b[:, 0]))), 'squared_error_ratios': ratios}
        if arm.startswith('gated'):
            e['gates'] = {'mean_D_change_negative': e['mean_shot_D_change'] < 0, 'bootstrap_upper_below_zero': ci[1] < 0,
                          'wins': wins is not None and wins >= gates['win_fraction_among_changed_shots_at_least'],
                          'no_contour_losses': e['contour_losses'] == 0,
                          'li': ratios['li'] <= gates['li_squared_error_ratio_at_most'],
                          'kappa': ratios['kappa'] <= gates['kappa_squared_error_ratio_at_most'],
                          'flux': ratios['flux_sse'] <= gates['flux_sse_ratio_at_most']}
        out[arm] = e
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        protocol = json.loads(PROTOCOL.read_text())
        assert protocol['status'] == 'frozen_before_any_oof_map_exists'
        data = [dict(np.load(args.kfold_dir / f'features_fold{k}.npz', allow_pickle=False)) for k in range(FOLDS)]
        archives = [args.kfold_dir / f'oof_fold{k}' / 'diii_d_train.npz' for k in range(FOLDS)]
        all_rows, per_fold = [], {}
        rng = np.random.default_rng(930)
        for j in range(FOLDS):
            stage = f'fit_outer_{j}'
            mu, sig = fit_predict([data[k] for k in range(FOLDS) if k != j], data[j])
            d = data[j]
            map_xy = d['map_features'][:, [0, 1, 7, 8]]
            tasks = []
            for s_i, stem in enumerate(d['stems'].astype(str)):
                sel = d['shot_of_row'] == s_i
                tasks.append((str(archives[j]), int(d['archive_index'][sel][0]), stem, d['frame'][sel],
                              map_xy[sel], mu[sel], sig[sel], d['target'][sel]))
            stage = f'arms_outer_{j}'
            with Pool(args.workers) as pool:
                rows = [r for part in pool.map(shot_arms, tasks) for r in part]
            per_fold[j] = summarise(rows, rng, protocol['admission_gates_for_gated_arms'])
            all_rows.extend(rows)
            print(json.dumps({'outer_fold': j, 'gated_full': per_fold[j]['gated_full']['mean_shot_D_change'],
                              'seconds': round(time.time() - started)}), flush=True)
        stage = 'summary'
        pooled = summarise(all_rows, rng, protocol['admission_gates_for_gated_arms'])
        folds_negative = {arm: sum(per_fold[j][arm]['mean_shot_D_change'] < 0 for j in range(FOLDS))
                          for arm in ('gated_full', 'gated_half')}
        admitted = {arm: all(pooled[arm]['gates'].values()) and folds_negative[arm] >= 4
                    for arm in ('gated_full', 'gated_half')}
        summary = {'status': 'nested_cross_fit_complete', 'protocol_sha256': digest(PROTOCOL),
                   'pooled': pooled, 'per_fold': per_fold, 'folds_with_negative_mean_D_change': folds_negative,
                   'admitted_for_single_full357_screen': admitted,
                   'validation_targets_used': False, 'MAST_targets_used': False}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
        (args.out / 'rows.json').write_text(json.dumps(all_rows))
        print(json.dumps({'admitted': admitted, 'folds_negative': folds_negative}), flush=True)
        code, stage = 0, 'complete'
    except Exception:
        (args.out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (args.out / 'terminal.json').write_text(json.dumps(
            {'actual_exit_code': code, 'stage': stage, 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
