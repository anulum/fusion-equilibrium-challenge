# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — R9 (diagnostic): is the per-shot offset of the network's scalar error predictable?
"""400 OOF shots (K-fold features_fold*.npz, every 2nd frame). Target per shot: mean(truth - OOF map) of li, kappa,
R_axis. Features per shot: mean and std of the 367 inputs over the shot, mean of the 7 map-read scalars, frame count.
Leave-one-fold-out HistGB (small, regularised) and ridge; report the share of the offset variance explained, and the
implied share of the total per-frame squared error removed. Diagnosis only (decision threshold: >= 10 % of offset)."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
K = Path(__file__).resolve().parent.parent / 'kfold'
TABLE = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_training.npz')
T = {'li': ('li_scored', 'li'), 'kappa': ('kappa', 'kappa'), 'R_axis': ('r_axis', 'R_axis')}
folds = [np.load(K / f'features_fold{k}.npz') for k in range(5)]
with np.load(TABLE) as t:
    tst = t['stems'].astype(str); start = np.r_[0, np.cumsum(np.bincount(t['shot_of_row'], minlength=len(tst)))]
    where = {s: i for i, s in enumerate(tst)}; names = t['label_names'].astype(str).tolist()
    labels = t['labels']
    X, Y, fid, frame_err = [], {k: [] for k in T}, [], {k: [] for k in T}
    for k, f in enumerate(folds):
        mc = [str(c) for c in f['map_columns']]
        for s_i, stem in enumerate(f['stems'].astype(str)):
            sel = f['shot_of_row'] == s_i
            rows = start[where[stem]] + f['frame'][sel]
            inp = f['inputs'][sel].astype(np.float64); mf = f['map_features'][sel]
            X.append(np.r_[inp.mean(0), inp.std(0), np.nanmean(mf[:, [mc.index(c) for c in ('R_axis','Z_axis','kappa','tri_top','tri_bot','volume','li')]], 0), sel.sum()])
            fid.append(k)
            for key, (tl, ml) in T.items():
                e = labels[rows, names.index(tl)].astype(np.float64) - mf[:, mc.index(ml)]
                e = e[np.isfinite(e)]
                Y[key].append(e.mean() if e.size else np.nan); frame_err[key].append(e)
X, fid = np.nan_to_num(np.asarray(X)), np.asarray(fid)
out = {}
for key in T:
    y = np.asarray(Y[key]); ok = np.isfinite(y)
    res = {}
    for name, mk in (('hgb', lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=20, l2_regularization=5.0, random_state=0)),
                     ('ridge', lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12))))):
        p = np.full(len(y), np.nan)
        for k in range(5):
            tr, te = ok & (fid != k), ok & (fid == k)
            p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        r2 = 1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        tot = sum(float(np.sum(e ** 2)) for e, o in zip(frame_err[key], ok) if o)
        after = sum(float(np.sum((e - pp) ** 2)) for e, pp, o in zip(frame_err[key], p, ok) if o)
        res[name] = {'offset_R2_explained': float(r2), 'frame_SSE_reduction_share': 1 - after / tot}
    res['offset_share_of_frame_SSE'] = sum(len(e) * float(np.mean(e)) ** 2 for e, o in zip(frame_err[key], ok) if o) / sum(float(np.sum(e ** 2)) for e, o in zip(frame_err[key], ok) if o)
    out[key] = res; print(key, json.dumps(res), flush=True)
(Path(__file__).resolve().parent / 'results' / 'r9_shot_offset.json').write_text(json.dumps(out, indent=1))
