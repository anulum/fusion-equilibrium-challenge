# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D re-evaluation R5: does whole-shot context add information?
"""OOF HistGB on 1000 training shots (every 4th frame), 5 folds by shot hash: base = 367 features;
plus = base + per-shot mean and std of every feature over the whole record + frame position in the
record (fraction, frames from start, frames to end). Targets li_scored, kappa, r_axis, and the first
five flux coordinates. Diagnosis only."""
import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
REPO = Path('$HOME/fusion-equilibrium-challenge')
TABLE = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_training.npz'
OUT = Path(__file__).resolve().parent / 'results' / 'r5_shot_context_test.json'
splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
reserved = set(json.loads((REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems'])
pick = sorted((s for s in splits['training'] if s not in reserved), key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:1000]
with np.load(TABLE) as t:
    stems = t['stems'].astype(str); sor = t['shot_of_row']; names = t['label_names'].astype(str).tolist()
    counts = np.bincount(sor, minlength=len(stems)); starts = np.concatenate(([0], np.cumsum(counts)))
    where = {s: i for i, s in enumerate(stems)}
    X, C, Y, F = [], [], [], []
    feats_all = t['features']; lab = t['labels']; coords = t['coordinates']
    for s in pick:
        i = where[s]; b, n = starts[i], counts[i]
        f = feats_all[b:b + n].astype(np.float32)
        ctx = np.concatenate([f.mean(0), f.std(0)])
        fr = np.arange(0, n, 4)
        pos = np.column_stack([fr / max(n - 1, 1), fr, n - 1 - fr]).astype(np.float32)
        X.append(f[fr]); C.append(np.column_stack([np.repeat(ctx[None], len(fr), 0), pos]))
        Y.append(np.column_stack([lab[b + fr][:, [names.index(k) for k in ('li_scored', 'kappa', 'r_axis')]], coords[b + fr, :5]]))
        F.append(np.full(len(fr), int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5))
X, C, Y, F = map(np.concatenate, (X, C, Y, F))
XP = np.column_stack([X, C])
out = {'rows': int(len(X))}
for c, name in enumerate(['li_scored', 'kappa', 'r_axis', 'c0', 'c1', 'c2', 'c3', 'c4']):
    ok = np.isfinite(Y[:, c]); res = {}
    for tag, xx in (('base', X), ('plus_shot_context', XP)):
        pred = np.full(len(Y), np.nan)
        for k in range(5):
            fit, held = ok & (F != k), ok & (F == k)
            pred[held] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                       early_stopping=False, random_state=0).fit(xx[fit], Y[fit, c]).predict(xx[held])
        r = Y[ok, c]; res[tag] = float(1 - np.sum((pred[ok] - r) ** 2) / np.sum((r - r.mean()) ** 2))
    out[name] = res; print(name, res, flush=True)
OUT.write_text(json.dumps(out, indent=1))
