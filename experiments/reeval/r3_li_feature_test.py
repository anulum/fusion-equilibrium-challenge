# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D re-evaluation R3: do current-diffusion features add li information?
"""Out-of-fold li prediction on 1000 training shots (every 4th frame), HistGB, 5 folds by shot hash:
base = the 367 table features; plus = base + the three current-diffusion outputs (li_hist,
li_relaxed, peaking). Targets: li_scored (the scorer's li of the basis map), kappa, r_axis as
controls. Reports OOF R2 for both. Training split only, reserved stems excluded; diagnosis only."""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

import current_diffusion as cd

REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$FEC_DATA/data/diii_d_train')
TABLE = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_training.npz'
OUT = Path(__file__).resolve().parent / 'results' / 'r3_li_feature_test.json'
SHOTS, STRIDE = 1000, 4


def feats(stem):
    t = pq.read_table(DATA / f'{stem}.parquet', columns=['thomson_core_times', 'thomson_core_Te', 'thomson_chord_name',
                                                         'thomson_chord_Z', 'magnetics_plasma_current',
                                                         'magnetics_plasma_current_times', 'efit_times'])
    names = list(t['thomson_chord_name'][0].as_py())
    z = np.asarray(t['thomson_chord_Z'][0].as_py(), float)
    te = np.asarray(t['thomson_core_Te'][0].as_py(), float)
    zc = z[[i for i, n in enumerate(names) if n.startswith('TS_core')]][:te.shape[1]]
    return stem, cd.simulate(np.asarray(t['magnetics_plasma_current_times'][0].as_py(), float),
                             np.asarray(t['magnetics_plasma_current'][0].as_py(), float),
                             np.asarray(t['thomson_core_times'][0].as_py(), float), zc, te,
                             np.asarray(t['efit_times'][0].as_py(), float))


def main():
    splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
    reserved = set(json.loads((REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems'])
    pool_stems = sorted((s for s in splits['training'] if s not in reserved),
                        key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:SHOTS]
    with Pool(6) as p:
        extra = dict(p.map(feats, pool_stems))
    with np.load(TABLE) as t:
        stems = t['stems'].astype(str)
        sor = t['shot_of_row']
        names = t['label_names'].astype(str).tolist()
        counts = np.bincount(sor, minlength=len(stems))
        starts = np.concatenate(([0], np.cumsum(counts)))
        where = {s: i for i, s in enumerate(stems)}
        rows, shot, add = [], [], []
        for s in pool_stems:
            i = where[s]
            n = counts[i]
            assert extra[s].shape[0] == n, s
            fr = np.arange(0, n, STRIDE)
            rows.append(starts[i] + fr); shot.append(np.full(len(fr), i)); add.append(extra[s][fr])
        rows = np.concatenate(rows); shot = np.concatenate(shot); add = np.concatenate(add)
        x = t['features'][rows].astype(np.float32)
        y = t['labels'][rows][:, [names.index(n) for n in ('li_scored', 'kappa', 'r_axis')]].astype(np.float64)
    fold = np.asarray([int(hashlib.sha256(stems[i].encode()).hexdigest(), 16) % 5 for i in shot])
    xp = np.column_stack([x, add.astype(np.float32)])
    out = {'shots': SHOTS, 'rows': int(len(rows)), 'diffusion_finite_share': float(np.isfinite(add[:, 0]).mean())}
    for c, target in enumerate(('li_scored', 'kappa', 'r_axis')):
        ok = np.isfinite(y[:, c])
        res = {}
        for tag, xx in (('base', x), ('plus_diffusion', xp)):
            pred = np.full(len(y), np.nan)
            for k in range(5):
                fit, held = ok & (fold != k), ok & (fold == k)
                m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                  early_stopping=False, random_state=0)
                pred[held] = m.fit(xx[fit], y[fit, c]).predict(xx[held])
            r = y[ok, c]
            res[tag] = float(1 - np.sum((pred[ok] - r) ** 2) / np.sum((r - r.mean()) ** 2))
        out[target] = res
        print(target, res, flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
