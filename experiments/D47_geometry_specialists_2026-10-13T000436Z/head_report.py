# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D47 stage 1: head-level report of the geometry specialists on the 120 screening shots
"""Pooled error of base heads, specialist mean and their .5 blend against the scorer's truth-derived scalars."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '$FEC_RUNS/ml350/runs/d39_finisher_20261011/src')
from d39_common import X, TREE_WEIGHTS, cases, load_cache, reference_of
import predict_diii_d as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
LABELS = ('r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored')
SEEDS = list(range(600, 616))
nets = [pd.Network.load(ROOT / f'geo_{seed}' / 'network.npz') for seed in SEEDS]
assert all(n.coordinates == 0 and tuple(n.label_names) == LABELS for n in nets)
base, truth, members = [], [], []
for case in cases():
    if case['index'] % 3:
        continue
    cache = load_cache(case['stem'])
    names = cache.label_names.astype(str).tolist(); tree_names = cache.tree_names.astype(str).tolist()
    labels = {n: cache.labels[:, names.index(n)].astype(float) for n in LABELS}
    for n in LABELS:
        if n in TREE_WEIGHTS:
            labels[n] += TREE_WEIGHTS[n] * (cache.tree[:, tree_names.index(n)] - labels[n])
    features = cache.features.astype(float)
    members.append(np.stack([np.column_stack([net.predict(features)[1][n] for n in LABELS]) for net in nets]))
    base.append(np.column_stack([labels[n] for n in LABELS]))
    truth.append(np.asarray(reference_of(case['stem'])[1], dtype=float))
base, truth, members = np.concatenate(base), np.concatenate(truth), np.concatenate(members, axis=1)
assert truth.shape == base.shape == members.shape[1:]


def rms(prediction):
    error = prediction - truth
    return np.sqrt(np.nanmean(np.where(np.isfinite(truth), error, np.nan) ** 2, axis=0))


coefficient = {'axis': 11.0, 'li': 10.6, 'kappa': 4.9, 'volume': 2.1, 'tri': 1.1}
report = {'frames': int(len(base)), 'labels': list(LABELS), 'base_rms': rms(base).tolist(), 'prefixes': {},
          'per_seed_raw_rms': [rms(members[i]).tolist() for i in range(len(SEEDS))],
          'finite_truth': np.isfinite(truth).sum(axis=0).astype(int).tolist()}
for size in (4, 8, 16):
    raw = members[:size].mean(axis=0); blend = .5 * base + .5 * raw
    pct = 100 * (1 - rms(blend) / rms(base))
    predicted = (coefficient['axis'] * (pct[0] + pct[1]) / 2 + coefficient['kappa'] * pct[2] + coefficient['tri'] * (pct[3] + pct[4]) / 2
                 + coefficient['volume'] * pct[5] + coefficient['li'] * pct[6])
    okay = np.isfinite(truth)
    corr = [float(np.corrcoef((raw - truth)[okay[:, j], j], (base - truth)[okay[:, j], j])[0, 1]) for j in range(7)]
    report['prefixes'][str(size)] = {'raw_rms': rms(raw).tolist(), 'blend_rms': rms(blend).tolist(), 'percent_reduction_blend': pct.tolist(),
                                     'error_correlation_raw_base': corr, 'predicted_microS': float(predicted)}
    print(size, 'pct', np.round(pct, 2).tolist(), 'raw/base rms ratio', np.round(rms(raw) / rms(base), 3).tolist(), 'P %.1f' % predicted, flush=True)
report['proceed'] = bool(report['prefixes']['16']['predicted_microS'] >= 30.0)
with OUT.open('x') as stream:
    json.dump(report, stream, indent=1)
print(json.dumps({'base_rms': np.round(rms(base), 5).tolist(), 'proceed': report['proceed']}))
