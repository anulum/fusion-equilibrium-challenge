# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D48 stage 1: head-level report of X-specialist arms on the 120 screening shots
"""Equal-shot mean distance of X targets to the true-map saddles: base, D40 static 8, and each named arm, raw and at share .5."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '$FEC_RUNS/ml350/runs/d39_finisher_20261011/src')
from d39_common import X, TREE_WEIGHTS, cases, load_cache
import predict_diii_d as pd
from fec_adapter.diii_d_accuracy import x_point
from fec_adapter.diii_d_accuracy.shape_edit import Reader

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ARMS = sys.argv[3:]
LABELS = x_point.LABELS
Y = X.parent / 'd43_deploy_20261012'
truth_file = ROOT / 'truth_x_120.npz'
selected = [c for c in cases() if c['index'] % 3 == 0]
if truth_file.exists():
    with np.load(truth_file, allow_pickle=False) as saved:
        truth = {s: saved[s] for s in saved.files}
else:
    truth, reader = {}, None
    for case in selected:
        psi = np.asarray(pq.read_table(case['path'], columns=['efit_psirz'], use_threads=False)['efit_psirz'][0].as_py(), dtype=float)
        reader = reader or Reader(str(load_cache(case['stem']).source))
        truth[case['stem']] = np.array([x_point.x_point_positions(reader, m) for m in psi])
    np.savez(truth_file, **truth)
models = {'D40_static8': ([pd.Network.load(Y / 'disjoint_static' / f'static_{s}' / 'network.npz') for s in range(500, 508)], 32)}
for arm in ARMS:
    models[arm] = ([pd.Network.load(ROOT / arm / f'x_{s}' / 'network.npz') for s in range(700, 708)], 0)
per_shot = {name: {'raw': [], 'blend': []} for name in models}
base_shot, counts = [], []
for case in selected:
    cache = load_cache(case['stem']); t = truth[case['stem']]
    names = cache.label_names.astype(str).tolist(); tree_names = cache.tree_names.astype(str).tolist()
    base = np.column_stack([cache.labels[:, names.index(n)].astype(float) for n in LABELS])
    base[:, 0] += TREE_WEIGHTS['x_lower_r'] * (cache.tree[:, tree_names.index('x_lower_r')] - base[:, 0])
    features = cache.features.astype(float)

    def distances(prediction):
        out = []
        for end in (0, 2):
            ok = np.isfinite(t[:, end]) & np.isfinite(t[:, end + 1])
            out.append(np.hypot(prediction[ok, end] - t[ok, end], prediction[ok, end + 1] - t[ok, end + 1]).mean() if ok.any() else np.nan)
        return out

    base_shot.append(distances(base))
    for name, (nets, pad) in models.items():
        inputs = np.concatenate([features, np.zeros((len(features), pad))], axis=1) if pad else features
        raw = np.mean([np.column_stack([net.predict(inputs)[1][n] for n in LABELS]) for net in nets], axis=0)
        per_shot[name]['raw'].append(distances(raw)); per_shot[name]['blend'].append(distances(.5 * base + .5 * raw))
base_mean = np.nanmean(np.array(base_shot), axis=0)
report = {'shots': len(selected), 'base_equal_shot_m': base_mean.tolist(), 'arms': {}}
print('base lower %.3f upper %.3f mm' % tuple(base_mean * 1000))
for name in models:
    raw = np.nanmean(np.array(per_shot[name]['raw']), axis=0); blend = np.nanmean(np.array(per_shot[name]['blend']), axis=0)
    pct = 100 * (1 - blend / base_mean); predicted = 4.47 * pct[0] + 2.76 * pct[1]
    report['arms'][name] = {'raw_equal_shot_m': raw.tolist(), 'blend_equal_shot_m': blend.tolist(), 'percent_reduction_blend': pct.tolist(), 'predicted_microS': float(predicted)}
    print('%-12s raw %.3f / %.3f mm  blend %.3f / %.3f mm  reduction %.2f / %.2f pct  P %.1f' % (name, *(raw * 1000), *(blend * 1000), *pct, predicted), flush=True)
with OUT.open('x') as stream:
    json.dump(report, stream, indent=1)
