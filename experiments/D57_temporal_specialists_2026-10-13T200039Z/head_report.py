# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D57 stage 0: whole-shot specialists against their per-frame control on the development shots
"""Apply the frozen rule of protocol_frozen_v1.json.

For both arms: mean of the 8 members on each of the 358 development shots (whole shot in one pass), blended .5 with
the deployed heads (tree-adjusted geometry heads; .75 map + .25 joint for q95 and betaN). Predicted score change of
a blend against the deployed heads: P = 11.0 pct(axis) + 10.6 pct(li) + 4.9 pct(kappa) + 2.1 pct(volume)
+ 1.1 pct(tri) microS (pct = percent reduction of the pooled rms error, D45 coefficients) + 75000 x (dR2 q95 +
dR2 betaN) microS. Paired bootstrap over whole shots of P(temporal) - P(control).

    python head_report.py FITS_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '$FEC_RUNS/ml350/runs/d39_finisher_20261011/src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d39_common import X, TREE_WEIGHTS, JOINT_SHARE, cases, load_cache, reference_of
import jax
import jax.numpy as jnp
import temporal_model as model

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LABELS = ('r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored', 'q95', 'beta_n')
SEEDS = list(range(800, 808))
SHARE, RESAMPLES, SEED = 0.5, 10000, 5701
MEANS = {'q95': 5.262893061875164, 'beta_n': 1.2688498003107427}


def load(arm, seed):
    with np.load(ROOT / f'{arm}_{seed}' / 'parameters.npz') as z:
        flat = {k: z[k] for k in z.files}
    tree = {'trunk': [{n: flat[f'trunk/{i}/{n}'] for n in 'wb'} for i in range(model.DEPTH)],
            'blocks': [{k: ({n: flat[f'blocks/{i}/{k}/{n}'] for n in 'wb'} if k not in ('g1', 'g2') else flat[f'blocks/{i}/{k}'])
                        for k in ('q', 'k', 'v', 'o', 'f1', 'f2', 'g1', 'g2')} for i in range(model.BLOCKS)]}
    for name in ('project', 'time', 'head'):
        tree[name] = {n: flat[f'{name}/{n}'] for n in 'wb'}
    assert sum(np.asarray(l).size for l in jax.tree.leaves(tree)) == sum(a.size for a in flat.values())
    return tree


norm = np.load(ROOT / 'normalization.npz')
assert tuple(norm['labels'].astype(str)) == LABELS
members = {arm: [load(arm, seed) for seed in SEEDS] for arm in ('temporal', 'control')}
run = {arm: jax.jit(lambda p, x, s, v, mix=(arm == 'temporal'): model.forward(p, x, s, v, mix)) for arm in members}
frames = np.load(X / 'reports/d49_qb_frames_358.npz')
listed = cases()
assert [c['stem'] for c in listed] == [str(s) for s in frames['stems']]
rows = {k: [] for k in ('base', 'temporal', 'control', 'truth', 'shot')}
for index, case in enumerate(listed):
    cache = load_cache(case['stem'])
    names = cache.label_names.astype(str).tolist(); tree_names = cache.tree_names.astype(str).tolist()
    base = {n: cache.labels[:, names.index(n)].astype(float) for n in LABELS}
    for n in LABELS[:7]:
        if n in TREE_WEIGHTS:
            base[n] += TREE_WEIGHTS[n] * (cache.tree[:, tree_names.index(n)] - base[n])
    base['q95'] = (1 - JOINT_SHARE) * base['q95'] + JOINT_SHARE * cache.joint_q95
    base['beta_n'] = (1 - JOINT_SHARE) * base['beta_n'] + JOINT_SHARE * cache.joint_beta_n
    raw = cache.features.astype(np.float64)
    x = np.clip((raw - norm['feature_mean']) / norm['feature_scale'], -8., 8.).astype(np.float32)[None]
    seconds = raw[:, int(norm['since_column'])].astype(np.float32)[None]
    valid = np.ones((1, x.shape[1]), dtype=bool)
    # pad to a multiple of 64 frames so that few shapes are compiled
    pad = (-x.shape[1]) % 64
    xp, sp, vp = np.pad(x, ((0, 0), (0, pad), (0, 0))), np.pad(seconds, ((0, 0), (0, pad))), np.pad(valid, ((0, 0), (0, pad)))
    for arm in members:
        out = np.mean([np.asarray(run[arm](p, xp, sp, vp))[0, :x.shape[1]] for p in members[arm]], axis=0)
        rows[arm].append(out.astype(np.float64) * norm['output_scale'] + norm['output_mean'])
    geometry = np.asarray(reference_of(case['stem'])[1], dtype=float)
    pick = frames['shot'] == index
    truth = np.column_stack([geometry, frames['q95_truth'][pick], frames['betaN_truth'][pick]])
    assert truth.shape == rows['temporal'][-1].shape
    rows['base'].append(np.column_stack([base[n] for n in LABELS])); rows['truth'].append(truth)
    rows['shot'].append(np.full(len(truth), index))
base, temporal, control, truth, shot = (np.concatenate(rows[k]) for k in ('base', 'temporal', 'control', 'truth', 'shot'))
shots = len(listed)
okay = np.isfinite(truth)


def per_shot(prediction):
    error = np.where(okay, prediction - np.nan_to_num(truth), 0.0) ** 2
    return np.stack([np.bincount(shot, weights=error[:, j], minlength=shots) for j in range(9)], axis=1)


count = np.stack([np.bincount(shot, weights=okay[:, j].astype(float), minlength=shots) for j in range(9)], axis=1)
total = np.stack([np.bincount(shot, weights=np.where(okay[:, j], (np.nan_to_num(truth[:, j]) - MEANS[n]) ** 2, 0.0), minlength=shots)
                  for j, n in ((7, 'q95'), (8, 'beta_n'))], axis=1)
sse = {'base': per_shot(base), 'temporal_raw': per_shot(temporal), 'control_raw': per_shot(control),
       'temporal': per_shot(SHARE * temporal + (1 - SHARE) * base), 'control': per_shot(SHARE * control + (1 - SHARE) * base)}


def predicted(weights, arm):
    """weights (R, shots) -> predicted microS of the arm's blend against the deployed heads, per resample."""
    n = weights @ count
    rms = lambda key: np.sqrt((weights @ sse[key]) / n)
    pct = 100 * (1 - rms(arm) / rms('base'))
    geometry = 11.0 * (pct[:, 0] + pct[:, 1]) / 2 + 4.9 * pct[:, 2] + 1.1 * (pct[:, 3] + pct[:, 4]) / 2 + 2.1 * pct[:, 5] + 10.6 * pct[:, 6]
    scalar = 75000.0 * np.sum(((weights @ sse['base'])[:, 7:] - (weights @ sse[arm])[:, 7:]) / (weights @ total), axis=1)
    return geometry + scalar, pct


one = np.ones((1, shots))
rng = np.random.default_rng(SEED)
resample = np.stack([np.bincount(p, minlength=shots) for p in rng.integers(0, shots, size=(RESAMPLES, shots))]).astype(float)
report = {'frames': int(len(truth)), 'shots': shots, 'labels': list(LABELS), 'share': SHARE}
for arm in ('temporal', 'control'):
    value, pct = predicted(one, arm)
    raw_ratio = np.sqrt(sse[arm + '_raw'].sum(axis=0) / sse['base'].sum(axis=0))
    corr = [float(np.corrcoef((eval(arm) - truth)[okay[:, j], j], (base - truth)[okay[:, j], j])[0, 1]) for j in range(9)]
    report[arm] = {'predicted_microS': float(value[0]), 'percent_reduction_blend': pct[0].tolist(),
                   'raw_rms_over_base': raw_ratio.tolist(), 'error_correlation_raw_base': corr}
difference = predicted(resample, 'temporal')[0] - predicted(resample, 'control')[0]
point = report['temporal']['predicted_microS'] - report['control']['predicted_microS']
report.update({'temporal_minus_control_microS': point, 'ci95': [float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))],
               'P_positive': float(np.mean(difference > 0)),
               'temporal_over_control_raw_rms': np.sqrt(sse['temporal_raw'].sum(axis=0) / sse['control_raw'].sum(axis=0)).tolist(),
               'screening_120_difference': float((predicted((np.arange(shots) % 3 == 0)[None].astype(float), 'temporal')[0] - predicted((np.arange(shots) % 3 == 0)[None].astype(float), 'control')[0])[0]),
               'other_238_difference': float((predicted((np.arange(shots) % 3 != 0)[None].astype(float), 'temporal')[0] - predicted((np.arange(shots) % 3 != 0)[None].astype(float), 'control')[0])[0]),
               'resamples': RESAMPLES, 'seed': SEED})
report['passes'] = bool(point >= 30.0 and report['P_positive'] >= 0.95 and report['temporal']['predicted_microS'] > 0.0)
with OUT.open('x') as stream:
    json.dump(report, stream, indent=1)
print(json.dumps(report, indent=1))
