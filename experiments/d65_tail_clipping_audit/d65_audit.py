# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D65: score-weighted tail table and input clipping audit of the deployed heads (measurement only)
"""python d65_audit.py PACKAGE_MODELS_DIR WORKERS; writes reports/d65_tail_clipping_358.json under the D39 run directory."""
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

import d39_common as c

TARGETS = ('q95', 'betaN', 'li_scored', 'r_axis', 'z_axis', 'kappa')
TRUTH_COL = {'r_axis': 0, 'z_axis': 1, 'kappa': 2, 'li_scored': 6}
SHARE = {'q95': .46, 'betaN': .87}
NORM = {}
OTHER = {}


def init(models):
    for path in sorted(glob.glob(models + '/**/network.npz', recursive=True)):
        with np.load(path) as n:
            if n['feature_mean'].shape == (582,):  # amendment 1: only networks that read the cached 582-feature table
                NORM[path] = (n['feature_mean'].astype(np.float64), n['feature_scale'].astype(np.float64))
            else:
                OTHER[path] = int(n['feature_mean'].shape[0])


def one(case):
    cache = c.load_cache(case['stem'])
    names = cache.label_names.astype(str).tolist()
    labels = {name: cache.labels[:, i].astype(float).copy() for i, name in enumerate(names)}
    for i, name in enumerate(cache.tree_names.astype(str).tolist()):
        labels[name] += c.TREE_WEIGHTS[name] * (cache.tree[:, i] - labels[name])
    table = pq.read_table(case['path'], columns=['efit_q95', 'efit_beta_n'], use_threads=False)
    derived = np.asarray(c.reference_of(case['stem'])[1], dtype=float)
    pred = {'q95': (1 - SHARE['q95']) * labels['q95'] + SHARE['q95'] * cache.joint_q95,
            'betaN': (1 - SHARE['betaN']) * labels['beta_n'] + SHARE['betaN'] * cache.joint_beta_n}
    truth = {'q95': np.asarray(table['efit_q95'][0].as_py(), float), 'betaN': np.asarray(table['efit_beta_n'][0].as_py(), float)}
    for name, col in TRUTH_COL.items():
        pred[name] = labels[name]
        truth[name] = derived[:, col]
    f = cache.features.astype(np.float64)
    clipped_nets = np.zeros(len(f)); counts = np.zeros(f.shape[1])
    for mean, scale in NORM.values():
        over = np.abs((f - mean) / scale) > 8
        clipped_nets += over.any(axis=1)
        counts += over.sum(axis=0)
    n = len(f)
    t = cache.efit_times.astype(float)
    return {'stem': case['stem'], 'screening': case['index'] % 3 == 0, 'n': n,
            'err': {k: (pred[k] - truth[k]).tolist() for k in TARGETS}, 'truth': {k: truth[k].tolist() for k in TARGETS},
            'clip_share': (clipped_nets / len(NORM)).tolist(), 'feature_clip_counts': counts.tolist(),
            'clipped_features_per_frame': [np.flatnonzero(np.abs((f[i] - NORM[next(iter(NORM))][0]) / NORM[next(iter(NORM))][1]) > 8).tolist() for i in range(n)],
            'ip': np.abs(np.asarray(cache.ip_A, float)).tolist(), 'pos': (np.arange(n) / max(n - 1, 1)).tolist(),
            'to_end_s': ((t[-1] - t) / (1e3 if t[-1] > 100 else 1)).tolist()}


def main():
    models, workers = sys.argv[1], int(sys.argv[2])
    init(models)
    ref = next(iter(NORM.values()))
    agree = {'networks': len(NORM), 'excluded_other_width': {p.split('/models/')[-1]: w for p, w in OTHER.items()}, 'max_abs_mean_diff': max(float(np.max(np.abs(m - ref[0]))) for m, _ in NORM.values()),
             'max_rel_scale_diff': max(float(np.max(np.abs(s / ref[1] - 1))) for _, s in NORM.values())}
    with ProcessPoolExecutor(workers, initializer=init, initargs=(models,)) as pool:
        rows = list(pool.map(one, c.cases(), chunksize=4))
    shot = np.concatenate([np.full(r['n'], k) for k, r in enumerate(rows)])
    cs = np.concatenate([r['clip_share'] for r in rows]); ip = np.concatenate([r['ip'] for r in rows])
    pos = np.concatenate([r['pos'] for r in rows]); toend = np.concatenate([r['to_end_s'] for r in rows])
    last3 = np.concatenate([np.arange(r['n']) >= r['n'] - 3 for r in rows])
    feats = [x for r in rows for x in r['clipped_features_per_frame']]
    report = {'protocol': 'protocol_d65.json', 'shots': len(rows), 'frames': int(shot.size), 'normalisers': agree,
              'frames_clipped_any_network': float((cs > 0).mean()), 'frames_clipped_half_networks': float((cs >= .5).mean()),
              'feature_clip_counts_top': sorted(((int(v), i) for i, v in enumerate(np.sum([r['feature_clip_counts'] for r in rows], axis=0)) if v > 0), reverse=True)[:25],
              'targets': {}}
    for k in TARGETS:
        e = np.concatenate([r['err'][k] for r in rows]); ok = np.isfinite(e)
        se = np.where(ok, e, 0.) ** 2; sse = se.sum()
        part = lambda m: {'frame_share': float((m & ok).sum() / ok.sum()), 'sse_share': float(se[m].sum() / sse)}
        half = part(cs >= .5)
        block = {'defined': int(ok.sum()), 'clipped_any': part(cs > 0), 'clipped_half': half,
                 'last3': part(last3), 'last_0p5s': part(toend <= .5),
                 'rule_candidate': bool(half['sse_share'] >= .10 and half['sse_share'] >= 3 * max(half['frame_share'], 1e-12))}
        order = np.argsort(-se)
        for q in (.001, .01, .05):
            pick = order[:max(1, int(q * ok.sum()))]
            block[f'worst_{q}'] = {'frames': int(pick.size), 'sse_share': float(se[pick].sum() / sse),
                                   'clipped_any_share': float((cs[pick] > 0).mean()), 'clipped_half_share': float((cs[pick] >= .5).mean()),
                                   'median_ip_MA': float(np.median(ip[pick]) / 1e6), 'median_pos': float(np.median(pos[pick])),
                                   'share_last_0p5s': float((toend[pick] <= .5).mean())}
        top = order[:max(1, int(.01 * ok.sum()))]
        cnt = {}
        for i in top:
            for j in feats[i]:
                cnt[j] = cnt.get(j, 0) + 1
        block['features_clipped_in_worst_1pct'] = sorted(((v, j) for j, v in cnt.items()), reverse=True)[:15]
        per_shot = np.bincount(shot, weights=se, minlength=len(rows))
        block['worst_shots'] = [{'stem': rows[s]['stem'], 'screening': rows[s]['screening'], 'sse_share': float(per_shot[s] / sse),
                                 'frames': rows[s]['n'], 'clipped_any_share': float((cs[shot == s] > 0).mean())}
                                for s in np.argsort(-per_shot)[:10]]
        report['targets'][k] = block
    out = c.X / 'reports/d65_tail_clipping_358.json'
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
