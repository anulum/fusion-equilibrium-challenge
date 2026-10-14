# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D49: anatomy of the q95 / betaN head error on the 358 development shots
"""Measurement only: nothing is fitted and nothing is selected.

The cohort is the D38 development set (120 screening shots exposed many times, 238 used once by D43).
Reads the cached raw answers of the deployed 69 map networks and the 8 joint networks, and the labels.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

import d39_common as c

SHARE = c.JOINT_SHARE


def one(case):
    cache = c.load_cache(case['stem'])
    names = list(cache.label_names)
    table = pq.read_table(case['path'], columns=['efit_q95', 'efit_beta_n'], use_threads=False)
    truth = {k: np.asarray(table[k][0].as_py(), dtype=float) for k in ('efit_q95', 'efit_beta_n')}
    out = {'stem': case['stem'], 'index': case['index'], 'n': int(len(cache.efit_times)),
           'time': cache.efit_times.tolist(), 'ip': np.asarray(cache.ip_A, dtype=float).tolist()}
    for key, label, joint in (('q95', 'q95', cache.joint_q95), ('betaN', 'beta_n', cache.joint_beta_n)):
        head = cache.labels[:, names.index(label)].astype(float)
        out[key] = {'truth': truth['efit_' + label].tolist(), 'map': head.tolist(),
                    'joint': np.asarray(joint, dtype=float).tolist()}
    return out


def r2(truth, prediction, mean):
    return 1 - np.sum((truth - prediction) ** 2) / np.sum((truth - mean) ** 2)


def main():
    cases = c.cases()
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(one, cases, chunksize=4))
    _, total, means = c.statistics()
    report = {'shots': len(rows), 'frames': sum(r['n'] for r in rows), 'scorer_means': np.asarray(means[1]).tolist()}
    for j, key in enumerate(('q95', 'betaN')):
        t = np.concatenate([r[key]['truth'] for r in rows])
        m = np.concatenate([r[key]['map'] for r in rows])
        jn = np.concatenate([r[key]['joint'] for r in rows])
        ip = np.abs(np.concatenate([r['ip'] for r in rows]))
        shot = np.concatenate([np.full(r['n'], k) for k, r in enumerate(rows)])
        pos = np.concatenate([np.arange(r['n']) / max(r['n'] - 1, 1) for r in rows])
        ok = np.isfinite(t)
        mean = float(means[1][j])
        d = (1 - SHARE) * m + SHARE * jn
        e = d - t
        sse = np.sum(e[ok] ** 2)
        block = {'defined': int(ok.sum()), 'truth_mean': float(t[ok].mean()), 'truth_sd': float(t[ok].std()),
                 'truth_min': float(t[ok].min()), 'truth_max': float(t[ok].max()),
                 'r2_map': float(r2(t[ok], m[ok], mean)), 'r2_joint': float(r2(t[ok], jn[ok], mean)),
                 'r2_deployed': float(r2(t[ok], d[ok], mean)),
                 'error_corr_map_joint': float(np.corrcoef((m - t)[ok], (jn - t)[ok])[0, 1]),
                 'bias': float(e[ok].mean()), 'rmse': float(np.sqrt(np.mean(e[ok] ** 2))),
                 'median_abs': float(np.median(np.abs(e[ok]))),
                 'shares_of_sse_by_joint_share': {str(s): float(r2(t[ok], ((1 - s) * m + s * jn)[ok], mean))
                                                  for s in (0, .25, .5, .75, 1)}}
        order = np.sort(e[ok] ** 2)[::-1]
        block['sse_share_of_worst_frames'] = {str(f): float(order[:max(1, int(f * order.size))].sum() / sse)
                                              for f in (.001, .01, .05, .1, .25)}
        per_shot = np.array([np.sum(e[ok & (shot == k)] ** 2) for k in range(len(rows))])
        order_s = np.sort(per_shot)[::-1]
        block['sse_share_of_worst_shots'] = {str(n): float(order_s[:n].sum() / sse) for n in (1, 3, 10, 30, 100)}
        worst = np.argsort(per_shot)[::-1][:10]
        block['worst_shots'] = [{'stem': rows[k]['stem'], 'screening': rows[k]['index'] % 3 == 0, 'frames': rows[k]['n'],
                                 'sse_share': float(per_shot[k] / sse),
                                 'bias': float(np.nanmean(e[shot == k])),
                                 'truth_range': [float(np.nanmin(t[shot == k])), float(np.nanmax(t[shot == k]))]}
                                for k in worst]
        # within-shot bias against frame-to-frame part
        bias_s = np.array([e[ok & (shot == k)].mean() if (ok & (shot == k)).any() else 0. for k in range(len(rows))])
        block['sse_share_shot_bias'] = float(np.sum(bias_s[shot[ok]] ** 2) / sse)
        # lag-1 autocorrelation of the error inside shots
        num = den = 0.
        for k in range(len(rows)):
            x = e[(shot == k) & ok]
            if x.size > 3:
                x = x - x.mean()
                num += np.sum(x[1:] * x[:-1]); den += np.sum(x * x)
        block['lag1_autocorrelation_within_shot'] = float(num / den)
        def bins(value, edges, label):
            table = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                pick = ok & (value >= lo) & (value < hi)
                if pick.any():
                    table.append({label: [float(lo), float(hi)], 'frames_share': float(pick.sum() / ok.sum()),
                                  'sse_share': float(np.sum(e[pick] ** 2) / sse), 'bias': float(e[pick].mean()),
                                  'rmse': float(np.sqrt(np.mean(e[pick] ** 2))),
                                  'relative_rmse': float(np.sqrt(np.mean((e[pick] / t[pick]) ** 2))) if key == 'q95' else None})
            return table
        qs = np.quantile(t[ok], [0, .1, .25, .5, .75, .9, .95, .99, 1]); qs[-1] += 1e-9
        block['by_truth_quantile'] = bins(t, qs, 'truth')
        block['by_ip_MA'] = bins(ip / 1e6, np.array([0, .4, .6, .8, 1., 1.2, 1.5, 3.]), 'ip')
        block['by_position_in_record'] = bins(pos, np.array([0, .05, .1, .25, .75, .9, .95, 1.0001]), 'pos')
        # truth frame-to-frame jumps: label steps
        jump = np.zeros_like(t)
        start = 0
        for r in rows:
            x = t[start:start + r['n']]
            if r['n'] > 2:
                j2 = np.abs(np.diff(x, prepend=x[0]))
                jump[start:start + r['n']] = np.maximum(j2, np.abs(np.diff(x, append=x[-1])))
            start += r['n']
        js = np.quantile(jump[ok], [0, .5, .9, .99, 1]); js[-1] += 1e-9
        block['by_truth_jump_to_neighbour'] = bins(jump, js, 'jump')
        report[key] = block
    report['r2_qb_deployed'] = (report['q95']['r2_deployed'] + report['betaN']['r2_deployed']) / 2
    out = c.X / 'reports/d49_qb_anatomy_358.json'
    out.write_text(json.dumps(report, indent=1))
    np.savez_compressed(c.X / 'reports/d49_qb_frames_358.npz',
                        **{f'{k}_{f}': np.concatenate([r[k][f] for r in rows]) for k in ('q95', 'betaN') for f in ('truth', 'map', 'joint')},
                        shot=np.concatenate([np.full(r['n'], k) for k, r in enumerate(rows)]),
                        time=np.concatenate([r['time'] for r in rows]), ip=np.concatenate([r['ip'] for r in rows]),
                        stems=np.array([r['stem'] for r in rows]))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    sys.exit(main())
