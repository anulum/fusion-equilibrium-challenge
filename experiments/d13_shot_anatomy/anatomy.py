# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D13: where the out-of-fold head error sits, over every training shot and frame
"""Anatomy of the S16 out-of-fold head errors on all 5936 training shots (nn_oof.npz), no fit and no selection.

For each head: concentration over shots and frames, split between a per-shot offset and the rest, and the share of
the squared error by discharge phase, current level, time from the start and to the end, position in the record,
label jumps and label extremes. A bin's 'lift' is its share of the squared error over its share of the frames.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NN = HERE.parent / 'd9_gbm_heads' / 'nn_oof.npz'


def shares(mask, se, ok):
    return float(mask[ok].mean()), float(se[ok][mask[ok]].sum() / se[ok].sum())


def main():
    z = np.load(NN)
    nn, lab, fold, shot = z['nn'].astype(float), z['labels'].astype(float), z['row_fold'], z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    c = np.load(HERE / 'columns.npz')
    assert np.array_equal(c['shot_of_row'], shot)
    col = {int(k): c['values'][:, i].astype(float) for i, k in enumerate(c['cols'])}
    ip_share, rate_share, since, until = col[62], col[63], col[64], col[65]
    n = len(shot)
    first = np.r_[True, shot[1:] != shot[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    index = np.arange(n) - start
    last = np.r_[shot[1:] != shot[:-1], True]
    end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1]
    from_end = end - np.arange(n)
    out = {}
    for name in sys.argv[1:] or targets:
        j = targets.index(name)
        y, p = lab[:, j], nn[:, j]
        ok = np.isfinite(y) & (fold >= 0)
        e = np.where(ok, p - y, 0.0)
        se = e ** 2
        total = se[ok].sum()
        r = {'frames': int(ok.sum()), 'r2': float(1 - total / np.sum((y[ok] - y[ok].mean()) ** 2)),
             'rmse': float(np.sqrt(total / ok.sum())), 'label_std': float(y[ok].std())}
        shots = int(shot.max()) + 1
        cnt = np.bincount(shot[ok], minlength=shots).astype(float)
        sse_shot = np.bincount(shot, weights=se, minlength=shots)
        mean_shot = np.bincount(shot, weights=e, minlength=shots) / np.maximum(cnt, 1)
        r['offset_share_of_sse'] = float(np.sum(cnt * mean_shot ** 2) / total)
        order = np.sort(sse_shot[cnt > 0])[::-1]
        m = order.size
        r['sse_share_of_worst_shots'] = {f'{q}%': float(order[:max(1, int(m * q / 100))].sum() / total) for q in (1, 5, 10, 25)}
        fr = np.sort(se[ok])[::-1]
        r['sse_share_of_worst_frames'] = {f'{q}%': float(fr[:max(1, int(fr.size * q / 100))].sum() / total) for q in (0.1, 1, 5)}
        bins = {}
        for label, mask in (
            ('Ip/peak < 0.3', ip_share < 0.3), ('Ip/peak 0.3-0.6', (ip_share >= 0.3) & (ip_share < 0.6)),
            ('Ip/peak 0.6-0.9', (ip_share >= 0.6) & (ip_share < 0.9)), ('Ip/peak >= 0.9', ip_share >= 0.9),
            ('ramp-up (dIp/dt/peak > +0.1/s)', rate_share > 0.1), ('flat (|rate| <= 0.1/s)', np.abs(rate_share) <= 0.1),
            ('ramp-down (< -0.1/s)', rate_share < -0.1),
            ('since start < 0.3 s', since < 0.3), ('since start 0.3-1 s', (since >= 0.3) & (since < 1.0)),
            ('since start 1-3 s', (since >= 1.0) & (since < 3.0)), ('since start >= 3 s', since >= 3.0),
            ('until end < 0.2 s', until < 0.2), ('until end 0.2-0.5 s', (until >= 0.2) & (until < 0.5)),
            ('first 3 frames of record', index < 3), ('last 3 frames of record', from_end < 3),
        ):
            a, b = shares(mask, se, ok)
            bins[label] = {'frames': round(a, 4), 'sse': round(b, 4), 'lift': round(b / max(a, 1e-12), 2)}
        dy = np.abs(np.diff(y, prepend=y[0]))
        dy[first] = 0.0
        dy[~np.isfinite(dy)] = 0.0
        typical = np.median(dy[ok & ~first])
        for mult in (10, 30):
            jump = dy > mult * typical
            near = jump | np.r_[jump[1:], False]
            a, b = shares(near, se, ok)
            bins[f'label jump > {mult}x median step (frame or the one before)'] = {
                'frames': round(a, 4), 'sse': round(b, 4), 'lift': round(b / max(a, 1e-12), 2)}
        lo, hi = np.percentile(y[ok], [0.5, 99.5])
        a, b = shares((y < lo) | (y > hi), se, ok)
        bins['label outside its 0.5-99.5 % range'] = {'frames': round(a, 4), 'sse': round(b, 4), 'lift': round(b / a, 2),
                                                      'range': [float(lo), float(hi)]}
        r['bins'] = bins
        r['signed_bias_by_bin'] = {k: float(e[ok & m].mean()) for k, m in (
            ('Ip/peak < 0.3', ip_share < 0.3), ('ramp-up', rate_share > 0.1), ('ramp-down', rate_share < -0.1),
            ('flat', np.abs(rate_share) <= 0.1))}
        out[name] = r
        print(f"\n=== {name}: R2 {r['r2']:.5f} rmse {r['rmse']:.4g} (label std {r['label_std']:.4g}); per-shot offset "
              f"{r['offset_share_of_sse']:.2f} of SSE; worst shots {r['sse_share_of_worst_shots']}; worst frames "
              f"{r['sse_share_of_worst_frames']}")
        for k, v in bins.items():
            print(f"  {k:58s} frames {v['frames']:.4f}  sse {v['sse']:.4f}  lift {v['lift']:6.2f}")
        print('  bias:', {k: round(v, 5) for k, v in r['signed_bias_by_bin'].items()})
    (HERE / 'anatomy.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
