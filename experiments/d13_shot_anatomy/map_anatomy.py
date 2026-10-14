# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D13b: where the error of the scalars READ FROM OUR MAPS sits along the shot (400 OOF shots)
"""Anatomy of the Consistency scalars and the boundary distance of the finished out-of-fold maps (R8 V0 rows).

Rows: reeval/r8/results/rows_V0_fold{k}.json, every second frame of 80 held-out shots per fold: D, the seven scalars
read from our finished map (p) and from the true map (t). The network head of the same frame comes from nn_oof.npz.
No fit; the in-regime least-squares weight of 'map reading + w (head - map reading)' only indicates whether more or
less trust in the head would have helped in that regime.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE.parent
NAMES = ['r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored']


def main():
    z = np.load(C / 'd9_gbm_heads' / 'nn_oof.npz')
    nn, shot_of_row, targets = z['nn'].astype(float), z['shot_of_row'], z['targets'].astype(str).tolist()
    c = np.load(HERE / 'columns.npz')
    stems = c['stems'].astype(str).tolist()
    col = {int(k): c['values'][:, i].astype(float) for i, k in enumerate(c['cols'])}
    start = np.r_[0, np.cumsum(np.bincount(shot_of_row))]
    P, T, H, D, fe, ips, sh = [], [], [], [], [], [], []
    for k in range(5):
        receipt = json.loads((C / f'kfold/oof_fold{k}/receipt.json').read_text())
        rows = json.loads((C / f'reeval/r8/results/rows_V0_fold{k}.json').read_text())
        by = {}
        for r in rows:
            by.setdefault(r['shot'], []).append(r)
        for key, lst in by.items():
            stem = receipt['stems'][int(key.split(':')[1])]
            s = stems.index(stem)
            n = start[s + 1] - start[s]
            frames = np.arange(len(lst)) * 2
            assert frames[-1] < n and n - frames[-1] <= 2, (stem, n, len(lst))
            idx = start[s] + frames
            P.append([r['p'] for r in lst]); T.append([r['t'] for r in lst]); D.append([r['D'] for r in lst])
            H.append(nn[idx][:, [targets.index(t) for t in NAMES]])
            fe.append(n - 1 - frames); ips.append(col[62][idx]); sh.append(np.full(len(lst), s))
    P, T, H = (np.concatenate([np.asarray(a, float) for a in x]) for x in (P, T, H))
    D, fe, ips, sh = (np.concatenate([np.asarray(a, float) for a in x]) for x in (D, fe, ips, sh))
    tail = (fe < 25) | (ips < 0.6)
    print(f'frames {len(D)}, shots {len(np.unique(sh))}, tail share {tail.mean():.3f}, last-3-frames share {(fe < 3).mean():.3f}')
    print(f'D_LCFS: mean {D.mean():.5f}; tail mean {D[tail].mean():.5f}, rest {D[~tail].mean():.5f}; share of sum(D) in tail '
          f'{D[tail].sum() / D.sum():.3f}; last 25 frames {D[fe < 25].sum() / D.sum():.3f}; Ip<0.6 {D[ips < 0.6].sum() / D.sum():.3f}')
    out = {}
    for j, name in enumerate(NAMES):
        t, p, h = T[:, j], P[:, j], H[:, j]
        ok = np.isfinite(t) & np.isfinite(p)
        se, sh_ = (p - t) ** 2, (h - t) ** 2
        tot = se[ok].sum()
        var = np.sum((t[ok] - t[ok].mean()) ** 2)
        line = (f"{name:10s} map R2 {1 - tot / var:.4f} head R2 {1 - sh_[ok].sum() / var:.4f} | map SSE share: last 3 "
                f"{se[ok & (fe < 3)].sum() / tot:.2f}, last 25 {se[ok & (fe < 25)].sum() / tot:.2f}, Ip<0.6 {se[ok & (ips < 0.6)].sum() / tot:.2f}, tail {se[ok & tail].sum() / tot:.2f} |")
        for label, m in (('tail', ok & tail), ('rest', ok & ~tail)):
            d = h[m] - p[m]
            w = np.dot(d, t[m] - p[m]) / np.dot(d, d)
            line += f" {label}: head/map SSE {sh_[m].sum() / se[m].sum():.2f}, w toward head {w:+.2f} -> {np.sum((p[m] + np.clip(w, 0, 1) * d - t[m]) ** 2) / se[m].sum():.3f};"
        print(line)
        out[name] = line
    (HERE / 'map_anatomy.txt').write_text('\n'.join(out.values()))


main()
