# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D13: head error along the shot's time axis and against discharge parameters (plot data)
"""Aggregate the S16 out-of-fold head errors of all training frames for the timeline page. No fit, no selection."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEADS = ['li_scored', 'q95', 'beta_n', 'r_axis', 'z_axis', 'kappa', 'volume', 'tri_top', 'tri_bot']
SPAN = 75


def rank(v):
    r = np.empty(len(v)); r[np.argsort(v, kind='stable')] = np.arange(len(v)); return r


def main():
    z = np.load(HERE.parent / 'd9_gbm_heads' / 'nn_oof.npz')
    nn, lab, fold, shot = z['nn'].astype(float), z['labels'].astype(float), z['row_fold'], z['shot_of_row']
    targets = z['targets'].astype(str).tolist()
    c = np.load(HERE / 'columns.npz')
    col = {int(k): c['values'][:, i].astype(float) for i, k in enumerate(c['cols'])}
    ips, rate, since, until = col[62], col[63], col[64], col[65]
    n = len(shot)
    first = np.r_[True, shot[1:] != shot[:-1]]; last = np.r_[shot[1:] != shot[:-1], True]
    idx = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))
    fe = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1] - np.arange(n)
    keep = fold >= 0
    out = {'frames': int(keep.sum()), 'shots': int(np.unique(shot[keep]).size), 'heads': {}, 'span': SPAN}
    ip_edges = np.linspace(0, 1.05, 22); rate_edges = np.linspace(-3, 3, 25)
    err = {}
    for name in targets:
        j = targets.index(name); y, p = lab[:, j], nn[:, j]; ok = keep & np.isfinite(y)
        err[name] = np.where(ok, p - y, np.nan)
    for name in HEADS:
        j = targets.index(name); y = lab[:, j]; e = err[name]; ok = np.isfinite(e)
        se = np.where(ok, e ** 2, 0.0); total = se.sum(); std = float(y[ok].std())
        def profile(key, span):
            rm, sh, lm = [], [], []
            for k in range(span + 1):
                s = ok & (key == k)
                rm.append(float(np.sqrt(se[s].sum() / max(s.sum(), 1)) / std)); sh.append(float(se[s].sum() / total))
                lm.append(float(y[s].mean()))
            return rm, sh, lm
        end_r, end_s, end_l = profile(fe, SPAN); st_r, st_s, st_l = profile(idx, SPAN)
        def binned(x, edges):
            w = np.digitize(x, edges) - 1; rm, sh, fr = [], [], []
            for b in range(len(edges) - 1):
                s = ok & (w == b)
                rm.append(float(np.sqrt(se[s].sum() / s.sum()) / std) if s.sum() > 200 else None)
                sh.append(float(se[s].sum() / total)); fr.append(float(s.sum() / ok.sum()))
            return rm, sh, fr
        ip_r, ip_s, ip_f = binned(ips, ip_edges); ra_r, ra_s, ra_f = binned(rate, rate_edges)
        a = np.abs(e[ok]); ra = rank(a)
        dy = np.abs(np.diff(y, prepend=y[0])); dy[first] = 0
        corr = {}
        for label, x in (('Ip / peak', ips), ('dIp/dt / peak', rate), ('|dIp/dt| / peak', np.abs(rate)),
                         ('seconds until end', until), ('seconds since start', since),
                         ('frames from record end', fe.astype(float)), ('frame index in record', idx.astype(float)),
                         ('label value', y), ('|label step| to previous frame', dy)):
            corr[label] = float(np.corrcoef(ra, rank(x[ok]))[0, 1])
        out['heads'][name] = {
            'label_std': std, 'r2': float(1 - total / np.sum((y[ok] - y[ok].mean()) ** 2)),
            'end_rmse': end_r, 'end_sse_share': end_s, 'end_label_mean': end_l,
            'start_rmse': st_r, 'start_sse_share': st_s,
            'ip_rmse': ip_r, 'ip_sse_share': ip_s, 'ip_frames': ip_f, 'rate_rmse': ra_r, 'rate_sse_share': ra_s,
            'rate_frames': ra_f, 'rank_corr_abs_error': corr,
            'sse_last3': float(se[ok & (fe < 3)].sum() / total), 'sse_last25': float(se[ok & (fe < 25)].sum() / total),
            'sse_first3': float(se[ok & (idx < 3)].sum() / total),
            'sse_low_ip': float(se[ok & (ips < 0.6)].sum() / total), 'frames_low_ip': float((ok & (ips < 0.6)).sum() / ok.sum()),
            'sse_rampdown': float(se[ok & (rate < -0.1)].sum() / total)}
    out['ip_centres'] = (0.5 * (ip_edges[1:] + ip_edges[:-1])).round(4).tolist()
    out['rate_centres'] = (0.5 * (rate_edges[1:] + rate_edges[:-1])).round(4).tolist()
    names = HEADS + ['x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
    def cmat(vals):
        m = np.full((len(names), len(names)), None, dtype=object)
        for a_i, a in enumerate(names):
            for b_i, b in enumerate(names):
                s = np.isfinite(vals[a]) & np.isfinite(vals[b])
                m[a_i][b_i] = round(float(np.corrcoef(vals[a][s], vals[b][s])[0, 1]), 3)
        return m.tolist()
    out['corr_names'] = names
    out['corr_frame_errors'] = cmat(err)
    shots = int(shot.max()) + 1
    offs = {}
    for a in names:
        s = np.isfinite(err[a]); cnt = np.bincount(shot[s], minlength=shots).astype(float)
        m = np.bincount(shot[s], weights=err[a][s], minlength=shots) / np.maximum(cnt, 1); m[cnt < 20] = np.nan; offs[a] = m
    out['corr_shot_offsets'] = cmat(offs)
    tailmask = fe < 25
    out['corr_frame_errors_last25'] = cmat({a: np.where(tailmask, err[a], np.nan) for a in names})
    # example shots: the four with the largest li squared error and two typical ones
    e = err['li_scored']; se = np.where(np.isfinite(e), e ** 2, 0.0)
    per = np.bincount(shot, weights=se, minlength=shots); order = np.argsort(per)[::-1]
    cnt = np.bincount(shot[keep], minlength=shots)
    typical = [int(s) for s in np.argsort(np.abs(per - np.median(per[cnt > 100]))) if cnt[s] > 150][:2]
    ex = []
    for s_id in [int(v) for v in order[:4]] + typical:
        rows = np.flatnonzero(shot == s_id)[-120:]
        ex.append({'kind': 'worst' if s_id in order[:4] else 'typical', 'sse_share': float(per[s_id] / se.sum()),
                   'from_end': fe[rows].tolist(), 'ip': ips[rows].round(4).tolist(),
                   **{h: {'label': lab[rows, targets.index(h)].round(4).tolist(), 'head': nn[rows, targets.index(h)].round(4).tolist()}
                      for h in ('li_scored', 'q95', 'beta_n')}})
    out['examples'] = ex
    (HERE / 'timeline_data.json').write_text(json.dumps(out))
    for h, v in out['heads'].items():
        print(f"{h:10s} last3 {v['sse_last3']:.3f} last25 {v['sse_last25']:.3f} first3 {v['sse_first3']:.3f} lowIp {v['sse_low_ip']:.3f} ({v['frames_low_ip']:.3f} frames) rampdown {v['sse_rampdown']:.3f} | " + ' '.join(f"{k[:14]}:{c:+.2f}" for k, c in v['rank_corr_abs_error'].items()))


main()
