# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — R7c: the reference's own frame-to-frame variation and the ceiling it implies
"""Diagnosis only, on the 358 holdout shots (v25 never saw them). Per scalar of the Consistency term:

1. truth = slow (centred 9-frame mean) + fast; the fast share of the truth's variance about its pooled mean;
2. whiteness of the fast part: lag-1 autocorrelation (a fit jitter is near 0 or negative; a physical event is
   positive over neighbouring frames);
3. do our predictions follow the truth's fast part? correlation of pred_fast with truth_fast;
4. ceilings: R2 of a predictor that equals the truth's slow part exactly (it misses only the fast part) — the
   best any method can do if the fast part is unpredictable — against our R2;
5. is the fast part tied to input activity? rank correlation of |truth_fast| with the frame's |dIp/dt| and with
   the time since the nearest Thomson frame.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
FRAMES = HERE / 'results' / 'r7b_frames.npz'
DATA = Path('$FEC_DATA/data/diii_d_train')
OUT = HERE / 'results' / 'r7c_reference_noise.json'
W = 9


def smooth(x, w=W):
    out = np.full_like(x, np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def lag1(x):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    x = x - x.mean()
    d = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / d) if d > 0 else np.nan


def activity(stem):
    """Per frame: |dIp/dt| (corrected clock) and time to the nearest core Thomson sample, in ms."""
    from fec_adapter import machine
    from fec_adapter.shot_reader import read_row
    row = read_row(DATA / f'{stem}.parquet')
    t = np.asarray(row['efit_times'], float)
    it = np.asarray(machine.plasma_current_times(row), float)
    iv = np.abs(np.asarray(row['magnetics_plasma_current'], float))
    didt = np.abs(np.interp(t + 5, it, iv) - np.interp(t - 5, it, iv)) / 10.0
    th = np.asarray(row['thomson_core_times'], float)
    gap = np.array([np.min(np.abs(th - x)) if th.size else np.nan for x in t])
    return didt, gap


def main():
    z = np.load(FRAMES)
    names = [str(n) for n in z['names']]
    lengths, stems = z['lengths'], [str(s) for s in z['stems']]
    starts = np.concatenate(([0], np.cumsum(lengths)))
    truth, pred = z['truth'], z['pred']
    acts = [activity(s) for s in stems]
    didt = np.concatenate([a[0] for a in acts])
    gap = np.concatenate([a[1] for a in acts])
    out = {'shots': len(stems), 'frames': int(len(truth)), 'window_frames': W}
    for c, name in enumerate(names):
        tf, pf, ts, lags = [], [], [], []
        for i in range(len(stems)):
            t = truth[starts[i]:starts[i + 1], c]
            p = pred[starts[i]:starts[i + 1], c]
            s_t, s_p = smooth(t), smooth(p)
            tf.append(t - s_t); pf.append(p - s_p); ts.append(s_t)
            lags.append(lag1(t - s_t))
        tf, pf, ts = np.concatenate(tf), np.concatenate(pf), np.concatenate(ts)
        t_all, p_all = truth[:, c], pred[:, c]
        ok = np.isfinite(t_all) & np.isfinite(p_all) & np.isfinite(tf) & np.isfinite(pf)
        var = float(np.sum((t_all[ok] - t_all[ok].mean()) ** 2))
        r2_ours = 1 - float(np.sum((p_all[ok] - t_all[ok]) ** 2)) / var
        r2_slow_oracle = 1 - float(np.sum(tf[ok] ** 2)) / var
        rho_d = spearmanr(np.abs(tf[ok]), didt[ok]).correlation
        rho_g = spearmanr(np.abs(tf[ok]), gap[ok]).correlation
        out[name] = {'truth_fast_share_of_variance': float(np.sum(tf[ok] ** 2)) / var,
                     'truth_fast_lag1_median': float(np.nanmedian(lags)),
                     'corr_pred_fast_truth_fast': float(np.corrcoef(pf[ok], tf[ok])[0, 1]),
                     'R2_ours': r2_ours, 'R2_ceiling_if_fast_unpredictable': r2_slow_oracle,
                     'share_of_our_error_that_is_truth_fast': float(np.sum(tf[ok] ** 2)) / float(np.sum((p_all[ok] - t_all[ok]) ** 2)),
                     'spearman_absfast_vs_dIpdt': float(rho_d), 'spearman_absfast_vs_thomson_gap': float(rho_g)}
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}), flush=True)
    cons_ours = np.mean([max(0, out[n]['R2_ours']) for n in names])
    cons_ceiling = np.mean([max(0, out[n]['R2_ceiling_if_fast_unpredictable']) for n in names])
    out['consistency'] = {'ours': cons_ours, 'ceiling_if_fast_unpredictable': cons_ceiling,
                          'S_room_from_consistency': 0.20 * (cons_ceiling - cons_ours)}
    print(json.dumps(out['consistency']), flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
