# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D67 report: clip 50 against clip 8, mean of four networks per arm, paired whole-shot bootstrap
"""python d67_report.py PREPARED FITS OUT.json"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

PER_PERCENT = {'r_axis': 11.0, 'z_axis': 11.0, 'li_scored': 10.6, 'kappa': 4.9, 'volume': 2.1, 'tri_top': 1.1, 'tri_bot': 1.1}  # micro-S (D45)
R2_WEIGHT = {'q95': .075, 'beta_n': .075}  # 0.15 x mean of the two R2


def main():
    prepared, fits, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    meta = json.loads((prepared / 'held/metadata.json').read_text())
    names = meta['label_names']
    truth = np.load(prepared / 'held/labels.npy', mmap_mode='r')
    coords_true = np.load(prepared / 'held/coordinates.npy', mmap_mode='r')[:, :64].astype(np.float64)
    shot = np.load(prepared / 'held/shot_of_row.npy')
    arms = {}
    for clip in (8, 50):
        files = sorted(glob.glob(str(fits / f'*_clip{clip}/held_predictions.npz')))
        arms[clip] = files
    members = sorted({Path(f).parent.name.rsplit('_clip', 1)[0] for f in arms[8]} & {Path(f).parent.name.rsplit('_clip', 1)[0] for f in arms[50]})
    assert members, 'no paired member'
    mean = {}
    for clip in (8, 50):
        parts = [np.load(fits / f'{m}_clip{clip}' / 'held_predictions.npz') for m in members]
        mean[clip] = {'coordinates': np.mean([p['coordinates'].astype(np.float64) for p in parts], axis=0)}
        for h in list(PER_PERCENT) + list(R2_WEIGHT):
            mean[clip][h] = np.mean([p['label_' + h].astype(np.float64) for p in parts], axis=0)
    _, inverse = np.unique(shot, return_inverse=True)
    n_shots = inverse.max() + 1
    heads = list(PER_PERCENT) + list(R2_WEIGHT)
    se = {}
    for h in heads:
        t = truth[:, names.index(h)].astype(np.float64); ok = np.isfinite(t)
        se[h] = {clip: np.bincount(inverse[ok], weights=(mean[clip][h][ok] - t[ok]) ** 2, minlength=n_shots) for clip in (8, 50)}
        tt = t[ok]; se[h]['var'] = np.bincount(inverse[ok], weights=(tt - tt.mean()) ** 2, minlength=n_shots)
    sec = {clip: np.bincount(inverse, weights=np.sum((mean[clip]['coordinates'] - coords_true) ** 2, axis=1), minlength=n_shots) for clip in (8, 50)}

    def predicted(weights):
        total, parts = 0.0, {}
        for h in heads:
            a, b, v = (se[h][8] * weights).sum(), (se[h][50] * weights).sum(), (se[h]['var'] * weights).sum()
            if h in PER_PERCENT:
                parts[h] = PER_PERCENT[h] * 100 * (1 - np.sqrt(b / a)) * 1e-6
            else:
                parts[h] = R2_WEIGHT[h] * (a - b) / v
            total += parts[h]
        return total, parts

    point, parts = predicted(np.ones(n_shots))
    rng = np.random.default_rng(0)
    draws = np.array([predicted(np.bincount(rng.integers(0, n_shots, n_shots), minlength=n_shots))[0] for _ in range(20000)])
    flux_ratio = float(sec[50].sum() / sec[8].sum())
    report = {'members_paired': members, 'shots': int(n_shots), 'frames': int(len(shot)),
              'squared_error_ratio_50_over_8': {h: float(se[h][50].sum() / se[h][8].sum()) for h in heads},
              'flux_coordinate_squared_error_ratio': flux_ratio,
              'predicted_change': point, 'predicted_parts': parts,
              'bootstrap_95': np.quantile(draws, [.025, .975]).tolist(), 'P_positive': float((draws > 0).mean()),
              'candidate': bool(point >= 1e-5 and (draws > 0).mean() >= .95 and flux_ratio <= 1.02)}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
