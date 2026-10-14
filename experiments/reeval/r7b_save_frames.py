# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — R7b: per-frame truth and v25 scalars on the holdout, saved for the noise analysis
"""Diagnosis only. On the 358 holdout shots (never seen by v25), read the seven Consistency scalars of the v25
finished map and of the true map with the organisers' reading, per frame. Split each scalar's squared error into
shot mean (bias), a slow part (centred moving average over 9 frames = 180 ms) and the fast remainder, and report
what an ideal temporal smoother of width w (3, 5, 9 frames) would leave. If the fast part is a large share, a
stronger temporal smoothing of the scalar targets in the finisher is worth a real test."""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import fec_adapter.split as split
from fec_adapter.diii_d_accuracy.shape_edit import Reader

split.refuse_holdout = lambda stem: None  # entrant-approved opening of the holdout part (2026-09-29)

HERE = Path(__file__).resolve().parent
DATA = Path('$FEC_DATA/data/diii_d_train')
ARCH = HERE.parent / 'alldata/holdout_eval/v25/diii_d_train.npz'
OUT = HERE / 'results' / 'r7b_frames.npz'
NAMES = ('R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li')
_R = None


def shot(task):
    global _R
    if _R is None:
        _R = Reader('DIII-D')
    idx, stem = task
    for attempt in range(6):
        try:
            truth = np.asarray(pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])['efit_psirz'][0].as_py(),
                               dtype=np.float64)
            break
        except PermissionError:
            import time
            time.sleep(2 + 3 * attempt)
    with np.load(ARCH) as a:
        pred = a[f'shot_{idx:04d}_psirz'].astype(np.float64)
    assert pred.shape == truth.shape, (stem, pred.shape, truth.shape)
    t = np.full((len(pred), 7), np.nan)
    p = np.full((len(pred), 7), np.nan)
    for f in range(len(pred)):
        rt, rp = _R.read(truth[f]), _R.read(pred[f])
        t[f] = [rt.values[n] for n in NAMES]
        p[f] = [rp.values[n] for n in NAMES]
    return stem, t, p


def smooth(x, w):
    """Centred moving average ignoring NaN, shrinking at the ends."""
    out = np.full_like(x, np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def main():
    stems = sorted(json.loads(Path('$HOME/fusion-equilibrium-challenge/fec_adapter/splits.json').read_text())['holdout'])
    with Pool(3) as pool:
        res = pool.map(shot, list(enumerate(stems)), chunksize=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, stems=np.asarray([r[0] for r in res]), lengths=np.asarray([len(r[1]) for r in res]),
             truth=np.concatenate([r[1] for r in res]), pred=np.concatenate([r[2] for r in res]), names=np.asarray(NAMES))
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
