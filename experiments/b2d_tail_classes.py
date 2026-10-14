# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2d diagnostic: classes of the boundary-distance tail
"""Sort every B2 frame by where its Hausdorff maximum sits (diagnosis only, no fit).

Classes, decided in this order from the point w where the symmetric Hausdorff distance
between the v25 and the true boundary is attained:

* ``corner``   - w within CORNER metres of the dominant (plasma-bounding) saddle of the
                 truth or of v25: the X-point corner itself;
* ``second_x`` - w within SECOND metres of the other saddle of the truth or of v25: the
                 boundary reaching, or not reaching, towards the second X-point;
* ``top`` / ``bottom`` / ``inboard`` / ``outboard`` - elsewhere, by the direction of w
                 from the true axis: a smooth-boundary shape error.

The dominant saddle is the one with the higher phi (the boundary closes at it). The report
gives, per class, the frame count and the share of the summed D and of the summed excess
over 0.0082 (the leaders' mean), for all frames and for the tail D > 0.02.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402
import b2c_saddle_gap_oracle as oracle  # noqa: E402

OUT = HERE / 'results' / 'b2d_tail_classes_v25_validation'
CORNER = 0.04
SECOND = 0.10
LEADERS = 0.0082
CLASSES = ('corner', 'second_x', 'top', 'bottom', 'inboard', 'outboard')


def classify(reader, truth, pred):
    rt, rp = reader.read(truth), reader.read(pred)
    ct, cp = np.asarray(rt.contour, float), np.asarray(rp.contour, float)
    w = b2.worst_point(cp, ct)
    dominant, other = [], []
    for flux in (truth, pred):
        s = [v for v in oracle.saddles(reader, flux).values() if v is not None]
        s.sort(key=lambda v: -v[1])
        if s:
            dominant.append(s[0][0])
        other.extend(v[0] for v in s[1:])
    near = lambda pts, radius: any(np.hypot(*(w - p)) < radius for p in pts)  # noqa: E731
    if near(dominant, CORNER):
        return 'corner'
    if near(other, SECOND):
        return 'second_x'
    angle = np.degrees(np.arctan2(w[1] - rt.axis[1], w[0] - rt.axis[0]))
    if 45 <= angle < 135:
        return 'top'
    if -135 <= angle < -45:
        return 'bottom'
    return 'inboard' if abs(angle) >= 135 else 'outboard'


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        sys.path.insert(0, str(b2.REPO / 'fec_harness'))
        import run_parallel_score as scoring
        stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
        z = np.load(b2.OUT / 'frames.npz')
        keys, dvals = z['keys'], z['data'][:, list(z['columns']).index('D')]
        reader = Reader('DIII-D')
        labels = []
        stage = 'frames'
        with np.load(b2.V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
            for shot in np.unique(keys[:, 0]):
                maps = archive[f'shot_{shot:04d}_psirz']
                truth = np.asarray(pq.read_table(b2.DATA / f'{stems[shot]}.parquet', columns=['efit_psirz'])
                                   ['efit_psirz'][0].as_py(), dtype=np.float64)
                for frame in keys[keys[:, 0] == shot, 1]:
                    labels.append(classify(reader, truth[frame], np.asarray(maps[frame], np.float64)))
        stage = 'summary'
        labels = np.asarray(labels)
        tail = dvals > 0.02
        excess = np.clip(dvals - LEADERS, 0, None)

        def table(mask):
            return {c: {'frames': int((labels[mask] == c).sum()),
                        'share_of_D_sum': float(dvals[mask & (labels == c)].sum() / dvals[mask].sum()),
                        'share_of_excess_over_leaders': float(excess[mask & (labels == c)].sum()
                                                              / excess[mask].sum()),
                        'mean_D': float(dvals[mask & (labels == c)].mean())
                        if (mask & (labels == c)).any() else None}
                    for c in CLASSES}

        summary = {'status': 'diagnostic_complete', 'selection_or_fit': False,
                   'frames': int(len(labels)), 'tail_frames': int(tail.sum()),
                   'corner_radius_m': CORNER, 'second_x_radius_m': SECOND,
                   'all': table(np.ones(len(labels), bool)), 'tail_D_gt_0p02': table(tail),
                   'MAST_targets_used': False, 'reserved_excluded': True}
        np.savez_compressed(OUT / 'labels.npz', labels=labels, keys=keys, D=dvals)
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=1), flush=True)
        code, stage = 0, 'complete'
    except Exception:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (OUT / 'terminal.json').write_text(json.dumps(
            {'actual_exit_code': code, 'stage': stage, 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
