# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2b diagnostic: what the heavy tail of the boundary distance is
"""Compare boundary-end topology of truth and v25 on the B2 tail frames (D > 0.02) and an
equal random sample of the other B2 frames. Diagnosis only; same frames as B2."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.x_point import boundary_end, boundary_saddles, ENDS
from fec_adapter.scoring_kit.common import AXIS_SIGN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402

OUT = HERE / 'results' / 'b2b_tail_topology_v25_validation.json'


def ends(reader, flux):
    reading = reader.read(flux)
    saddles = boundary_saddles(reader, flux, reading)
    phi = AXIS_SIGN['DIII-D'] * flux
    out = {}
    for end in ENDS:
        tip = boundary_end(reading.contour, end)
        s = saddles[end]
        is_x = s is not None and float(np.hypot(*(s.position - tip))) < 0.05
        out[end] = {'x': bool(is_x), 'tip': tip,
                    'phi': float(b2.sample(phi, reader, s.position[None])[0]) if s is not None else np.nan}
    level = float(np.mean(b2.sample(phi, reader, np.asarray(reading.contour))))
    axis = float(b2.sample(phi, reader, np.asarray([reading.axis]))[0])
    # normalised flux gap between the two saddles (dRsep-like), positive = lower dominant
    gap = (out['lower']['phi'] - out['upper']['phi']) / (axis - level)
    return out, gap


def main():
    sys.path.insert(0, str(b2.REPO / 'fec_harness'))
    import run_parallel_score as scoring
    stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
    z = np.load(b2.OUT / 'frames.npz')
    data, cols, keys = z['data'], list(z['columns']), z['keys']
    d = data[:, cols.index('D')]
    tail = np.nonzero(d > 0.02)[0]
    rng = np.random.default_rng(29)
    rest = rng.choice(np.nonzero(d <= 0.02)[0], size=len(tail), replace=False)
    reader = Reader('DIII-D')
    rows = []
    with np.load(b2.V25 / 'diii_d_train.npz') as archive:
        for group, picks in (('tail', tail), ('rest', rest)):
            for shot in np.unique(keys[picks, 0]):
                stem = stems[shot]
                maps = archive[f'shot_{shot:04d}_psirz']
                truth = np.asarray(pq.read_table(b2.DATA / f'{stem}.parquet', columns=['efit_psirz'])
                                   ['efit_psirz'][0].as_py(), dtype=np.float64)
                for i in picks[keys[picks, 0] == shot]:
                    frame = int(keys[i, 1])
                    t, gt = ends(reader, truth[frame])
                    p, gp = ends(reader, np.asarray(maps[frame], np.float64))
                    rows.append({'group': group, 'D': float(d[i]),
                                 'truth_x': [t[e]['x'] for e in ENDS], 'pred_x': [p[e]['x'] for e in ENDS],
                                 'gap_truth': gt, 'gap_pred': gp,
                                 'tip_err_m': [float(np.hypot(*(t[e]['tip'] - p[e]['tip']))) for e in ENDS]})
    summary = {}
    for group in ('tail', 'rest'):
        g = [r for r in rows if r['group'] == group]
        tx = np.array([r['truth_x'] for r in g]); px = np.array([r['pred_x'] for r in g])
        gt = np.array([r['gap_truth'] for r in g]); gp = np.array([r['gap_pred'] for r in g])
        both = np.isfinite(gt) & np.isfinite(gp)
        summary[group] = {
            'frames': len(g), 'mean_D': float(np.mean([r['D'] for r in g])),
            'end_type_mismatch_share': float(np.mean(np.any(tx != px, axis=1))),
            'truth_double_x_share': float(np.mean(tx.all(axis=1))),
            'truth_lower_only_share': float(np.mean(tx[:, 0] & ~tx[:, 1])),
            'truth_upper_only_share': float(np.mean(~tx[:, 0] & tx[:, 1])),
            'truth_no_x_share': float(np.mean(~tx.any(axis=1))),
            'dominant_null_flip_share_of_both_defined': float(np.mean(np.sign(gt[both]) != np.sign(gp[both]))) if both.any() else None,
            'median_abs_gap_truth': float(np.median(np.abs(gt[both]))) if both.any() else None,
            'median_tip_err_m_lower_upper': np.median([r['tip_err_m'] for r in g], axis=0).tolist(),
        }
    OUT.write_text(json.dumps({'selection_or_fit': False, 'summary': summary, 'rows': rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
