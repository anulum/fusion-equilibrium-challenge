# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2e oracles: what decides the X-point corner of the boundary
"""Oracles on the B2d 'corner' frames (diagnosis only; truth is the oracle target).

1. ``xpos``  - the v25 map's boundary saddles moved to the TRUE saddle positions with the
               pipeline's own local move (x_point.with_x_points, full share, 3 rounds).
2. ``patchR`` - the v25 map replaced by the true map inside a disc of radius R around the
               true dominant saddle, blended over a cosine ramp of the same width; for
               R = 0.04, 0.08, 0.15 m. Shows how local the corner error is.
3. ``xpos_patch`` is not run: if patch0.04 already fixes the corner, position alone is
   the question; if not, the corner is decided by the wider neighbourhood.

Also applied to an equal-size random sample of non-corner frames as a control.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.x_point import boundary_saddles, with_x_points

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402
import b2c_saddle_gap_oracle as oracle  # noqa: E402

OUT = HERE / 'results' / 'b2e_corner_oracles_v25_validation'
RADII = (0.04, 0.08, 0.15)


def distance(reader, flux, ct, rgeo):
    r = reader.read(flux)
    return np.nan if r.contour is None else b2.hausdorff(np.asarray(r.contour, float), ct) / rgeo


def patched(reader, pred, truth, centre, radius):
    rho = np.hypot(reader.rr - centre[0], reader.zz - centre[1])
    weight = np.clip((2 * radius - rho) / radius, 0, 1)
    weight = 0.5 - 0.5 * np.cos(np.pi * weight)
    return pred + weight * (truth - pred)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        sys.path.insert(0, str(b2.REPO / 'fec_harness'))
        import run_parallel_score as scoring
        stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
        lab = np.load(HERE / 'results/b2d_tail_classes_v25_validation/labels.npz')
        labels, keys = lab['labels'], lab['keys']
        corner = np.nonzero(labels == 'corner')[0]
        rng = np.random.default_rng(31)
        control = rng.choice(np.nonzero(labels != 'corner')[0], size=len(corner), replace=False)
        reader = Reader('DIII-D')
        rows = []
        stage = 'frames'
        with np.load(b2.V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
            for group, picks in (('corner', corner), ('control', control)):
                for shot in np.unique(keys[picks, 0]):
                    maps = archive[f'shot_{shot:04d}_psirz']
                    truth_maps = np.asarray(pq.read_table(b2.DATA / f'{stems[shot]}.parquet',
                                                          columns=['efit_psirz'])['efit_psirz'][0].as_py(),
                                            dtype=np.float64)
                    for i in picks[keys[picks, 0] == shot]:
                        frame = int(keys[i, 1])
                        truth, pred = truth_maps[frame], np.asarray(maps[frame], np.float64)
                        rt = reader.read(truth)
                        ct = np.asarray(rt.contour, float)
                        rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
                        row = {'group': group, 'base': distance(reader, pred, ct, rgeo)}
                        wanted = {end: (s.position if s is not None else np.full(2, np.nan))
                                  for end, s in boundary_saddles(reader, truth, rt).items()}
                        moved, _ = with_x_points(reader, pred, wanted)
                        row['xpos'] = distance(reader, moved, ct, rgeo)
                        s = [v for v in oracle.saddles(reader, truth).values() if v is not None]
                        if s:
                            centre = max(s, key=lambda v: v[1])[0]
                            for radius in RADII:
                                row[f'patch{radius}'] = distance(
                                    reader, patched(reader, pred, truth, centre, radius), ct, rgeo)
                        rows.append(row)
        stage = 'summary'
        summary = {'status': 'diagnostic_complete', 'oracle': True, 'selection_or_fit': False,
                   'MAST_targets_used': False, 'reserved_excluded': True}
        for group in ('corner', 'control'):
            g = [r for r in rows if r['group'] == group]
            summary[group] = {'frames': len(g)}
            for k in ['base', 'xpos', *[f'patch{r}' for r in RADII]]:
                v = np.array([r.get(k, np.nan) for r in g], float)
                summary[group][f'{k}_mean_D'] = float(np.nanmean(v))
                summary[group][f'{k}_lost_or_missing'] = int(np.isnan(v).sum())
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        (OUT / 'rows.json').write_text(json.dumps(rows))
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
