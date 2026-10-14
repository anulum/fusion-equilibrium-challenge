# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2g oracle: the true-X-point move on every score term
"""All B2 frames (5322, OOS validation, reserved excluded; diagnosis only): the v25 map with its
boundary saddles moved to the TRUE positions (with_x_points, share 1, 3 rounds), scored per frame
on D, the seven derived scalars (squared error vs the truth map's derivation) and the flux SSE,
against v25 unchanged. Frame-pooled sums, not the official shot-pooled R2."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.x_point import boundary_saddles, with_x_points
from fec_adapter.scoring_kit.common import CONS_SCALARS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402

OUT = HERE / 'results' / 'b2g_xpos_oracle_all_terms_v25_validation.json'


def terms(reader, rt, ct, rgeo, flux, truth):
    r = reader.read(flux)
    d = np.nan if r.contour is None else b2.hausdorff(np.asarray(r.contour, float), ct) / rgeo
    return [d, *[(r.values[n] - rt.values[n]) ** 2 for n in CONS_SCALARS], float(np.sum((flux - truth) ** 2))]


def main():
    sys.path.insert(0, str(b2.REPO / 'fec_harness'))
    import run_parallel_score as scoring
    stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
    lab = np.load(HERE / 'results/b2d_tail_classes_v25_validation/labels.npz')
    labels, keys = lab['labels'], lab['keys']
    reader = Reader('DIII-D')
    base, moved = [], []
    with np.load(b2.V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
        for shot in np.unique(keys[:, 0]):
            maps = archive[f'shot_{shot:04d}_psirz']
            tmaps = np.asarray(pq.read_table(b2.DATA / f'{stems[shot]}.parquet', columns=['efit_psirz'])
                               ['efit_psirz'][0].as_py(), dtype=np.float64)
            for f in keys[keys[:, 0] == shot, 1]:
                truth, pred = tmaps[f], np.asarray(maps[f], np.float64)
                rt = reader.read(truth)
                ct = np.asarray(rt.contour, float)
                rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
                want = {e: (s.position if s is not None else np.full(2, np.nan))
                        for e, s in boundary_saddles(reader, truth, rt).items()}
                base.append(terms(reader, rt, ct, rgeo, pred, truth))
                moved.append(terms(reader, rt, ct, rgeo, with_x_points(reader, pred, want)[0], truth))
    base, moved = np.asarray(base), np.asarray(moved)
    names = ['D', *[f'se_{n}' for n in CONS_SCALARS], 'flux_sse']
    corner = labels == 'corner'
    out = {'selection_or_fit': False, 'oracle': True, 'frames': len(base), 'reserved_excluded': True,
           'note': 'frame-pooled; D as mean, others as ratio moved/base of summed squared error'}
    for tag, m in (('all', np.ones(len(base), bool)), ('corner', corner), ('other', ~corner)):
        out[tag] = {'D_base': float(np.nanmean(base[m, 0])), 'D_moved': float(np.nanmean(moved[m, 0])),
                    'lost': int(np.isnan(moved[m, 0]).sum()),
                    **{f'{n}_ratio': float(np.nansum(moved[m, i]) / np.nansum(base[m, i]))
                       for i, n in enumerate(names) if i > 0}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
