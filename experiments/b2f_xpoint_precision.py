# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2f diagnostic: how precise the X-point must be
"""On the B2d corner frames and an equal control sample (diagnosis only): the v25 boundary
saddles' distance from the true ones, and the boundary distance when the saddles are moved a
share s of the way to the true positions (s = 0.25, 0.5, 0.75, 1), i.e. with the position error
scaled by 1 - s. Also the same for a perturbed target: truth plus isotropic Gaussian noise of
2, 4, 8 mm, which is what a head of that accuracy would give."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.x_point import boundary_saddles, with_x_points

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402
from b2e_corner_oracles import distance  # noqa: E402

OUT = HERE / 'results' / 'b2f_xpoint_precision_v25_validation.json'
SHARES = (0.25, 0.5, 0.75, 1.0)
NOISE = (0.002, 0.004, 0.008)


def main():
    sys.path.insert(0, str(b2.REPO / 'fec_harness'))
    import run_parallel_score as scoring
    stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
    lab = np.load(HERE / 'results/b2d_tail_classes_v25_validation/labels.npz')
    labels, keys = lab['labels'], lab['keys']
    rng = np.random.default_rng(37)
    corner = np.nonzero(labels == 'corner')[0]
    control = rng.choice(np.nonzero(labels != 'corner')[0], size=len(corner), replace=False)
    reader = Reader('DIII-D')
    rows = []
    with np.load(b2.V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
        for group, picks in (('corner', corner), ('control', control)):
            for shot in np.unique(keys[picks, 0]):
                maps = archive[f'shot_{shot:04d}_psirz']
                tmaps = np.asarray(pq.read_table(b2.DATA / f'{stems[shot]}.parquet', columns=['efit_psirz'])
                                   ['efit_psirz'][0].as_py(), dtype=np.float64)
                for i in picks[keys[picks, 0] == shot]:
                    f = int(keys[i, 1])
                    truth, pred = tmaps[f], np.asarray(maps[f], np.float64)
                    rt = reader.read(truth)
                    ct = np.asarray(rt.contour, float)
                    rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
                    st, sp = boundary_saddles(reader, truth, rt), boundary_saddles(reader, pred)
                    want = {e: (s.position if s is not None else np.full(2, np.nan)) for e, s in st.items()}
                    err = {e: (float(np.hypot(*(st[e].position - sp[e].position)))
                               if st[e] is not None and sp[e] is not None else None) for e in st}
                    row = {'group': group, 'err': err, 'base': distance(reader, pred, ct, rgeo)}
                    for s in SHARES:
                        row[f'share{s}'] = distance(reader, with_x_points(reader, pred, want, share=s)[0], ct, rgeo)
                    for sigma in NOISE:
                        noisy = {e: v + rng.normal(0, sigma, 2) for e, v in want.items()}
                        row[f'noise{sigma}'] = distance(reader, with_x_points(reader, pred, noisy)[0], ct, rgeo)
                    rows.append(row)
    summary = {'selection_or_fit': False, 'oracle': True, 'reserved_excluded': True}
    for group in ('corner', 'control'):
        g = [r for r in rows if r['group'] == group]
        errs = np.array([v for r in g for v in r['err'].values() if v is not None])
        summary[group] = {'frames': len(g), 'saddle_error_m_quantiles_25_50_75_90':
                          np.quantile(errs, [.25, .5, .75, .9]).tolist() if len(errs) else None,
                          **{k: float(np.nanmean([r[k] for r in g]))
                             for k in ['base', *[f'share{s}' for s in SHARES], *[f'noise{n}' for n in NOISE]]}}
    OUT.write_text(json.dumps({'summary': summary, 'rows': rows}))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
