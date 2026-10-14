# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D58: anatomy of what is left of the flux map after the networks' answer
"""Measurement only (entrant 2026-10-14: filter everything out, keep the remainder, look at its structure).

For the 358 development shots: the true map and the fit-disjoint 69-network answer, both as coordinates in the
128-direction flux basis (the networks answer 64, the deployed model keeps 45). Reports, per coordinate and in sum:
how much of the true variance the answer explains, the least-squares slope of truth on answer, where the left-over
energy sits (kept coordinates, answered-but-dropped coordinates, never-answered coordinates), how persistent the
left-over is along a shot, how much of it is one offset per shot, and whether the dropped coordinates would help
with a shrink weight fitted on other shots (five groups of whole shots). Nothing is selected here.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

import d39_common as c

KEPT, ANSWERED, FULL, GROUPS = 45, 64, 128, 5


def one(case):
    import predict_diii_d as pd
    basis = pd.FluxBasis.load(c.PKG / 'models/flux_basis.npz')
    cache = c.load_cache(case['stem'])
    table = pq.read_table(case['path'], columns=['efit_psirz'], use_threads=False)
    truth = np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float64)
    code = basis.truncated(FULL).encode(truth)
    centred = truth.reshape(len(truth), -1) - basis.mean.reshape(1, -1)
    outside = np.sum(centred ** 2, axis=1) - np.sum(code ** 2, axis=1)     # energy outside the 128 directions
    return case['index'], code.astype(np.float32), cache.coordinates.astype(np.float32), outside.astype(np.float32)


def main():
    cases = c.cases()
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(one, cases, chunksize=4))
    shot = np.concatenate([np.full(len(r[1]), r[0]) for r in rows])
    order = {index: k for k, index in enumerate(sorted({r[0] for r in rows}))}
    shot = np.asarray([order[s] for s in shot])
    truth = np.concatenate([r[1] for r in rows]).astype(float)
    answer = np.concatenate([r[2] for r in rows]).astype(float)
    outside = np.concatenate([r[3] for r in rows]).astype(float)
    assert answer.shape[1] == ANSWERED and truth.shape[1] == FULL
    shots = int(shot.max()) + 1
    n = len(truth)
    residual = truth.copy()
    residual[:, :ANSWERED] -= answer
    deployed = truth.copy()
    deployed[:, :KEPT] -= answer[:, :KEPT]
    energy_truth = np.sum(truth ** 2, axis=0)                  # the basis is centred on the training mean
    per = {'explained_share': (1 - np.sum(residual[:, :ANSWERED] ** 2, axis=0) / energy_truth[:ANSWERED]).tolist(),
           'slope_truth_on_answer': (np.sum(truth[:, :ANSWERED] * answer, axis=0) / np.sum(answer ** 2, axis=0)).tolist(),
           'correlation': [float(np.corrcoef(truth[:, k], answer[:, k])[0, 1]) for k in range(ANSWERED)],
           'truth_energy_share': (energy_truth / energy_truth.sum()).tolist()}
    left = np.sum(deployed ** 2, axis=0)
    total_left = left.sum() + outside.sum()
    report = {'shots': shots, 'frames': n,
              'left_over_energy_shares_deployed': {'error_in_kept_1_45': float(left[:KEPT].sum() / total_left),
                                                   'dropped_46_64': float(left[KEPT:ANSWERED].sum() / total_left),
                                                   'never_answered_65_128': float(left[ANSWERED:].sum() / total_left),
                                                   'outside_128': float(outside.sum() / total_left)},
              'flux_r2_proxy_deployed': float(1 - total_left / (energy_truth.sum() + outside.sum())),
              'per_coordinate': per}
    # persistence and per-shot offset of the left-over, per block
    def structure(block):
        x = deployed[:, block]
        count = np.bincount(shot, minlength=shots).astype(float)
        mean = np.stack([np.bincount(shot, weights=x[:, j], minlength=shots) for j in range(x.shape[1])], axis=1) / count[:, None]
        offset_share = float(np.sum(count[:, None] * mean ** 2) / np.sum(x ** 2))
        same = shot[1:] == shot[:-1]
        centred = x - mean[shot]
        lag = float(np.sum(centred[1:][same] * centred[:-1][same]) / np.sum(centred ** 2))
        return {'per_shot_offset_share': offset_share, 'lag1_autocorrelation_after_offset': lag}
    report['structure'] = {'kept_1_45': structure(slice(0, KEPT)), 'dropped_46_64': structure(slice(KEPT, ANSWERED)),
                           'never_answered_65_128': structure(slice(ANSWERED, FULL))}
    # principal directions of the left-over inside the 128 directions
    _, s, _ = np.linalg.svd(deployed - deployed.mean(axis=0), full_matrices=False)
    share = np.cumsum(s ** 2) / np.sum(s ** 2)
    report['left_over_directions_for_share'] = {str(q): int(np.searchsorted(share, q) + 1) for q in (.5, .8, .9, .95, .99)}
    # would the dropped coordinates help with a shrink weight from other shots?
    group = shot % GROUPS
    variants = {}
    base = np.sum(deployed ** 2) + outside.sum()
    for name, use in (('hard_64', 'one'), ('shrunk_46_64', 'slope'), ('shrunk_1_64', 'slope_all')):
        left_over = 0.0
        for g in range(GROUPS):
            fit, hold = group != g, group == g
            w = np.ones(ANSWERED)
            slope = np.sum(truth[fit][:, :ANSWERED] * answer[fit], axis=0) / np.sum(answer[fit] ** 2, axis=0)
            if use == 'one':
                pass
            elif use == 'slope':
                w[KEPT:] = np.clip(slope[KEPT:], 0, 1)
            else:
                w = np.clip(slope, 0, 1.5)
            r = truth[hold].copy()
            r[:, :ANSWERED] -= w * answer[hold]
            left_over += np.sum(r ** 2) + outside[hold].sum()
        variants[name] = {'left_over_energy_over_deployed': float(left_over / base),
                          'flux_r2_proxy': float(1 - left_over / (energy_truth.sum() + outside.sum()))}
    report['nested_variants'] = variants
    report['weights_all_shots_46_64'] = np.clip(np.sum(truth[:, KEPT:ANSWERED] * answer[:, KEPT:], axis=0) / np.sum(answer[:, KEPT:] ** 2, axis=0), 0, 1).tolist()
    np.savez_compressed(c.X / 'reports/d58_coordinates_358.npz', truth=truth.astype(np.float32), answer=answer.astype(np.float32),
                        outside=outside.astype(np.float32), shot=shot)
    (c.X / 'reports/d58_residual_anatomy_358.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_coordinate'}, indent=1))
    r = report['per_coordinate']
    for k in (0, 1, 2, 4, 9, 19, 29, 39, 44, 45, 47, 49, 54, 59, 63):
        print('coordinate', k + 1, 'explained', round(r['explained_share'][k], 4), 'slope', round(r['slope_truth_on_answer'][k], 3),
              'corr', round(r['correlation'][k], 3), 'energy share', '%.2e' % r['truth_energy_share'][k])


if __name__ == '__main__':
    sys.exit(main())
