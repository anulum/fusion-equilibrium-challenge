# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D51: share of the joint scalar networks in q95 and betaN, by nested least squares
"""Apply the frozen rule of protocol_d51.json to the D49 frame file (358 development shots).

answer = map + w (joint - map); w per scalar by least squares, clipped to [0, 1]. Nested: the cohort is cut into
K groups of whole shots (listing index mod K); every group's frames are answered with the w fitted on the other
groups. The deployed answer has w = .25 for both scalars. The score change is 0.15 x mean of the two dR2, with the
scorer's own label means of the cohort; paired bootstrap over whole shots.

    python nested_share.py FRAMES.npz RESULT.json
"""
import json
import sys

import numpy as np

K, RESAMPLES, SEED, DEPLOYED = 5, 10000, 5101, 0.25
MEANS = {'q95': 5.262893061875164, 'betaN': 1.2688498003107427}


def weight(truth, base, joint):
    d = joint - base
    return float(np.clip(np.dot(d, truth - base) / np.dot(d, d), 0.0, 1.0))


def main():
    z = np.load(sys.argv[1])
    shot = z['shot']
    shots = int(shot.max()) + 1
    group = shot % K
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(p, minlength=shots) for p in rng.integers(0, shots, size=(RESAMPLES, shots))]).astype(float)
    per_shot = lambda v: np.bincount(shot, weights=v, minlength=shots)
    result, delta, boot, parts = {}, 0.0, 0.0, {'screening_120': 0.0, 'other_238': 0.0}
    screening = np.arange(shots) % 3 == 0
    for key in ('q95', 'betaN'):
        t, m, j = z[key + '_truth'].astype(float), z[key + '_map'].astype(float), z[key + '_joint'].astype(float)
        assert np.isfinite(t).all() and np.isfinite(m).all() and np.isfinite(j).all()
        nested, weights = np.empty_like(t), []
        for k in range(K):
            fit, use = group != k, group == k
            weights.append(weight(t[fit], m[fit], j[fit]))
            nested[use] = m[use] + weights[-1] * (j[use] - m[use])
        total = per_shot((t - MEANS[key]) ** 2)
        e_base = per_shot((m + DEPLOYED * (j - m) - t) ** 2)
        e_new = per_shot((nested - t) ** 2)
        r2 = lambda e, pick=slice(None): 1 - e[pick].sum() / total[pick].sum()
        d = r2(e_new) - r2(e_base)
        b = (1 - (counts @ e_new) / (counts @ total)) - (1 - (counts @ e_base) / (counts @ total))
        result[key] = {'weights_by_group': weights, 'weight_all_358': weight(t, m, j), 'r2_deployed': r2(e_base),
                       'r2_nested': r2(e_new), 'dR2': d,
                       'dR2_ci99': [float(np.percentile(b, 0.5)), float(np.percentile(b, 99.5))],
                       'P_positive': float(np.mean(b > 0)),
                       'dR2_screening_120': r2(e_new, screening) - r2(e_base, screening),
                       'dR2_other_238': r2(e_new, ~screening) - r2(e_base, ~screening)}
        delta += 0.15 * d / 2
        boot = boot + 0.15 * b / 2
        parts['screening_120'] += 0.15 * result[key]['dR2_screening_120'] / 2
        parts['other_238'] += 0.15 * result[key]['dR2_other_238'] / 2
    low, high = np.percentile(boot, [0.5, 99.5])
    result.update({'protocol': 'protocol_d51.json', 'dS': float(delta), 'dS_ci99': [float(low), float(high)],
                   'P_dS_positive': float(np.mean(boot > 0)), 'dS_by_part': parts,
                   'passes': bool(delta >= 1e-5 and low > 0 and min(parts.values()) > 0),
                   'groups': K, 'resamples': RESAMPLES, 'seed': SEED})
    open(sys.argv[2], 'w').write(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
