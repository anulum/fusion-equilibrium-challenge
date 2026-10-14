# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D52: one third per network family in q95 and betaN, against the deployed weights
"""Apply the frozen rule of protocol_d52.json: no number is fitted.

deployed  .75 x (21 hx + 48 hc) / 69 + .25 x joint
candidate (hx + hc + joint) / 3

    python equal_family.py FAMILY_MEANS.npz RESULT.json
"""
import json
import sys

import numpy as np

RESAMPLES, SEED = 10000, 5201
MEANS = {'q95': 5.262893061875164, 'betaN': 1.2688498003107427}


def main():
    z = np.load(sys.argv[1])
    shot = z['shot']
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(p, minlength=shots) for p in rng.integers(0, shots, size=(RESAMPLES, shots))]).astype(float)
    per_shot = lambda v: np.bincount(shot, weights=v, minlength=shots)
    screening = np.arange(shots) % 3 == 0
    result, delta, boot, parts = {}, 0.0, 0.0, {'screening_120': 0.0, 'other_238': 0.0}
    for key in ('q95', 'betaN'):
        t, hx, hc, joint = (z[f'{key}_{n}'].astype(float) for n in ('truth', 'hx', 'hc', 'joint'))
        deployed = .75 * (21 * hx + 48 * hc) / 69 + .25 * joint
        candidate = (hx + hc + joint) / 3
        total = per_shot((t - MEANS[key]) ** 2)
        e_base, e_new = per_shot((deployed - t) ** 2), per_shot((candidate - t) ** 2)
        r2 = lambda e, pick=slice(None): float(1 - e[pick].sum() / total[pick].sum())
        b = (counts @ e_base - counts @ e_new) / (counts @ total)
        result[key] = {'r2_deployed': r2(e_base), 'r2_candidate': r2(e_new), 'dR2': r2(e_new) - r2(e_base),
                       'dR2_ci99': [float(np.percentile(b, 0.5)), float(np.percentile(b, 99.5))],
                       'P_positive': float(np.mean(b > 0)),
                       'dR2_screening_120': r2(e_new, screening) - r2(e_base, screening),
                       'dR2_other_238': r2(e_new, ~screening) - r2(e_base, ~screening),
                       'r2_alone': {n: r2(per_shot((v - t) ** 2)) for n, v in (('hx', hx), ('hc', hc), ('joint', joint))},
                       'error_correlation': {'hx_hc': float(np.corrcoef(hx - t, hc - t)[0, 1]),
                                             'hx_joint': float(np.corrcoef(hx - t, joint - t)[0, 1]),
                                             'hc_joint': float(np.corrcoef(hc - t, joint - t)[0, 1])}}
        delta += 0.15 * result[key]['dR2'] / 2
        boot = boot + 0.15 * b / 2
        parts['screening_120'] += 0.15 * result[key]['dR2_screening_120'] / 2
        parts['other_238'] += 0.15 * result[key]['dR2_other_238'] / 2
    low, high = np.percentile(boot, [0.5, 99.5])
    result.update({'protocol': 'protocol_d52.json', 'dS': float(delta), 'dS_ci99': [float(low), float(high)],
                   'P_dS_positive': float(np.mean(boot > 0)), 'dS_by_part': parts,
                   'passes': bool(delta >= 1e-5 and low > 0 and min(parts.values()) > 0),
                   'resamples': RESAMPLES, 'seed': SEED})
    open(sys.argv[2], 'w').write(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
