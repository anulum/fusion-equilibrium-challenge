# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D72 frozen gate: Grad-Shafranov interior against the submitted pipeline, 358 shots
"""python gate_d72.py RUN_DIR OUT.json; scores() is copied verbatim from d45_report.py (as in D66 and D70).

Pass iff delta S >= 1e-5, the 99 % interval of 20 000 whole-shot multinomial draws (seed 72020) lies above zero,
and neither the boundary failures nor the derived-scalar failures increase."""
import json
import sys
from pathlib import Path

import numpy as np

from d39_common import cases, statistics


def scores(counts, references, parts):
    totals = {k: counts @ np.asarray([r[k] for r in references], dtype=float) for k in references[0]}

    def denominator(prefix):
        return totals[prefix + '_sumsq'] - totals[prefix + '_sum'] ** 2 / np.maximum(totals[prefix + '_n'], 1.)

    flux_total, scalar_total, cons_total = denominator('psi'), denominator('scal'), denominator('cons')
    flux_res = counts @ np.asarray([p['ss_res_psi']['1.0'] for p in parts])
    scalar_res = counts @ np.asarray([p['ss_res_scal'] for p in parts])
    cons_res = counts @ np.asarray([p['ss_res_cons'] for p in parts])
    flux = np.where(flux_total > 0., 1. - flux_res / np.maximum(flux_total, 1e-300), 0.)
    scalar_each = np.where(scalar_total > 0., 1. - scalar_res / np.maximum(scalar_total, 1e-300), 0.)
    cons_each = np.where(cons_total > 0., 1. - cons_res / np.maximum(cons_total, 1e-300), np.nan)
    cons = np.nanmean(np.maximum(0., cons_each), axis=1)
    distance = np.asarray([np.nan if p['shot_dlcfs'] is None else p['shot_dlcfs'] for p in parts])
    valid = np.isfinite(distance)
    count = counts @ valid.astype(float)
    boundary = np.where(count > 0., (counts @ np.nan_to_num(distance)) / np.maximum(count, 1.), 1.)
    total = .55 * np.maximum(0., flux) + .15 * np.maximum(0., scalar_each.mean(axis=1)) + .1 * (1. - np.minimum(1., boundary)) + .2 * cons
    return {'S': total, 'psi': flux, 'qb': scalar_each.mean(axis=1), 'D': boundary, 'C': cons, 'cons_each': cons_each}


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    listed = cases(); rows, _t, _m = statistics()
    names = ('base', 'gs')
    recs = {c['stem']: json.loads((run / c['stem'] / 'complete.json').read_text())['parts'] for c in listed}
    parts = {n: [recs[c['stem']][n] for c in listed] for n in names}
    once = np.ones((1, len(listed)))
    exact = {n: scores(once, rows, parts[n]) for n in names}
    rng = np.random.default_rng(72020)
    weights = rng.multinomial(len(listed), np.full(len(listed), 1. / len(listed)), size=20000).astype(float)
    boot = {n: scores(weights, rows, parts[n])['S'] for n in names}
    report = {'shots': len(listed), 'bootstrap': '20000 whole-shot multinomial draws, seed 72020'}
    for n in names:
        e = exact[n]
        report[n] = {k: float(e[k][0]) for k in ('S', 'psi', 'qb', 'D', 'C')}
        report[n]['cons_each'] = e['cons_each'][0].tolist()
        report[n]['lcfs_fails'] = int(sum(p['lcfs_fails'] for p in parts[n]))
        report[n]['cons_fails'] = int(sum(sum(p['cons_fails']) for p in parts[n]))
    d = boot['gs'] - boot['base']
    delta = report['gs']['S'] - report['base']['S']
    ci99 = np.quantile(d, [.005, .995]).tolist()
    passed = (delta >= 1e-5 and ci99[0] > 0 and report['gs']['lcfs_fails'] <= report['base']['lcfs_fails']
              and report['gs']['cons_fails'] <= report['base']['cons_fails'])
    report['gs_minus_base'] = {'delta_S': delta, 'CI99': ci99, 'CI95': np.quantile(d, [.025, .975]).tolist(),
                               'P_positive': float((d > 0).mean()), 'PASS': bool(passed)}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
