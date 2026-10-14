# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D39 finisher study: guarded aggregate, paired bootstrap and frozen selector (v3), D41 output directory
"""Aggregate every variant over the shots all of them have finished, and bootstrap whole shots against the base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from d39_common import X, cases, statistics


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*', default=None)
    parser.add_argument('--base', default='base')
    parser.add_argument('--seed', type=int, default=3901)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--offsets', type=int, nargs='+', required=True)
    parser.add_argument('--select', type=Path, default=None, help='amendment 4: apply its frozen selector')
    args = parser.parse_args()
    declared = json.loads(args.variants.read_text())
    names = args.names or list(declared['variants'])
    if args.select:  # amendment 5: a selection reads the whole declared screening batch or nothing
        assert args.stride == 3 and args.offsets == [0], 'selection is made on the 120 screening shots only'
        assert names == list(declared['variants']) and len(names) == 32, 'selection needs all 32 declared variants'
        assert args.base == 'base' and args.seed == 3901 and args.replicates == 10000
    listed = cases()
    rows, _total, _means = statistics()
    assert len(rows) == len(listed)
    done = [i for i, c in enumerate(listed) if c['index'] % args.stride in args.offsets]
    expected = {(0,): 120, (1, 2): 238}.get(tuple(sorted(args.offsets))) if args.stride == 3 else None
    assert expected is not None and len(done) == expected, 'only the declared screening or confirmation part is read'
    missing = [(n, listed[i]['stem']) for n in names for i in done
               if not (X / 'd41_variants' / n / (listed[i]['stem'] + '.json')).exists()]
    assert not missing, f'{len(missing)} shot-variant records are missing; nothing is read before the batch is complete'
    references = [rows[i] for i in done]
    parts = {n: [json.loads((X / 'd41_variants' / n / (listed[i]['stem'] + '.json')).read_text())['part'] for i in done] for n in names}
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(len(done), np.full(len(done), 1. / len(done)), size=args.replicates).astype(float)
    once = np.ones((1, len(done)))
    exact = {n: scores(once, references, parts[n]) for n in names}
    base = scores(weights, references, parts[args.base])['S']
    report = {'shots': len(done), 'frames': int(sum(listed[i]['span'][1] - listed[i]['span'][0] for i in done)), 'variants': {}}
    print(f"shots {len(done)} of {len(listed)}")
    print(f"{'variant':10s} {'S':>9s} {'dS e-6':>8s} {'lo99':>7s} {'hi99':>7s} {'psi':>9s} {'qb':>8s} {'D':>8s} {'C':>8s} fails | consistency each")
    for n in names:
        e = exact[n]
        delta = float(e['S'][0] - exact[args.base]['S'][0])
        interval = np.quantile(scores(weights, references, parts[n])['S'] - base, [.005, .995]) if n != args.base else np.zeros(2)
        fails = sum(p['lcfs_fails'] for p in parts[n])
        report['variants'][n] = {'S': float(e['S'][0]), 'delta_S': delta, 'CI99': interval.tolist(), 'psi': float(e['psi'][0]),
                                 'qb': float(e['qb'][0]), 'D': float(e['D'][0]), 'C': float(e['C'][0]), 'lcfs_fails': int(fails),
                                 'cons_each': e['cons_each'][0].tolist()}
        print(f"{n:10s} {e['S'][0]:.6f} {delta*1e6:8.1f} {interval[0]*1e6:7.1f} {interval[1]*1e6:7.1f} {e['psi'][0]:.6f} {e['qb'][0]:.5f} "
              f"{e['D'][0]:.6f} {e['C'][0]:.5f} {fails:3d} | " + ' '.join(f'{v:.4f}' for v in e['cons_each'][0]))
    if args.select:
        frozen = json.loads(args.select.read_text())['3_selector']
        base_fails = report['variants'][args.base]['lcfs_fails']
        chosen, combination = {}, {}
        for group, members in frozen['groups'].items():
            passing = [(report['variants'][n]['delta_S'], -k, n) for k, n in enumerate(members)
                       if report['variants'][n]['delta_S'] >= 1e-5 and report['variants'][n]['CI99'][0] > 0
                       and report['variants'][n]['lcfs_fails'] <= base_fails]
            chosen[group] = max(passing)[2] if passing else None
            if passing:
                combination.update(declared['variants'][chosen[group]])
        assert not ({'oracle', 'oracle_x'} & set(combination)), 'a diagnostic key cannot enter a deployable combination'
        assert not (set(frozen['not_selectable']) & {n for n in chosen.values() if n})
        report['selection'] = {'winners': chosen, 'combination': combination}
        print('selection', json.dumps(report['selection']))
    if args.out:
        with args.out.open('x') as stream:
            json.dump(report, stream, indent=1)


if __name__ == '__main__':
    main()
