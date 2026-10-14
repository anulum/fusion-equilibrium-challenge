# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete independent cohort confirmation
"""Confirm one locked candidate on every reserved shot using newly eligible trees."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from campaign_common import HERE, DATA, digest, write_json
from evaluate_development import anatomy, part_numbers, CASES as DEVELOPMENT
from analyze_development import scores
import confirmation_prediction as prediction

COHORT = sorted(json.loads((HERE / 'cohort.json').read_text())['reserved_confirmation'])
assert len(COHORT) == 357
STATE = {}


def truth_of(path: Path) -> dict:
    """Read only the official target columns after the candidate is locked."""
    table = pq.read_table(path, columns=['efit_psirz', 'efit_q95', 'efit_beta_n'], use_threads=False)
    return {'psi': np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float32),
            'q95': np.asarray(table['efit_q95'][0].as_py(), dtype=float),
            'betaN': np.asarray(table['efit_beta_n'][0].as_py(), dtype=float)}


def reference_of(stem: str) -> dict:
    """Build unchanged native references and pooled ground-truth statistics."""
    path = DATA / (stem + '.parquet')
    saved = HERE / 'confirmation/references' / (stem + '.pkl')
    receipt = saved.with_suffix('.json')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert digest(saved) == row['reference_sha256']
        return row
    truth = truth_of(path)
    radius, height, coarse, mask = anatomy.scorer._grid()
    reference = anatomy.scorer.local_score.build_reference(truth['psi'], radius, height, coarse, mask)
    with saved.open('xb') as stream:
        pickle.dump(reference, stream)
    flux = truth['psi'].astype(float)
    valid = np.isfinite(flux)
    totals = {'psi_sum': float(flux[valid].sum()), 'psi_sumsq': float((flux[valid] ** 2).sum()), 'psi_n': int(valid.sum()),
              'scal_sum': np.zeros(2), 'scal_sumsq': np.zeros(2), 'scal_n': np.zeros(2),
              'cons_sum': np.zeros(anatomy.scorer.N_CONS), 'cons_sumsq': np.zeros(anatomy.scorer.N_CONS), 'cons_n': np.zeros(anatomy.scorer.N_CONS)}
    for index, name in enumerate(anatomy.scorer.SCALARS):
        values = truth[name][np.isfinite(truth[name])]
        totals['scal_sum'][index], totals['scal_sumsq'][index], totals['scal_n'][index] = values.sum(), (values ** 2).sum(), len(values)
    derived, defined = reference[1], reference[2]
    for index in range(anatomy.scorer.N_CONS):
        values = derived[defined[:, index], index]
        totals['cons_sum'][index], totals['cons_sumsq'][index], totals['cons_n'][index] = values.sum(), (values ** 2).sum(), len(values)
    row = {'stem': stem, 'frames': len(truth['psi']), 'statistics': anatomy.serial(totals), 'reference_sha256': digest(saved)}
    write_json(receipt, row)
    return row


def one_case(stem: str) -> dict:
    """Score all frames of exactly the candidate and matched comparator."""
    out = HERE / 'confirmation/cases'
    receipt = out / (stem + '.json')
    arrays = receipt.with_suffix('.npz')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert digest(arrays) == row['arrays_sha256'] and row['winner'] == STATE['winner']
        return row
    started = time.monotonic()
    path = DATA / (stem + '.parquet')
    truth = truth_of(path)
    reference = pickle.loads((HERE / 'confirmation/references' / (stem + '.pkl')).read_bytes())
    predicted, shot = prediction.predict(path)
    assert len(truth['psi']) == shot.n_frames
    saved, metrics = {}, {}
    for arm in ('matched69_scalar8', STATE['winner']):
        metrics[arm], distances, derived = anatomy.score_maps(truth, reference, predicted[arm])
        saved[arm + '_distance'], saved[arm + '_derived'] = distances, derived
        saved[arm + '_flux_mse'] = np.mean((predicted[arm]['psirz'].astype(float) - truth['psi']) ** 2, axis=(1, 2))
        saved[arm + '_q95'], saved[arm + '_betaN'] = predicted[arm]['q95'], predicted[arm]['betaN']
    saved.update(times_ms=shot.times_ms, ip_A=shot.plasma_current)
    temporary = arrays.with_name(stem + '.partial.npz')
    np.savez_compressed(temporary, **saved); temporary.replace(arrays)
    row = {'stem': stem, 'winner': STATE['winner'], 'frames': shot.n_frames, 'parts': anatomy.serial(metrics),
           'arrays_sha256': digest(arrays), 'seconds': time.monotonic() - started}
    write_json(receipt, row)
    print(json.dumps({'confirmation_shot_complete': stem, 'frames': shot.n_frames, 'seconds': row['seconds']}), flush=True)
    return row


def hashes_of(path: str) -> dict:
    """Exercise the exact locked confirmation path in a real inherited fork."""
    values, _shot = prediction.predict(Path(path))
    return {arm: {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in fields.items()} for arm, fields in values.items()}


def qualify(winner: str) -> None:
    """Require exact equality with the already qualified development arithmetic."""
    prediction.initialize(winner, original_trees=True)
    path = Path(DEVELOPMENT[0]['path'])
    original, _shot = prediction.cp.predict(path)
    locked, _shot = prediction.predict(path)
    for arm in ('matched69_scalar8', winner):
        assert all(np.array_equal(original[arm][name], values) for name, values in locked[arm].items()), arm
    hashes = {arm: {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in fields.items()} for arm, fields in locked.items()}
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        inherited = pool.submit(hashes_of, str(path)).result(timeout=1800)
    assert hashes == inherited
    write_json(HERE / 'confirmation/qualification.json', {'status': 'locked_two_arm_native_development_and_real_fork_bitexact',
               'winner': winner, 'no_confirmation_targets_read': True, 'hashes': hashes})


def main() -> None:
    """Open the reserved target cohort only after every prospective development gate."""
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true'); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    analysis = json.loads((HERE / 'evaluation/development_analysis.json').read_text())
    winner = analysis['frozen_confirmation_candidate']
    assert winner is not None and analysis['decisions'][winner]['development_gate_pass']
    STATE['winner'] = winner
    root = HERE / 'confirmation'; root.mkdir(exist_ok=True)
    if args.qualify:
        qualify(winner); return
    qualification = json.loads((root / 'qualification.json').read_text())
    assert qualification['status'] == 'locked_two_arm_native_development_and_real_fork_bitexact' and qualification['winner'] == winner
    assert json.loads((HERE / 'confirmation_trees/receipt.json').read_text())['status'] == 'all_four_eligible_shared_confirmation_trees_complete'
    for name in ('references', 'cases'):
        (root / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        references = list(pool.map(reference_of, COHORT, chunksize=1))
    assert len(references) == 357 and sum(r['frames'] for r in references) == 76251
    statistics = [r['statistics'] for r in references]
    totals = {k: np.sum([r[k] for r in statistics], axis=0) for k in statistics[0]}
    anatomy.MEANS = (totals['psi_sum'] / totals['psi_n'], totals['scal_sum'] / np.maximum(totals['scal_n'], 1),
                     totals['cons_sum'] / np.maximum(totals['cons_n'], 1))
    prediction.initialize(winner)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        rows = list(pool.map(one_case, COHORT, chunksize=1))
    assert len(rows) == 357 and sum(r['frames'] for r in rows) == 76251
    arms = ('matched69_scalar8', winner)
    parts = {arm: [part_numbers(row['parts'][arm]) for row in rows] for arm in arms}
    official, exact = {}, {}
    for arm in arms:
        acc = anatomy.scorer.Accum(); acc.psi_sign = 1.
        for part in parts[arm]:
            acc.add(part)
        official[arm] = anatomy.scorer.finalize_machine(acc, totals)
        assert not official[arm]['psi_sign_probe_disagrees']
        exact[arm] = {k: float(v[0]) for k, v in scores(np.ones((1, 357)), statistics, parts[arm]).items()}
        assert all(round(value, 6) == official[arm][key] for key, value in exact[arm].items())
    config = json.loads((HERE / 'confirmation_protocol.json').read_text())
    rng = np.random.default_rng(config['bootstrap_seed'])
    weights = rng.multinomial(357, np.full(357, 1. / 357), size=10000).astype(float)
    differences = scores(weights, statistics, parts[winner])['S'] - scores(weights, statistics, parts[arms[0]])['S']
    delta = exact[winner]['S'] - exact[arms[0]]['S']
    interval = np.quantile(differences, [.005, .995])
    failures = {arm: sum(part['lcfs_fails'] for part in parts[arm]) for arm in arms}
    nonfinite = {arm: sum(part['psi_nonfinite'] for part in parts[arm]) for arm in arms}
    passed = delta >= 3e-5 and interval[0] > 0 and failures[winner] <= failures[arms[0]] and nonfinite[winner] == 0
    write_json(root / 'analysis.json', {'status': 'all357_independent_confirmation_complete', 'shots': 357, 'frames': 76251,
               'winner_locked_before_confirmation': winner, 'unrounded_scores': exact, 'official_scores': official,
               'delta_S': delta, 'paired_shot_CI99': interval.tolist(), 'LCFS_failures': failures,
               'nonfinite_pixels': nonfinite, 'same_prospective_gate_confirmation_pass': bool(passed),
               'all_four_shared_trees_refit_excluding_both_cohorts': True, 'publication': False,
               'next': 'eligible_for_candidate_build_review' if passed else 'rejected_no_submission'})


if __name__ == '__main__':
    main()
