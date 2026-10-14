# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — single joint candidate reserved confirmation
"""Open all357 reserved targets only after one joint candidate is locked."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

import joint_confirmation_prediction as prediction
from joint_prediction import HERE, D38, BASE
import evaluate_confirmation as original

original.HERE = HERE  # native references are retained under this separate study
anatomy = original.anatomy
STATE = {}


def write(path, data):
    with path.open('x') as stream:
        json.dump(anatomy.serial(data), stream, indent=2, allow_nan=False)


def hashes(path):
    values, _shot = prediction.predict(Path(path))
    return {arm: {name: hashlib.sha256(v.tobytes()).hexdigest() for name, v in fields.items()}
            for arm, fields in values.items()}


def one(stem):
    target = HERE / 'confirmation/cases' / (stem + '.json')
    arrays_path = target.with_suffix('.npz')
    if target.exists():
        row = json.loads(target.read_text())
        assert row['lock_sha256'] == STATE['lock_sha256'] and row['winner'] == STATE['winner']
        assert prediction.jp.cp.digest(arrays_path) == row['arrays_sha256']
        return row
    started = time.monotonic()
    path = original.DATA / (stem + '.parquet')
    truth = original.truth_of(path)
    reference_path = HERE / 'confirmation/references' / (stem + '.pkl')
    reference = pickle.loads(reference_path.read_bytes())
    values, shot = prediction.predict(path)
    assert set(values) == {BASE, STATE['winner']}
    assert shot.n_frames == len(truth['psi'])
    parts, arrays = {}, {}
    for arm, fields in values.items():
        parts[arm], distance, derived = anatomy.score_maps(truth, reference, fields)
        arrays[arm + '_distance'], arrays[arm + '_derived'] = distance, derived
        arrays[arm + '_flux_mse'] = np.mean((fields['psirz'].astype(float) - truth['psi']) ** 2, axis=(1, 2))
        arrays[arm + '_q95'], arrays[arm + '_betaN'] = fields['q95'], fields['betaN']
    tmp = arrays_path.with_name(stem + '.partial.npz')
    np.savez_compressed(tmp, **arrays); tmp.replace(arrays_path)
    row = {'stem': stem, 'frames': shot.n_frames, 'winner': STATE['winner'], 'parts': parts,
           'lock_sha256': STATE['lock_sha256'], 'arrays_sha256': prediction.jp.cp.digest(arrays_path),
           'reference_sha256': prediction.jp.cp.digest(reference_path), 'seconds': time.monotonic() - started}
    write(target, row)
    print(json.dumps({'joint_confirmation_complete_shot': stem, 'frames': shot.n_frames}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    lock_path = HERE / 'concrete_confirmation_candidate_lock.json'
    lock = json.loads(lock_path.read_text())
    central = json.loads((HERE / 'central_development_analysis.json').read_text())
    assert lock['one_candidate_only'] and lock['locked_before_reserved_candidate_scoring']
    assert lock['winner'] == central['one_joint_winner']
    assert lock['central_analysis_sha256'] == prediction.jp.cp.digest(HERE / 'central_development_analysis.json')
    STATE.update(winner=lock['winner'], lock_sha256=prediction.jp.cp.digest(lock_path))
    root = HERE / 'confirmation'; root.mkdir(exist_ok=True)
    if args.qualify:
        prediction.initialize(original_trees=True)
        path = original.DEVELOPMENT[0]['path']
        actual = hashes(path)
        expected = json.loads((HERE / 'concrete_finalist_qualification.json').read_text())['prediction_hashes']
        assert actual == {arm: expected[arm] for arm in (BASE, STATE['winner'])}
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
            inherited = pool.submit(hashes, path).result(timeout=1800)
        assert inherited == actual
        write(root / 'qualification.json', {'status': 'one_locked_joint_candidate_native_and_real_fork_bitexact',
               'lock_sha256': STATE['lock_sha256'], 'prediction_hashes': actual, 'no_reserved_targets_opened': True})
        return
    qualification = json.loads((root / 'qualification.json').read_text())
    assert qualification['lock_sha256'] == STATE['lock_sha256']
    assert json.loads((D38 / 'confirmation_trees/receipt.json').read_text())['status'] == 'all_four_eligible_shared_confirmation_trees_complete'
    for name in ('references', 'cases'):
        (root / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        references = list(pool.map(original.reference_of, original.COHORT, chunksize=1))
    assert len(references) == 357 and sum(r['frames'] for r in references) == 76251
    statistics = [r['statistics'] for r in references]
    totals = {k: np.sum([r[k] for r in statistics], axis=0) for k in statistics[0]}
    anatomy.MEANS = (totals['psi_sum'] / totals['psi_n'], totals['scal_sum'] / np.maximum(totals['scal_n'], 1),
                     totals['cons_sum'] / np.maximum(totals['cons_n'], 1))
    prediction.initialize()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        rows = list(pool.map(one, original.COHORT, chunksize=1))
    assert len(rows) == 357 and sum(r['frames'] for r in rows) == 76251
    arms = (BASE, STATE['winner'])
    parts = {a: [original.part_numbers(r['parts'][a]) for r in rows] for a in arms}
    exact, official = {}, {}
    for arm in arms:
        acc = anatomy.scorer.Accum(); acc.psi_sign = 1.
        for p in parts[arm]:
            acc.add(p)
        official[arm] = anatomy.scorer.finalize_machine(acc, totals)
        assert not official[arm]['psi_sign_probe_disagrees']
        exact[arm] = {k: float(v[0]) for k, v in original.scores(np.ones((1, 357)), statistics, parts[arm]).items()}
        assert all(round(v, 6) == official[arm][k] for k, v in exact[arm].items())
    rng = np.random.default_rng(3802)
    weights = rng.multinomial(357, np.full(357, 1. / 357), size=10000).astype(float)
    delta = exact[arms[1]]['S'] - exact[BASE]['S']
    differences = original.scores(weights, statistics, parts[arms[1]])['S'] - original.scores(weights, statistics, parts[BASE])['S']
    interval = np.quantile(differences, [.005, .995])
    failures = {a: sum(p['lcfs_fails'] for p in parts[a]) for a in arms}
    nonfinite = {a: sum(p['psi_nonfinite'] for p in parts[a]) for a in arms}
    passed = delta >= 3e-5 and interval[0] > 0 and failures[arms[1]] <= failures[BASE] and nonfinite[arms[1]] == 0
    write(root / 'analysis.json', {'status': 'all357_one_joint_candidate_confirmation_complete',
          'shots': 357, 'frames': 76251, 'lock_sha256': STATE['lock_sha256'],
          'unrounded_scores': exact, 'official_scores': official, 'delta_S': delta,
          'CI99': interval.tolist(), 'LCFS_failures': failures, 'nonfinite_pixels': nonfinite,
          'confirmation_gate_pass': bool(passed), 'bootstrap_seed': 3802, 'replicates': 10000,
          'one_candidate_only': True, 'shared_eligible_trees': True, 'publication': False})


if __name__ == '__main__':
    main()
