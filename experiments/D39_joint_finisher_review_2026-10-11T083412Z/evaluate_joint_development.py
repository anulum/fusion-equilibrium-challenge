# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete central finalist development evaluation
"""Score fixed finalists on all358 before a single joint selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np
import pyarrow.parquet as pq

import joint_prediction as jp
from evaluate_development import anatomy, part_numbers, CASES
from analyze_development import scores

STATE = {}
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')


def write(path, data):
    with path.open('x') as stream:
        json.dump(anatomy.serial(data), stream, indent=2, allow_nan=False)


def hashes(values):
    return {arm: {name: hashlib.sha256(v.tobytes()).hexdigest() for name, v in fields.items()}
            for arm, fields in values.items()}


def prediction_hashes(path):
    values, _shot = jp.predict(Path(path), STATE['d38_winner'], STATE['d39_change'])
    return hashes(values)


def numeric_difference(left, right):
    """Quantify every cross-host metric field without introducing a tolerance."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        return max((numeric_difference(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        return max((numeric_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.)
    if left is None or isinstance(left, str):
        assert left == right
        return 0.
    assert isinstance(right, (int, float))
    return abs(float(left) - float(right))


def cross_host(case, parts):
    """Bind the complete peer baseline/combo records and compare native parts."""
    result = {}
    if STATE['d39_change'] is None:
        return result
    for central_arm, peer_arm in [(jp.BASE, 'base'), ('d39_finalist', 'combo')]:
        path = PEER / 'variants' / peer_arm / (case['stem'] + '.json')
        digest = jp.cp.digest(path)
        assert digest == STATE['peer_result_sha256'][peer_arm][case['stem']]
        remote = json.loads(path.read_text())
        assert remote['stem'] == case['stem'] and remote['frames'] == case['span'][1] - case['span'][0]
        local = anatomy.serial(parts[central_arm])
        difference = numeric_difference(local, remote['part'])
        result[central_arm] = {'max_abs_numeric_difference': difference,
                               'exact_all_fields': local == remote['part'], 'peer_receipt_sha256': digest}
    return result


def one(case):
    root = jp.HERE / 'central_development'
    target = root / (case['stem'] + '.json')
    array_path = target.with_suffix('.npz')
    if target.exists():
        row = json.loads(target.read_text())
        assert row['lock_sha256'] == STATE['lock_sha256'] and set(row['parts']) == set(STATE['arms'])
        assert jp.cp.digest(array_path) == row['arrays_sha256']
        return row
    started = time.monotonic()
    path = Path(case['path'])
    table = pq.read_table(path, columns=['efit_psirz', 'efit_q95', 'efit_beta_n'], use_threads=False)
    truth = {'psi': np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float32),
             'q95': np.asarray(table['efit_q95'][0].as_py(), dtype=float),
             'betaN': np.asarray(table['efit_beta_n'][0].as_py(), dtype=float)}
    del table
    reference_path = jp.cp.D32 / 'references' / (case['stem'] + '.pkl')
    reference = pickle.loads(reference_path.read_bytes())
    values, shot = jp.predict(path, STATE['d38_winner'], STATE['d39_change'])
    assert set(values) == set(STATE['arms'])
    assert shot.n_frames == len(truth['psi']) == case['span'][1] - case['span'][0]
    parts, arrays = {}, {}
    for arm in STATE['arms']:
        fields = values[arm]
        parts[arm], distance, derived = anatomy.score_maps(truth, reference, fields)
        arrays[arm + '_distance'], arrays[arm + '_derived'] = distance, derived
        arrays[arm + '_flux_mse'] = np.mean((fields['psirz'].astype(float) - truth['psi']) ** 2, axis=(1, 2))
        arrays[arm + '_q95'], arrays[arm + '_betaN'] = fields['q95'], fields['betaN']
    # Require same-host baseline contributions to agree on every whole shot,
    # not only on the qualification example.
    original = json.loads((jp.D38 / 'evaluation/development' / (case['stem'] + '.json')).read_text())
    assert anatomy.serial(parts[jp.BASE]) == original['parts'][jp.BASE], case['stem']
    if STATE['d38_winner'] is not None:
        assert anatomy.serial(parts['d38_finalist']) == original['parts'][STATE['d38_winner']], case['stem']
    host_audit = cross_host(case, parts)
    tmp = array_path.with_name(case['stem'] + '.partial.npz')
    np.savez_compressed(tmp, **arrays); tmp.replace(array_path)
    row = {'stem': case['stem'], 'index': case['index'], 'frames': shot.n_frames,
           'parts': parts, 'arrays_sha256': jp.cp.digest(array_path),
           'reference_sha256': jp.cp.digest(reference_path), 'prediction_hashes': hashes(values),
           'cross_host_audit': host_audit,
           'lock_sha256': STATE['lock_sha256'], 'seconds': time.monotonic() - started}
    write(target, row)
    print(json.dumps({'central_complete_shot': case['index'], 'frames': shot.n_frames}), flush=True)
    return row


def analyze(rows):
    assert len(rows) == 358 and sum(r['frames'] for r in rows) == 78875
    if STATE['d39_change'] is not None:
        audit = {arm: {'max_abs_numeric_difference': max(r['cross_host_audit'][arm]['max_abs_numeric_difference'] for r in rows),
                       'all358_exact_all_fields': all(r['cross_host_audit'][arm]['exact_all_fields'] for r in rows)}
                 for arm in (jp.BASE, 'd39_finalist')}
        write(jp.HERE / 'cross_host_all358_audit.json', audit)
        assert all(v['all358_exact_all_fields'] for v in audit.values()), 'cross-host differences retained; integrity review required before selection'
    references = anatomy.STATISTICS
    parts = {a: [part_numbers(r['parts'][a]) for r in rows] for a in STATE['arms']}
    exact, official = {}, {}
    for arm in STATE['arms']:
        acc = anatomy.scorer.Accum(); acc.psi_sign = 1.
        for p in parts[arm]:
            acc.add(p)
        official[arm] = anatomy.scorer.finalize_machine(acc, anatomy.STATS)
        assert not official[arm]['psi_sign_probe_disagrees']
        exact[arm] = {k: float(v[0]) for k, v in scores(np.ones((1, 358)), references, parts[arm]).items()}
        assert all(round(v, 6) == official[arm][k] for k, v in exact[arm].items())
    rng = np.random.default_rng(3941)
    weights = rng.multinomial(358, np.full(358, 1. / 358), size=10000).astype(float)
    baseline = scores(weights, references, parts[jp.BASE])['S']
    decisions = {}
    for arm in STATE['arms'][1:]:
        delta = exact[arm]['S'] - exact[jp.BASE]['S']
        interval = np.quantile(scores(weights, references, parts[arm])['S'] - baseline, [.005, .995])
        failures = sum(p['lcfs_fails'] for p in parts[arm])
        base_failures = sum(p['lcfs_fails'] for p in parts[jp.BASE])
        finite = sum(p['psi_nonfinite'] for p in parts[arm]) == 0
        passed = delta >= 3e-5 and interval[0] > 0 and failures <= base_failures and finite
        decisions[arm] = {'delta_S': delta, 'CI99': interval.tolist(), 'LCFS_failures': failures,
                          'baseline_LCFS_failures': base_failures, 'finite': finite,
                          'central_composition_gate_pass': bool(passed)}
    # Individual eligibility is already established prospectively by its own
    # study. Only the additional composition uses this selection-cohort gate.
    eligible = [a for a in ('d38_finalist', 'd39_finalist') if a in STATE['arms']]
    if 'composition' in decisions and decisions['composition']['central_composition_gate_pass']:
        eligible.append('composition')
    assert eligible
    winner = max(eligible, key=lambda a: exact[a]['S'])  # list order fixes exact ties
    write(jp.HERE / 'central_development_analysis.json', {
        'status': 'all358_central_locked_finalists_complete', 'shots': 358, 'frames': 78875,
        'unrounded_scores': exact, 'official_scores': official, 'decisions': decisions,
        'one_joint_winner': winner, 'lock_sha256': STATE['lock_sha256'],
        'bootstrap_seed': 3941, 'replicates': 10000, 'CI_is_per_contrast_not_familywise': True,
        'selection_cohort_not_independent_confirmation': True, 'candidate_admitted': False,
        'selection_asymmetry': 'D38 max-of-six uses all358; D39 combination selected on120 and gated on238. One357 confirmation decides admission.',
        'tree_scale_transfer': 'D39 scale, if selected, was selected under D9 trees including357; both reserved arms use new eligible trees. No357 candidate scores used for tuning.',
        'publication': False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    lock_path = jp.HERE / 'concrete_development_finalists_lock.json'
    lock = json.loads(lock_path.read_text())
    assert lock['individual_development_gates_verified'] and lock['locked_before_central_scoring']
    assert json.loads((jp.HERE / 'central_transfer_qualification.json').read_text())['all_six_real_fork_paths_bitexact']
    d38 = lock['d38_winner']; d39 = lock['d39_change']
    if d39 is not None:
        assert not ({'shape', 'reshape_targets', 'oracle', 'oracle_x'} & set(d39)), 'only declared deployable grid knobs'
    assert d38 is not None or d39 is not None
    arms = [jp.BASE] + (['d38_finalist'] if d38 is not None else []) + (['d39_finalist'] if d39 is not None else [])
    if d38 is not None and d39 is not None:
        arms.append('composition')
    STATE.update(d38_winner=d38, d39_change=d39, arms=arms, lock_sha256=jp.cp.digest(lock_path),
                 peer_result_sha256=lock.get('peer_result_sha256', {}))
    (jp.HERE / 'central_development').mkdir(exist_ok=True)
    jp.initialize()
    if args.qualify:
        actual = prediction_hashes(CASES[0]['path'])
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
            inherited = pool.submit(prediction_hashes, CASES[0]['path']).result(timeout=1800)
        assert inherited == actual
        write(jp.HERE / 'concrete_finalist_qualification.json', {
            'status': 'concrete_locked_finalists_whole213_frame_real_fork_bitexact',
            'prediction_hashes': actual, 'lock_sha256': STATE['lock_sha256'],
            'candidate_admitted': False})
        one(CASES[0]); return
    assert json.loads((jp.HERE / 'concrete_finalist_qualification.json').read_text())['lock_sha256'] == STATE['lock_sha256']
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for completed, _row in enumerate(pool.map(one, CASES, chunksize=1), 1):
            (jp.HERE / 'central_development_state.json').write_text(json.dumps({'completed': completed, 'total': 358, 'UTC_seconds': time.time()}))
    rows = [json.loads((jp.HERE / 'central_development' / (c['stem'] + '.json')).read_text()) for c in CASES]
    analyze(rows)


if __name__ == '__main__':
    main()
