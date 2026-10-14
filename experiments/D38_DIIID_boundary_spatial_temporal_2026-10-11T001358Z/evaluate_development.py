# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — whole-shot complete-69 development evaluation
"""Score every declared arm on all frames before comparative reduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from campaign_common import HERE, digest, write_json
import campaign_prediction as prediction

D37 = HERE.parent / 'D37_H69_boundary_anatomy_2026-10-10T235436Z'
sys.path.insert(0, str(D37))
import anatomy

CASES = json.loads((prediction.D32 / 'manifest.json').read_text())['cases']
assert len(CASES) == 358
assert [c['stem'] for c in CASES] == sorted(json.loads((HERE / 'cohort.json').read_text())['development'])


def part_numbers(part: dict) -> dict:
    """Restore native float-key metric mappings after JSON serialization."""
    part = dict(part)
    for name in ('ss_res_psi', 'shot_r2_psi'):
        if part[name] is not None:
            part[name] = {float(k): v for k, v in part[name].items()}
    return part


def one_case(case: dict) -> dict:
    """Retain all seven official arm contributions and frame attribution."""
    out = HERE / 'evaluation/development'
    receipt = out / (case['stem'] + '.json')
    arrays = receipt.with_suffix('.npz')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert digest(arrays) == row['arrays_sha256']
        return row
    started = time.monotonic()
    path = Path(case['path'])
    targets = pq.read_table(path, columns=['efit_psirz', 'efit_q95', 'efit_beta_n'], use_threads=False)
    truth = {'psi': np.asarray(targets['efit_psirz'][0].as_py(), dtype=np.float32),
             'q95': np.asarray(targets['efit_q95'][0].as_py(), dtype=float),
             'betaN': np.asarray(targets['efit_beta_n'][0].as_py(), dtype=float)}
    del targets
    reference_path = prediction.D32 / 'references' / (case['stem'] + '.pkl')
    reference = pickle.loads(reference_path.read_bytes())
    predictions, shot = prediction.predict(path)
    assert shot.n_frames == len(truth['psi']) == case['span'][1] - case['span'][0]
    metrics, saved = {}, {}
    for arm in prediction.ARMS:
        values = predictions[arm]
        metrics[arm], distances, derived = anatomy.score_maps(truth, reference, values)
        saved[arm + '_distance'] = distances
        saved[arm + '_derived'] = derived
        saved[arm + '_flux_mse'] = np.mean((values['psirz'].astype(float) - truth['psi']) ** 2, axis=(1, 2))
        saved[arm + '_q95'] = values['q95']
        saved[arm + '_betaN'] = values['betaN']
    saved.update(times_ms=shot.times_ms, ip_A=shot.plasma_current, truth_derived=reference[1], truth_defined=reference[2])
    temporary = arrays.with_name(case['stem'] + '.partial.npz')
    np.savez_compressed(temporary, **saved)
    temporary.replace(arrays)
    row = {'stem': case['stem'], 'index': case['index'], 'frames': shot.n_frames,
           'parts': anatomy.serial(metrics), 'arrays_sha256': digest(arrays),
           'reference_sha256': digest(reference_path), 'seconds': time.monotonic() - started}
    write_json(receipt, row)
    print(json.dumps({'development_shot_complete': case['index'], 'frames': shot.n_frames, 'seconds': row['seconds']}), flush=True)
    return row


def aggregate(rows: list) -> dict:
    """Use the unchanged scorer reduction on the complete declared development set."""
    assert len(rows) == 358 and sum(r['frames'] for r in rows) == 78875
    results = {}
    for arm in prediction.ARMS:
        acc = anatomy.scorer.Accum()
        acc.psi_sign = 1.
        for row in rows:
            acc.add(part_numbers(row['parts'][arm]))
        result = anatomy.scorer.finalize_machine(acc, anatomy.STATS)
        assert not result['psi_sign_probe_disagrees']
        results[arm] = result
    return results


def fork_hashes(path: str) -> dict:
    """Exercise inherited models through a real forked whole-shot prediction."""
    values, _shot = prediction.predict(Path(path))
    return {arm: {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in fields.items()}
            for arm, fields in values.items()}


def main() -> None:
    """Qualify the actual native finisher or complete all358 seven-arm scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = HERE / 'evaluation/development'
    out.mkdir(exist_ok=True)
    prediction.initialize()
    if args.qualify:
        values, shot = prediction.predict(Path(CASES[0]['path']), qualify=True)
        hashes = {arm: {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in fields.items()}
                  for arm, fields in values.items()}
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
            inherited = pool.submit(fork_hashes, CASES[0]['path']).result(timeout=1800)
        assert inherited == hashes, 'forked whole-shot predictions differ'
        qualification = {'status': 'full69_native_map_and_scalar_finisher_bitexact_qualified', 'frames': shot.n_frames,
                         'all_seven_arms_finite_native_float32': True, 'development_and_confirmation_excluded_from_neural_fits': True,
                         'all_seven_arms_real_fork_bitexact': True,
                         'prediction_hashes': hashes}
        write_json(HERE / 'evaluation/qualification.json', qualification)
        one_case(CASES[0])
        return
    assert json.loads((HERE / 'evaluation/qualification.json').read_text())['status'] == 'full69_native_map_and_scalar_finisher_bitexact_qualified'
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for completed, _row in enumerate(pool.map(one_case, CASES, chunksize=1), 1):
            (HERE / 'evaluation/state.json').write_text(json.dumps({'complete_shots': completed, 'total': 358, 'UTC_seconds': time.time()}))
    rows = [json.loads((out / (c['stem'] + '.json')).read_text()) for c in CASES]
    results = aggregate(rows)
    write_json(HERE / 'evaluation/development_scores.json', {'status': 'all358_whole_shot_seven_arm_scores_complete',
               'shots': 358, 'frames': 78875, 'results': results, 'native_finalization': True,
               'all125_neural_fits_terminal_before_evaluation': True, 'candidate_admitted': False})


if __name__ == '__main__':
    main()
