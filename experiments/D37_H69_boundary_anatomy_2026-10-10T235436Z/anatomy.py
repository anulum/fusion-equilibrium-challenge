# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — score every native map stage on whole historical shots
"""Run full official metrics and retain per-frame attribution for all stages."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import production_path as production
from production_path import D32, HERE, R, digest

sys.path.insert(0, str(R / 'fec_harness'))
import run_parallel_score as scorer

PROTOCOL = json.loads((HERE / 'protocol_v3.json').read_text())
ARMS = PROTOCOL['arms']
STATISTICS = json.loads((D32 / 'reference_statistics.json').read_text())
STATS = {key: sum(np.asarray(row[key]) for row in STATISTICS) for key in STATISTICS[0]}
MEANS = (STATS['psi_sum'] / STATS['psi_n'], STATS['scal_sum'] / np.maximum(STATS['scal_n'], 1),
         STATS['cons_sum'] / np.maximum(STATS['cons_n'], 1))


def serial(value):
    """Convert native NumPy metric structures to JSON without changing values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def write_json(path: Path, value: dict) -> None:
    """Write a new immutable receipt."""
    with path.open('x') as stream:
        json.dump(serial(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def score_maps(truth: dict, reference: tuple, prediction: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Capture native LCFS/derived values during the unmodified official score loop."""
    radius, height, coarse, mask = scorer._grid()
    contours, derived, defined, rgeo = reference
    eligible = iter(k for k in range(len(contours)) if contours[k] is not None or defined[k].any())
    distances = np.full(len(contours), np.nan)
    values = np.full_like(derived, np.nan, dtype=float)
    active = [-1]
    local = scorer.local_score
    extract, hausdorff, derive = local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame

    def extracted(*args, **kwargs):
        index = next(eligible)
        active[0] = index
        result = extract(*args, **kwargs)
        if result[0] is None and contours[index] is not None:
            distances[index] = 1.
        return result

    def distance(*args, **kwargs):
        result = hausdorff(*args, **kwargs)
        distances[active[0]] = min(1., result / rgeo if rgeo > 0 else 1.)
        return result

    def derived_frame(*args, **kwargs):
        result = derive(*args, **kwargs)
        values[active[0]] = [result[name] for name in local.CONS_SCALARS]
        return result

    local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame = extracted, distance, derived_frame
    try:
        result = local.score_shot(truth, reference, prediction, radius, height, coarse, mask, 1., MEANS)
    finally:
        local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame = extract, hausdorff, derive
    assert next(eligible, None) is None
    if result['shot_dlcfs'] is not None:
        assert abs(np.nanmean(distances) - result['shot_dlcfs']) < 1e-14
    return result, distances, values


def one_case(case: dict) -> dict:
    """Process all declared stages of one entire shot and save resumable evidence."""
    stem = case['stem']
    receipt = HERE / 'cases' / (stem + '.json')
    arrays = receipt.with_suffix('.npz')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert digest(arrays) == row['arrays_sha256']
        return row
    started = time.monotonic()
    import pyarrow.parquet as pq
    with_targets = pq.read_table(case['path'], columns=['efit_psirz', 'efit_q95', 'efit_beta_n'])
    truth = {'psi': np.asarray(with_targets['efit_psirz'][0].as_py(), dtype=np.float32),
        'q95': np.asarray(with_targets['efit_q95'][0].as_py(), dtype=float),
        'betaN': np.asarray(with_targets['efit_beta_n'][0].as_py(), dtype=float)}
    del with_targets
    reference_path = D32 / 'references' / (stem + '.pkl')
    reference = pickle.loads(reference_path.read_bytes())
    maps, scalars, shot = production.maps_of(Path(case['path']), truth['psi'])
    assert set(maps) == set(ARMS) and len(truth['psi']) == case['span'][1] - case['span'][0]
    parts, per_frame, stage_seconds = {}, {}, {}
    for arm in ARMS:
        assert maps[arm].dtype == np.float32 and np.isfinite(maps[arm]).all()
        before = time.monotonic()
        part, distance, values = score_maps(truth, reference, {'psirz': maps[arm], **scalars})
        parts[arm] = part
        per_frame[arm + '_distance'] = distance
        per_frame[arm + '_derived'] = values
        per_frame[arm + '_flux_mse'] = np.mean((maps[arm].astype(float) - truth['psi']) ** 2, axis=(1, 2))
        stage_seconds[arm] = time.monotonic() - before
    per_frame['times_ms'] = shot.times_ms
    per_frame['ip_A'] = shot.plasma_current
    per_frame['truth_derived'] = reference[1]
    per_frame['truth_defined'] = reference[2]
    temporary = arrays.with_name(stem + '.partial.npz')
    np.savez_compressed(temporary, **per_frame)
    temporary.replace(arrays)
    production.STATE.pd.read_row = production.STATE.read_row
    production.STATE.pd.layout_features = production.STATE.layout_features
    production.STATE.pd.predict_together = production.STATE.together
    row = {'index': case['index'], 'stem': stem, 'frames': len(truth['psi']), 'parts': parts,
        'arrays_sha256': digest(arrays), 'reference_sha256': digest(reference_path),
        'stage_seconds': stage_seconds, 'seconds': time.monotonic() - started}
    write_json(receipt, row)
    print(json.dumps({'case_complete': case['index'], 'stem': stem, 'frames': row['frames'],
                      'seconds': row['seconds']}), flush=True)
    return serial(row)


def qualify() -> None:
    """Require bit-exact real public archive reproduction before diagnostics."""
    production.initialize()
    data = Path('$FEC_DATA/data/diii_d_public_test')
    path = sorted(data.glob('*.parquet'))[0]
    maps, scalars, _shot = production.maps_of(path)
    published = HERE.parent / 'D36_production_scalar_submission_2026-10-10T223719Z/submission/diii_d_public_test.npz'
    with np.load(published, allow_pickle=False) as z:
        archive_variation = {}
        for field, actual in [('psirz', maps['final']), ('q95', scalars['q95']), ('betaN', scalars['betaN'])]:
            expected = z['shot_0000_' + field]
            archive_variation[field] = {'bitexact': bool(np.array_equal(expected, actual)),
                'max_abs': float(np.max(np.abs(expected - actual))), 'different_values': int(np.sum(expected != actual))}
            if field == 'psirz':
                native_path = HERE / 'profiles/uninstrumented_public_psirz.npy'
                assert digest(native_path) == PROTOCOL['original_uninstrumented_public_map_sha256']
                native = np.load(native_path, allow_pickle=False)
                assert np.array_equal(native, actual), 'instrumented map differs from current original uninstrumented builder'
            else:
                assert np.array_equal(expected, actual), (field, float(np.max(np.abs(expected - actual))))
    write_json(HERE / 'public_parity.json', {'status': 'instrumented_map_bitexact_current_native_D36_scalars_bitexact_archive',
        'stem': path.stem, 'models': 69, 'joint_scalar_models': 8, 'prediction_inputs_only': True,
        'baseline_archive': str(published), 'numerical_gate_uses_no_tolerance': True,
        'archived_map_strict_parity_remains_failed': True, 'archive_variation': archive_variation,
        'original_builder_also_has_archived_map_difference': True})
    row = one_case(PROTOCOL['cases'][0])
    print(json.dumps({'qualification_seconds_per_whole_shot': row['seconds'],
        'frames': row['frames'], 'stages': len(row['parts'])}), flush=True)


def main() -> None:
    """Complete all cases before any comparative aggregate interpretation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.qualify:
        qualify()
        return
    assert json.loads((HERE / 'public_parity.json').read_text())['status'] == 'instrumented_map_bitexact_current_native_D36_scalars_bitexact_archive'
    assert json.loads((HERE / 'fork_parity.json').read_text())['status'] == 'real_forked_current_native_map_and_archived_scalars_bitexact'
    production.initialize()
    context = multiprocessing.get_context('fork')
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for row in pool.map(one_case, PROTOCOL['cases'], chunksize=1):
            completed += 1
            state = {'phase': 'all_stages_per_whole_shot', 'completed': completed, 'total': 358, 'UTC_seconds': time.time()}
            (HERE / 'run_state.partial.json').write_text(json.dumps(state))
            (HERE / 'run_state.partial.json').replace(HERE / 'run_state.json')
    rows = [json.loads((HERE / 'cases' / (case['stem'] + '.json')).read_text()) for case in PROTOCOL['cases']]
    assert len(rows) == 358 and sum(row['frames'] for row in rows) == 78875
    results = {}
    for arm in ARMS:
        acc = scorer.Accum()
        acc.psi_sign = 1.
        for row in rows:
            part = row['parts'][arm]
            part['ss_res_psi'] = {float(k): v for k, v in part['ss_res_psi'].items()}
            part['shot_r2_psi'] = {float(k): v for k, v in part['shot_r2_psi'].items()}
            acc.add(part)
        results[arm] = scorer.finalize_machine(acc, STATS)
    write_json(HERE / 'summary.json', {'status': 'all358_all14_stage_official_scores_complete',
        'shots': 358, 'frames': 78875, 'results': results, 'diagnostic_only': True,
        'in_sample_production_models': True, 'new_candidate_admitted': False})
    (HERE / 'run_state.json').write_text(json.dumps({'phase': 'complete', 'completed': 358, 'total': 358}))
    print(json.dumps({'status': 'diagnostic_complete', 'shots': 358, 'stages': len(results)}), flush=True)


if __name__ == '__main__':
    main()
