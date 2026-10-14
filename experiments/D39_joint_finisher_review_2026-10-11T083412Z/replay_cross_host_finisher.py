# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed-cache cross-host finisher replay
"""Isolate raw arithmetic on four already reviewed cases, using one native host."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np
import joint_prediction as jp
from X_target_causal_split import truth_of
from evaluate_development import anatomy

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cross_host_finisher_replay'
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(anatomy.serial(value), stream, indent=2, allow_nan=False)


def verify():
    for path, expected in json.loads((OUT / 'sources.json').read_text()).items():
        assert digest(path) == expected, path


def compare(left, right):
    assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    difference = left.astype(float)-right.astype(float)
    return {'exact': bool(np.array_equal(left, right)), 'differing_elements': int(np.count_nonzero(difference)),
            'max_abs': float(np.max(np.abs(difference))), 'RMS': float(np.sqrt(np.mean(difference**2)))}


def one(case):
    started = time.monotonic()
    path = Path(case['path'])
    cp, pd = jp.cp, jp.cp.STATE.pd
    row = cp.read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    features = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'],
                               cp.STATE.subframe.features_of(path).astype(float)], axis=1)
    coordinates, labels = cp.STATE.together(cp.STATE.maps, {'discharge': features})
    _, joint = cp.STATE.together(cp.STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    trees = cp.trees_of(features)
    with np.load(PEER / 'cache' / (path.stem+'.npz'), allow_pickle=False) as stored:
        cached = {name: stored[name].copy() for name in stored.files}
    assert np.array_equal(features.astype(np.float32), cached['features'])
    assert np.array_equal(shot.r_grid, cached['r_grid']) and np.array_equal(shot.z_grid, cached['z_grid'])
    assert str(shot.source) == str(cached['source'])
    cached_labels = {str(name): cached['labels'][:, i] for i, name in enumerate(cached['label_names'])}
    assert set(cached_labels) == set(labels)
    for i, name in enumerate(cached['tree_names']):
        assert np.array_equal(trees[str(name)], cached['tree'][:, i])
    truth = truth_of(path)
    reference = pickle.loads((cp.D32 / 'references' / (path.stem+'.pkl')).read_bytes())
    predictions, parts, arrays = {}, {}, {}
    for arm, co, heads, scalar in [('root_raw', coordinates, labels, joint),
                                  ('peer_raw', cached['coordinates'], cached_labels,
                                   {'q95': cached['joint_q95'], 'beta_n': cached['joint_beta_n']})]:
        finished = jp.finished(path, row, features, co, heads, trees, {})
        values = {'psirz': finished[0], **jp.blend_scalars(finished, scalar)}
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in values.values())
        predictions[arm] = values
        part, distance, derived = anatomy.score_maps(truth, reference, values)
        parts[arm] = anatomy.serial(part)
        arrays[arm+'_distance'], arrays[arm+'_derived'] = distance, derived
    root_record = json.loads((HERE / 'X_target_causal_split/cases' / (path.stem+'.json')).read_text())
    assert parts['root_raw'] == root_record['parts']['baseline'], 'same-host original baseline did not reproduce'
    peer_record = json.loads((PEER / 'variants/base' / (path.stem+'.json')).read_text())
    differences = {name: compare(predictions['root_raw'][name], predictions['peer_raw'][name]) for name in predictions['root_raw']}
    np.savez_compressed(OUT / (path.stem+'.npz'), **arrays)
    report = {'stem': path.stem, 'frames': shot.n_frames, 'parts': parts,
              'root_raw_exact_original_root_parts': True,
              'peer_raw_exact_original_remote_parts': parts['peer_raw'] == peer_record['part'],
              'prediction_differences_same_host': differences,
              'prediction_hashes': {arm: {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in fields.items()}
                                    for arm, fields in predictions.items()},
              'arrays_sha256': digest(OUT / (path.stem+'.npz')), 'seconds': time.monotonic()-started}
    write(OUT / (path.stem+'.json'), report)
    print(json.dumps({'finisher_replay_complete': path.stem, 'peer_metric_exact': report['peer_raw_exact_original_remote_parts']}), flush=True)
    return report


def main():
    started = time.monotonic()
    try:
        verify()
        protocol = json.loads((OUT / 'protocol.json').read_text())
        cases = {c['stem']: c for c in json.loads((jp.cp.D32 / 'manifest.json').read_text())['cases']}
        jp.initialize()
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('fork')) as pool:
            reports = list(pool.map(one, [cases[s] for s in protocol['cases']], chunksize=1))
        verify()
        write(OUT / 'analysis.json', {'status': 'all4_fixed_raw_native_finisher_replays_complete', 'reports': reports,
              'root_raw_reproduces_original_parts': all(r['root_raw_exact_original_root_parts'] for r in reports),
              'peer_raw_reproduces_remote_parts': all(r['peer_raw_exact_original_remote_parts'] for r in reports),
              'raw_prediction_rounding_amplification_isolated_same_host': any(not r['prediction_differences_same_host']['psirz']['exact'] for r in reports),
              'not_candidate_admission_no_gate_tolerance_waiver': True, 'no_reserved357_targets_read': True,
              'publication': False, 'seconds': time.monotonic()-started})
    except Exception as error:
        write(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
