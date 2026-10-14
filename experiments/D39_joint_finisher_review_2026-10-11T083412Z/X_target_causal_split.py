# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — diagnostic X position and validity factorial
"""Intervene after native target smoothing, never on a deployable prediction path."""
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
from evaluate_development import CASES, anatomy, part_numbers
from analyze_development import scores

ROOT = jp.HERE / 'X_target_causal_split'
ARMS = ('baseline', 'positions_only', 'validity_only', 'positions_and_validity')
SELECTED = [c for c in CASES if c['index'] % 3 == 0]
assert len(SELECTED) == 120
STATE = {}


def write(path, value):
    with path.open('x') as stream:
        json.dump(anatomy.serial(value), stream, indent=2, allow_nan=False)


def truth_of(path):
    table = pq.read_table(path, columns=['efit_psirz', 'efit_q95', 'efit_beta_n'], use_threads=False)
    return {'psi': np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float32),
            'q95': np.asarray(table['efit_q95'][0].as_py(), dtype=float),
            'betaN': np.asarray(table['efit_beta_n'][0].as_py(), dtype=float)}


def initialize():
    jp.initialize()
    STATE['protocol_sha256'] = jp.cp.digest(jp.HERE / 'X_target_causal_split_protocol_v2.json')
    assert jp.cp.STATE.finish.x_points


def predict(path, truth, noop=False):
    cp, pd = jp.cp, jp.cp.STATE.pd
    row = cp.read_inputs(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
    assert shot.n_frames == len(truth['psi'])
    features = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'],
                               cp.STATE.subframe.features_of(path).astype(float)], axis=1)
    coordinates, labels = cp.STATE.together(cp.STATE.maps, {'discharge': features})
    trees = cp.trees_of(features)
    base = jp.finished(path, row, features, coordinates, labels, trees, {})
    _, joint = cp.STATE.together(cp.STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
    scalars = jp.blend_scalars(base, joint)
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    from fec_adapter.diii_d_accuracy.x_point import x_point_positions, LABELS
    reader = Reader(str(shot.source))
    reference_x = np.asarray([x_point_positions(reader, m.astype(float)) for m in truth['psi']])
    assert reference_x.shape == (shot.n_frames, 4)
    result = {'baseline': {'psirz': base[0], **scalars}}
    kinds = ARMS if noop else ARMS[1:]
    for kind in kinds:
        original = pd.edited_x_points
        calls = []

        def intervention(maps, targets, source, **options):
            copied = {n: np.asarray(v, dtype=float).copy() for n, v in targets.items()}
            assert all(v.shape == (shot.n_frames,) for v in copied.values())
            for end in (0, 2):
                defined = np.isfinite(reference_x[:, end:end + 2]).all(axis=1)
                for column in (end, end + 1):
                    name = LABELS[column]
                    if kind in ('positions_only', 'positions_and_validity'):
                        copied[name][defined] = reference_x[defined, column]
                    if kind in ('validity_only', 'positions_and_validity'):
                        copied[name][~defined] = np.nan
            calls.append(1)
            return original(maps, copied, source, **options)

        pd.edited_x_points = intervention
        try:
            final = jp.finished(path, row, features, coordinates, labels, trees, {})
            assert len(calls) == 1, 'must modify exactly the actual post-smoothing X call'
        finally:
            pd.edited_x_points = original
        fields = {'psirz': final[0], **jp.blend_scalars(final, joint)}
        assert all(np.array_equal(fields[n], scalars[n]) for n in scalars)
        if kind == 'baseline':
            assert all(np.array_equal(fields[n], result['baseline'][n]) for n in fields), 'noop hook changed baseline'
        result[kind] = fields
    assert set(result) == set(ARMS)
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for a in result.values() for v in a.values())
    return result, shot, {'lower': int(np.isfinite(reference_x[:, :2]).all(axis=1).sum()),
                          'upper': int(np.isfinite(reference_x[:, 2:]).all(axis=1).sum())}


def hashes(path):
    path = Path(path)
    values, _shot, _valid = predict(path, truth_of(path), noop=True)
    return {a: {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in fields.items()} for a, fields in values.items()}


def one(case):
    out = ROOT / 'cases' / (case['stem'] + '.json')
    arrays_path = out.with_suffix('.npz')
    if out.exists():
        receipt = json.loads(out.read_text())
        assert receipt['protocol_sha256'] == STATE['protocol_sha256']
        assert jp.cp.digest(arrays_path) == receipt['arrays_sha256']
        return receipt
    started = time.monotonic()
    path = Path(case['path']); truth = truth_of(path)
    values, shot, validity = predict(path, truth)
    reference_path = jp.cp.D32 / 'references' / (case['stem'] + '.pkl')
    reference = pickle.loads(reference_path.read_bytes())
    parts, arrays = {}, {}
    for arm, fields in values.items():
        parts[arm], distance, derived = anatomy.score_maps(truth, reference, fields)
        arrays[arm + '_distance'], arrays[arm + '_derived'] = distance, derived
    expected = json.loads((jp.D38 / 'evaluation/development' / (case['stem'] + '.json')).read_text())
    assert anatomy.serial(parts['baseline']) == expected['parts'][jp.BASE], case['stem']
    tmp = arrays_path.with_name(case['stem'] + '.partial.npz')
    np.savez_compressed(tmp, **arrays); tmp.replace(arrays_path)
    receipt = {'stem': case['stem'], 'index': case['index'], 'frames': shot.n_frames,
               'parts': parts, 'reference_valid_ends': validity,
               'protocol_sha256': STATE['protocol_sha256'], 'arrays_sha256': jp.cp.digest(arrays_path),
               'reference_sha256': jp.cp.digest(reference_path), 'seconds': time.monotonic() - started}
    write(out, receipt)
    print(json.dumps({'X_split_complete': case['index'], 'frames': shot.n_frames}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--workers', type=int, default=2); args = parser.parse_args()
    ROOT.mkdir(exist_ok=True); initialize()
    if args.qualify:
        actual = hashes(CASES[0]['path'])
        expected = json.loads((jp.D38 / 'evaluation/qualification.json').read_text())['prediction_hashes'][jp.BASE]
        assert actual['baseline'] == expected
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
            assert pool.submit(hashes, CASES[0]['path']).result(timeout=1800) == actual
        write(ROOT / 'qualification.json', {'status': 'post_smoothing_X_factorial_native_noop_and_real_fork_bitexact',
              'prediction_hashes': actual, 'protocol_sha256': STATE['protocol_sha256'], 'no357_targets_read': True})
        return
    assert json.loads((ROOT / 'qualification.json').read_text())['protocol_sha256'] == STATE['protocol_sha256']
    (ROOT / 'cases').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        rows = list(pool.map(one, SELECTED, chunksize=1))
    assert len(rows) == 120 and sum(r['frames'] for r in rows) == sum(c['span'][1] - c['span'][0] for c in SELECTED)
    references = [anatomy.STATISTICS[c['index']] for c in SELECTED]
    parts = {a: [part_numbers(r['parts'][a]) for r in rows] for a in ARMS}
    exact = {a: {k: float(v[0]) for k, v in scores(np.ones((1, 120)), references, parts[a]).items()} for a in ARMS}
    weights = np.random.default_rng(3942).multinomial(120, np.full(120, 1. / 120), size=10000).astype(float)
    sampled = {a: scores(weights, references, parts[a])['S'] for a in ARMS}
    contrasts = {a: {'delta_S': exact[a]['S'] - exact['baseline']['S'],
                     'CI99': np.quantile(sampled[a] - sampled['baseline'], [.005, .995]).tolist()} for a in ARMS[1:]}
    interaction = sampled['positions_and_validity'] - sampled['positions_only'] - sampled['validity_only'] + sampled['baseline']
    contrasts['interaction'] = {'delta_S': exact['positions_and_validity']['S'] - exact['positions_only']['S'] - exact['validity_only']['S'] + exact['baseline']['S'],
                                'CI99': np.quantile(interaction, [.005, .995]).tolist()}
    write(ROOT / 'analysis.json', {'status': 'all4x120_post_smoothing_X_causal_diagnostic_complete',
          'shots': 120, 'frames': sum(r['frames'] for r in rows), 'unrounded_scores': exact, 'contrasts': contrasts,
          'protocol_sha256': STATE['protocol_sha256'], 'bootstrap_seed': 3942, 'replicates': 10000,
          'extractor_defined_validity_not_physical_absence': True, 'nondeployable_nonselectable': True,
          'not_peer_pre_smoothing_oracle_equivalent': True, 'CI_per_contrast_not_familywise': True,
          'current_selection_unchanged': True, 'no357_targets_read': True, 'publication': False})


if __name__ == '__main__':
    main()
