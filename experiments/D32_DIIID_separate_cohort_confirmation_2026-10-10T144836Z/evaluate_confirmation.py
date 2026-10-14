# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed separate-cohort confirmation of D31
"""Predict every declared shot before interpreting the common official scores."""

import argparse
import json
import multiprocessing
import pickle
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from campaign_common import D26, HERE, SAS, SEAT, base_maps, digest, feature_block, write_json

sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(D26))
import xgboost as xgb
import finish_and_score as previous

from tree_replica import lookup_for, predict_like_gpu

STATE = {}


def initialize(manifest):
    previous.initialize(manifest, HERE)
    trees, lookups, lookup_receipts = {}, {}, {}
    for name, record in manifest['trees'].items():
        assert digest(record['path']) == record['sha256']
        booster = xgb.Booster()
        booster.load_model(record['path'])
        booster.set_param({'device': 'cpu', 'nthread': 1})
        assert int(booster.attr('best_iteration')) == record['best_iteration']
        trees[name] = booster
        matrix, base, receipt = lookup_for(name, booster, record)
        lookups[name] = (matrix, base)
        lookup_receipts[name] = receipt
    if (HERE / 'qualification.json').exists():
        expected = json.loads((HERE / 'qualification.json').read_text())['lookup_receipts']
        assert lookup_receipts == expected
    STATE.update(manifest=manifest, base=base_maps(), trees=trees, lookups=lookups, lookup_receipts=lookup_receipts, fold=None,
                 protocol=json.loads((HERE / 'protocol.json').read_text()))


def load_fold(fold):
    if STATE['fold'] == fold:
        return
    pd = previous.STATE['pd']
    networks = {}
    for arm, roles in STATE['protocol']['recipes'].items():
        paths = [m['path'] for m in STATE['manifest']['models'] if m['fold'] == fold and m['role'] in roles]
        assert len(paths) == 4
        networks[arm] = [pd.Network.load(p) for p in paths]
    STATE.update(networks=networks, fold=fold)


def tree_predictions(features):
    inputs = np.asarray(features[:, :367], dtype=np.float32, order='C')
    return {name: predict_like_gpu(booster, inputs, *STATE['lookups'][name])
            for name, booster in STATE['trees'].items()}


def build_case(case):
    path = HERE / 'shots' / (case['stem'] + '.npz')
    receipt = path.with_suffix('.json')
    if receipt.exists():
        kept = json.loads(receipt.read_text())
        assert digest(path) == kept['sha256']
        return kept
    load_fold(case['fold'])
    pd = previous.STATE['pd']
    file = Path(case['path'])
    row = previous.STATE['raw_read'](file)
    a, b = case['span']
    features = feature_block(STATE['base']['training'], {}, np.arange(a, b), 'control')
    assert len(row['efit_times']) == b - a
    if case['index'] == 0:
        shot = pd.aligned_shot(row, case['stem'], np.asarray(row['efit_times'], dtype=float))
        from fec_adapter.diii_d_accuracy.history import DischargeRecord
        # Reproduce the actual original training extraction storage precision.
        record = DischargeRecord.from_row(row, case['stem'])
        live = DischargeRecord.from_stored(record.arrays()).features()
        sys.path.insert(0, str(SEAT / 'd16_subframe'))
        from subframe_features import features_of
        subframe = features_of(file)
        assembled = np.column_stack([live.astype(np.float32), subframe])
        assert assembled.shape == (b - a, 582)
        assert np.array_equal(assembled.view(np.uint32), features[:, :582].astype(np.float32).view(np.uint32))
        if not (HERE / 'feature_qualification.json').exists():
            write_json(HERE / 'feature_qualification.json', {
                'status': 'all582_live_input_features_match_heldout_table', 'stem': case['stem'],
                'max_abs': float(np.max(np.abs(assembled - features[:, :582]))),
                'label_columns_read': False, 'frames': b - a, 'all582_bit_exact': True,
                'original_float32_aligned_input_storage_roundtrip': True})
    tree = tree_predictions(features)
    assert all(np.isfinite(v).all() for v in tree.values())
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    previous.STATE['stage_active'] = False
    arrays = {}
    for arm, networks in STATE['networks'].items():
        coordinates, labels = previous.STATE['raw_together'](networks, {'discharge': features})
        if case['index'] == 0:
            answers = [net.predict(features[:, :len(net.feature_mean)]) for net in networks]
            assert np.array_equal(coordinates, np.mean([v[0] for v in answers], axis=0))
            for name, values in labels.items():
                assert np.array_equal(values, np.mean([v[1][name] for v in answers if name in v[1]], axis=0))
        for name, weight in STATE['protocol']['tree_weights'].items():
            labels[name] += weight * (tree[name].astype(float) - labels[name])
        pd.predict_together = lambda *_args: (coordinates.copy(), {k: v.copy() for k, v in labels.items()})
        predicted = pd.predict_shot(file, networks, previous.STATE['basis'],
                                    replace(previous.STATE['finish'], kept_coordinates=45))
        for field, values in zip(('psirz', 'q95', 'betaN'), predicted, strict=True):
            assert np.isfinite(values).all() and len(values) == b - a
            arrays[arm + '_' + field] = np.asarray(values, dtype=np.float32)
    temporary = path.with_name(case['stem'] + '.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)
    record = {'index': case['index'], 'stem': case['stem'], 'frames': b - a,
              'arms': STATE['manifest']['arms'], 'sha256': digest(path)}
    write_json(receipt, record)
    print(json.dumps({'confirmation_prediction_case_complete': case['index']}), flush=True)
    return record


def qualify(manifest):
    initialize(manifest)
    root = SAS / 'workstation/first_agent_workspace_20261008/d9_gbm_heads'
    with np.load(root / 'kaggle_ds_test/test_features.npz', allow_pickle=False) as stored:
        all_features = stored['features']
        features = all_features[:128]
        distributed_indices = np.unique(np.linspace(0, len(all_features) - 1, 1024, dtype=int))
        distributed = all_features[distributed_indices]
    with np.load(root / 'kaggle_out_deploy/tree_heads_public_test.npz', allow_pickle=False) as stored:
        names = stored['targets'].astype(str).tolist()
        all_reference = stored['pred']
        reference = all_reference[:128]
        distributed_reference = all_reference[distributed_indices]
    heads = tree_predictions(features)
    errors = {}
    for name, values in heads.items():
        target = reference[:, names.index(name)]
        errors[name] = float(np.max(np.abs(target - values)))
        assert np.array_equal(target, values), (name, errors[name])
    distributed_heads = tree_predictions(distributed)
    for name, values in distributed_heads.items():
        assert np.array_equal(values, distributed_reference[:, names.index(name)]), name
    if not (HERE / 'tree_qualification.json').exists():
        write_json(HERE / 'tree_qualification.json', {'status': 'all4_original_tree_head_predictions_match',
               'xgboost': xgb.__version__, 'rows': 128, 'distributed_rows': len(distributed_indices), 'max_abs': errors,
               'bit_exact': True, 'comparison_tolerance_relaxed': False, 'cohort_labels_read': False})
    first = build_case(manifest['cases'][0])
    write_json(HERE / 'qualification.json', {'status': 'real_separate_cohort_path_qualified',
               'first_prediction': first, 'all582_feature_parity': True,
               'mixed_network_public_mean_parity': True, 'all4_tree_head_parity': True,
               'comparative_scores_processed': False, 'lookup_receipts': STATE['lookup_receipts']})


def scored_case(task):
    case, arm, means = task
    score = previous.scorer()
    truth = score._truth(Path(case['path']))
    reference = pickle.loads((HERE / 'references' / (case['stem'] + '.pkl')).read_bytes())
    prediction = score._prediction(HERE / 'evaluation' / (arm + '.npz'), case['index'])
    r, z, coarse, floating = score._grid()
    row = score.local_score.score_shot(truth, reference, prediction, r, z, coarse, floating, 1., means)
    print(json.dumps({'confirmation_score_case_complete': case['index'], 'arm': arm}), flush=True)
    return case['index'], row


def cached_parts(path):
    rows = json.loads(path.read_text())
    for row in rows:
        for name in ('ss_res_psi', 'shot_r2_psi'):
            row[name] = {float(k): v for k, v in row[name].items()}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if args.qualify:
        qualify(manifest)
        return
    assert json.loads((HERE / 'qualification.json').read_text())['status'] == 'real_separate_cohort_path_qualified'
    cases, arms = manifest['cases'], manifest['arms']
    count = len(cases)
    assert count == 358
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=initialize, initargs=(manifest,)) as pool:
        receipts = list(pool.map(build_case, cases, chunksize=1))
    assert len(receipts) == count and sum(r['frames'] for r in receipts) == 78875
    for arm in arms:
        archive = HERE / 'evaluation' / (arm + '.npz')
        if archive.exists():
            with zipfile.ZipFile(archive) as stored:
                assert len(stored.namelist()) == count * 3
            continue
        temporary = archive.with_suffix('.partial.npz')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as stored:
            for case, receipt in zip(cases, receipts, strict=True):
                path = HERE / 'shots' / (case['stem'] + '.npz')
                assert digest(path) == receipt['sha256']
                with np.load(path, allow_pickle=False) as values:
                    for field in ('psirz', 'q95', 'betaN'):
                        with stored.open(f"shot_{case['index']:04d}_{field}.npy", 'w', force_zip64=True) as stream:
                            np.lib.format.write_array(stream, values[arm + '_' + field], allow_pickle=False)
        with zipfile.ZipFile(temporary) as stored:
            assert len(stored.namelist()) == count * 3
        temporary.replace(archive)
    write_phase('all358_three_arm_prediction_archives_complete')
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        statistics = list(pool.map(previous.reference_case, [(Path(c['path']), HERE / 'references') for c in cases]))
    stats = {key: sum(x[key] for x in statistics) for key in statistics[0]}
    if not (HERE / 'reference_statistics.json').exists():
        write_json(HERE / 'reference_statistics.json', previous.serial(statistics))
    means = (stats['psi_sum'] / stats['psi_n'], stats['scal_sum'] / np.maximum(stats['scal_n'], 1),
             stats['cons_sum'] / np.maximum(stats['cons_n'], 1))
    write_phase('all358_official_references_complete_scoring_three_arms')
    parts, results = {}, {}
    score = previous.scorer()
    for arm in arms:
        path = HERE / 'evaluation' / (arm + '_parts.json')
        if path.exists():
            parts[arm] = cached_parts(path)
        else:
            with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
                numbered = list(pool.map(scored_case, [(c, arm, means) for c in cases]))
            parts[arm] = [row for _, row in sorted(numbered)]
            assert len(parts[arm]) == count
            write_json(path, previous.serial(parts[arm]))
        acc = score.Accum()
        for row in parts[arm]:
            acc.add(row)
        acc.psi_sign = 1.
        assert acc.ss_res_psi[1.] <= acc.ss_res_psi[-1.]
        results[arm] = score.finalize_machine(acc, stats)
        results[arm]['S_unrounded'] = previous.unrounded_score(parts[arm], statistics, np.ones(count))
        assert abs(results[arm]['S_unrounded'] - results[arm]['S']) < 5.01e-7
    protocol = json.loads((HERE / 'protocol.json').read_text())
    gate = protocol['gate']
    rng = np.random.default_rng(gate['seed'])
    draws = [np.bincount(rng.integers(0, count, count), minlength=count) for _ in range(gate['bootstrap_draws'])]
    comparisons = {}
    for control in gate['controls_required']:
        delta = results['candidate001']['S_unrounded'] - results[control]['S_unrounded']
        boots = [previous.unrounded_score(parts['candidate001'], statistics, draw) - previous.unrounded_score(parts[control], statistics, draw) for draw in draws]
        probability = float(np.mean(np.asarray(boots) > 0))
        failures = sum(r['lcfs_fails'] for r in parts['candidate001']) <= sum(r['lcfs_fails'] for r in parts[control])
        comparisons[control] = {'delta_S': delta, 'ci95': np.percentile(boots, [2.5, 97.5]).tolist(),
            'P_positive': probability, 'no_increased_lcfs_failures': failures,
            'passes': bool(delta >= gate['minimum_delta_S'] and probability >= gate['minimum_P_positive'] and failures)}
    write_json(HERE / 'evaluation/summary.json', {'status': 'separate_cohort_confirmation_complete',
        'shots': count, 'frames': 78875, 'results': results, 'comparisons': comparisons,
        'confirmation_passes': all(v['passes'] for v in comparisons.values()),
        'cohort_historically_used_disclosed': True, 'fresh_project_virgin_external_test': False,
        'neural_or_tree_fits_performed': False, 'public_upload_or_submission': False})


def write_phase(phase):
    write_json(HERE / (phase + '.json'), {'status': phase}) if not (HERE / (phase + '.json')).exists() else None
    print(json.dumps({'confirmation_phase': phase}), flush=True)


if __name__ == '__main__':
    main()
