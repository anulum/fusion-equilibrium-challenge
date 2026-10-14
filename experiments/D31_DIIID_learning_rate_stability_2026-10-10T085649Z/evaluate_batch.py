# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete DIII-D learning rate stability evaluation
"""Score complete arms against both matched and stronger existing baselines."""

import json
import multiprocessing
import pickle
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from campaign_common import ARMS, D26, D27, D30, HERE, SEAT, base_maps, digest, feature_block, write_json

sys.path.insert(0, str(D26))
import finish_and_score as previous

STATE = {}


def initialize(manifest: dict, out: Path) -> None:
    previous.initialize(manifest, out)
    STATE.update(base=base_maps(), out=out, manifest=manifest, fold=None)


def load_fold(fold: int) -> None:
    if STATE['fold'] == fold:
        return
    pd, manifest = previous.STATE['pd'], STATE['manifest']
    STATE['networks'] = {}
    for arm in manifest['complete_arms']:
        paths = [m['path'] for m in manifest['models'] if m['fold'] == fold and m['role'] in ('hx_shared', arm)]
        assert len(paths) == 4
        STATE['networks'][arm] = [pd.Network.load(path) for path in paths]
    with np.load(SEAT / f'd9_gbm_heads/kaggle_out/gbm_oof_fold{fold}.npz', allow_pickle=False) as stored:
        names = stored['targets'].astype(str).tolist()
        tree = np.full((manifest['original_rows'], 4), np.nan, dtype=np.float32)
        tree[stored['rows']] = stored['pred'][:, [names.index(n) for n in previous.WEIGHTS]]
    STATE.update(tree=tree, fold=fold)


def build_case(case: dict) -> dict:
    file = STATE['out'] / 'shots' / (case['stem'] + '.npz')
    receipt = file.with_suffix('.json')
    if receipt.exists():
        record = json.loads(receipt.read_text())
        assert digest(file) == record['sha256']
        return record
    load_fold(case['fold'])
    pd = previous.STATE['pd']
    path = Path(case['path'])
    row = previous.STATE['raw_read'](path)
    a, b = case['span']
    features = feature_block(STATE['base']['training'], {}, np.arange(a, b), 'control')
    assert len(row['efit_times']) == b - a
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    previous.STATE['stage_active'] = False
    arrays = {}
    for arm in STATE['manifest']['complete_arms']:
        networks = STATE['networks'][arm]
        coordinates, labels = previous.STATE['raw_together'](networks, {'discharge': features})
        for j, (name, weight) in enumerate(previous.WEIGHTS.items()):
            head = STATE['tree'][a:b, j].astype(float)
            assert np.isfinite(head).all()
            labels[name] += weight * (head - labels[name])
        pd.predict_together = lambda *_args: (coordinates.copy(), {k: v.copy() for k, v in labels.items()})
        prediction = pd.predict_shot(path, networks, previous.STATE['basis'],
                                    replace(previous.STATE['finish'], kept_coordinates=45))
        for field, values in zip(('psirz', 'q95', 'betaN'), prediction, strict=True):
            assert np.isfinite(values).all()
            arrays[arm + '_' + field] = np.asarray(values, dtype=np.float32)
    temporary = file.with_name(case['stem'] + '.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(file)
    record = {'index': case['index'], 'stem': case['stem'], 'frames': b - a,
              'arms': STATE['manifest']['complete_arms'], 'sha256': digest(file)}
    write_json(receipt, record)
    print(json.dumps({'learning_rate_prediction_case_complete': case['index']}), flush=True)
    return record


def cached_parts(file: Path) -> list:
    rows = json.loads(file.read_text())
    for row in rows:
        for name in ('ss_res_psi', 'shot_r2_psi'):
            row[name] = {float(k): v for k, v in row[name].items()}
    return rows


def scored_case(task: tuple) -> tuple:
    case, arm, out, means = task
    score = previous.scorer()
    truth = score._truth(Path(case['path']))
    reference = pickle.loads((D26 / 'full_score_v2/references' / (case['stem'] + '.pkl')).read_bytes())
    pred = score._prediction(out / (arm + '.npz'), case['index'])
    r, z, coarse, floating = score._grid()
    row = score.local_score.score_shot(truth, reference, pred, r, z, coarse, floating, 1., means)
    print(json.dumps({'learning_rate_score_case_complete': case['index'], 'arm': arm}), flush=True)
    return case['index'], row


def main() -> None:
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    readouts = json.loads((HERE / 'readouts/receipt.json').read_text())
    assert neural['status'] == 'all_30_neural_tasks_terminal' and len(neural['tasks']) == 30
    assert readouts['status'] == 'all_readout_tasks_terminal' and len(readouts['tasks']) == 30
    out = HERE / 'evaluation'
    (out / 'shots').mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        return
    old_manifest = json.loads((D27 / 'evaluation/manifest.json').read_text())
    cases = old_manifest['cases']
    models = [m for m in previous.qualified_models() if m['role'] == 'hx_shared']
    complete, incomplete = [], {}
    for arm in ARMS:
        tasks = [r for r in readouts['tasks'] if r['name'].startswith(arm + '_seed')]
        if len(tasks) != 10 or any(t['status'] != 'geometry_readout_complete' for t in tasks):
            incomplete[arm] = 'No complete fivefold paired ensemble; no partial population substitution'
            continue
        complete.append(arm)
        for task in tasks:
            path = HERE / 'readouts' / task['name'] / 'network.npz'
            assert digest(path) == task['network_sha256']
            models.append({'role': arm, 'fold': task['fold'], 'path': str(path), 'sha256': task['network_sha256']})
    manifest = {'cases': cases, 'models': models, 'complete_arms': complete,
                'original_rows': old_manifest['original_rows'], 'incomplete_arms': incomplete}
    if not (out / 'manifest.json').exists():
        write_json(out / 'manifest.json', manifest)
    else:
        assert json.loads((out / 'manifest.json').read_text()) == manifest
    context = multiprocessing.get_context('spawn')
    if complete:
        with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=initialize, initargs=(manifest, out)) as pool:
            records = list(pool.map(build_case, cases, chunksize=1))
        assert len(records) == 400 and sum(r['frames'] for r in records) == 89584
        for arm in complete:
            archive = out / (arm + '.npz')
            if archive.exists():
                with zipfile.ZipFile(archive) as stored:
                    assert len(stored.namelist()) == 1200
                continue
            temporary = out / (arm + '.partial.npz')
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as stored:
                for case, record in zip(cases, records, strict=True):
                    file = out / 'shots' / (case['stem'] + '.npz')
                    assert digest(file) == record['sha256']
                    with np.load(file, allow_pickle=False) as values:
                        for field in ('psirz', 'q95', 'betaN'):
                            with stored.open(f"shot_{case['index']:04d}_{field}.npy", 'w', force_zip64=True) as stream:
                                np.lib.format.write_array(stream, values[arm + '_' + field], allow_pickle=False)
            with zipfile.ZipFile(temporary) as stored:
                assert len(stored.namelist()) == 1200
            temporary.replace(archive)
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        statistics = list(pool.map(previous.reference_case, [(Path(c['path']), D26 / 'full_score_v2/references') for c in cases]))
    stats = {key: sum(x[key] for x in statistics) for key in statistics[0]}
    means = (stats['psi_sum'] / stats['psi_n'], stats['scal_sum'] / stats['scal_n'], stats['cons_sum'] / stats['cons_n'])
    parts = {'control': cached_parts(D27 / 'evaluation/control45_parts.json'),
             'stronger_p45': cached_parts(D26 / 'full_score_v2/p45_parts.json'),
             'wide_lr002': cached_parts(D30 / 'evaluation/capacity_parts.json')}
    for arm in complete:
        kept = out / (arm + '_parts.json')
        if kept.exists():
            parts[arm] = cached_parts(kept)
            continue
        with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
            numbered = list(pool.map(scored_case, [(c, arm, out, means) for c in cases]))
        parts[arm] = [r for _, r in sorted(numbered)]
        write_json(kept, previous.serial(parts[arm]))
    score = previous.scorer()
    results = {}
    for arm, rows in parts.items():
        acc = score.Accum()
        for row in rows:
            acc.add(row)
        acc.psi_sign = 1.
        assert acc.ss_res_psi[1.] <= acc.ss_res_psi[-1.]
        result = score.finalize_machine(acc, stats)
        result['S_unrounded'] = previous.unrounded_score(rows, statistics, np.ones(400))
        assert abs(result['S_unrounded'] - result['S']) < 5.01e-7
        results[arm] = result
    comparisons = {a: {'status': 'incomplete_no_admission', 'admission_passes': False} for a in incomplete}
    rng = np.random.default_rng(993)
    draws = [np.bincount(rng.integers(0, 400, 400), minlength=400) for _ in range(2000)]
    for arm in complete:
        baselines = {}
        for control in ('control', 'stronger_p45', 'wide_lr002'):
            delta = results[arm]['S_unrounded'] - results[control]['S_unrounded']
            boots = [previous.unrounded_score(parts[arm], statistics, draw) - previous.unrounded_score(parts[control], statistics, draw) for draw in draws]
            probability = float(np.mean(np.asarray(boots) > 0))
            failures = sum(r['lcfs_fails'] for r in parts[arm]) <= sum(r['lcfs_fails'] for r in parts[control])
            baselines[control] = {'delta_S': delta, 'ci95': np.percentile(boots, [2.5, 97.5]).tolist(),
                                 'P_positive': probability, 'no_increased_lcfs_failures': failures,
                                 'passes': bool(delta >= 3e-5 and probability >= .99 and failures)}
        comparisons[arm] = {'baselines': baselines, 'admission_passes': all(baselines[b]['passes'] for b in ('control', 'stronger_p45'))}
    write_json(out / 'summary.json', {'status': 'all_three_learning_rate_comparisons_processed', 'results': results,
        'comparisons': comparisons, 'incomplete_arms': incomplete, 'shots': 400, 'frames': 89584,
        'fresh_reserved_confirmation_required_for_admitted_arm': True, 'MAST_new_work': False,
        'public_upload_or_submission': False, 'combined_candidate_built': False})


if __name__ == '__main__':
    main()
