# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D26 finished OOF comparisons and physical diagnostics
"""Finish the frozen D26 comparisons without making a public candidate."""

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import pickle
import struct
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('$LAB')
REPO = Path('$HOME/fusion-equilibrium-challenge')
SEAT = ROOT / 'work/first_agent'
STORAGE = Path('$FEC_RUNS')
PKG = STORAGE / 'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
WIDE_TABLE = STORAGE / 'ml350/runs/tables28_subframe_20261006/table_training.npz'
ORIGINAL_TABLE = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_training.npz'
DATA = Path('$FEC_DATA/data/diii_d_train')
BASIS = PKG / 'models/flux_basis.npz'
ARMS = ('p45', 's45', 'p64')
WEIGHTS = {'li_scored': .26, 'r_axis': .14, 'kappa': .09, 'x_lower_r': .16}
FLAGS = ['--kept-coordinates', '45', '--smooth-from', '30', '--smooth-axis', '0',
         '--shape', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li', '--x-points',
         '--reshape-after-x-points', '--geometric-share', '.5', '--li-share', '.6',
         '--edit-margin', '1', '--contour', '--contour-first', '--contour-share', '.5',
         '--contour-rounds', '1', '--scalar-share', 'kappa=.5', 'volume=.5',
         'tri_top=.25', 'tri_bot=.25']
STATE = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def mapped_features(path):
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo('features.npy')
        assert member.compress_type == zipfile.ZIP_STORED
    with path.open('rb') as stream:
        stream.seek(member.header_offset)
        header = stream.read(30)
        name_size, extra_size = struct.unpack('<HH', header[26:30])
        stream.seek(name_size + extra_size, 1)
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
    assert not fortran and dtype == np.dtype('float32')
    return np.memmap(path, mode='r', offset=offset, shape=shape, dtype=dtype)


def entrypoint():
    for path in (PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter', PKG / 'entrypoints'):
        sys.path.insert(0, str(path))
    import predict_diii_d
    return predict_diii_d


def scorer():
    sys.path.insert(0, str(REPO / 'fec_harness'))
    import run_parallel_score
    return run_parallel_score


def population():
    splits = json.loads((SEAT / 'kfold/splits.json').read_text())
    reserved = set(json.loads((SEAT / 'kfold/reserved_for_evaluation.json').read_text())['stems'])
    with np.load(ORIGINAL_TABLE, allow_pickle=False) as stored:
        original = stored['stems'].astype(str)
        counts = np.bincount(stored['shot_of_row'], minlength=len(original))
    with np.load(WIDE_TABLE, allow_pickle=False) as stored:
        wide = stored['stems'].astype(str)
        all_counts = np.bincount(stored['shot_of_row'], minlength=len(wide))
        assert str(stored['feature_layout']) == 'discharge'
    assert set(original) == set(splits['training'])
    assert np.array_equal(original, wide[:len(original)])
    assert np.array_equal(counts, all_counts[:len(original)])
    bounds = np.r_[0, np.cumsum(counts)]
    spans = {stem: (int(bounds[i]), int(bounds[i + 1])) for i, stem in enumerate(original)}
    cases = []
    for fold in range(5):
        held = sorted((s for s in original if s not in reserved and
                       int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == fold),
                      key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:80]
        assert len(held) == 80
        for stem in held:
            cases.append({'index': len(cases), 'fold': fold, 'stem': stem,
                          'span': spans[stem], 'path': str(DATA / f'{stem}.parquet')})
    return cases, int(bounds[-1])


def qualified_models():
    receipt = json.loads((HERE / 'geometry_readout_receipt.json').read_text())
    assert receipt['status'] == '30_fold_excluded_geometry_readouts_completed'
    assert len(receipt['results']) == 30
    models = []
    for item in receipt['results']:
        assert item['heldfold_reserved_excluded']
        assert item['CPU_GPU_moment_relative_max'] < 1e-9
        path = HERE / f"readouts_f{item['fold']}" / Path(item['input']).parent.name / 'network.npz'
        assert digest(path) == item['network_sha256']
        models.append({'fold': item['fold'], 'role': item['role'], 'path': str(path),
                       'sha256': item['network_sha256']})
    for fold in range(5):
        for role in ('hx_shared', 'hc_p', 'hc_s'):
            assert sum(x['fold'] == fold and x['role'] == role for x in models) == 2
    return models


def initialize(manifest, out):
    STATE['manifest'], STATE['out'] = manifest, Path(out)
    pd = entrypoint()
    score = scorer()
    STATE.update(pd=pd, score=score, matrix=mapped_features(WIDE_TABLE), fold=None,
                 raw_together=pd.predict_together, raw_read=pd.read_row,
                 raw_layout=pd.layout_features, stage_active=False, stage_values={})
    STATE['basis'] = pd.FluxBasis.load(BASIS).truncated(64)
    args = pd._parser().parse_args(['--data-root', str(DATA.parent), '--network',
                                  manifest['models'][0]['path'], '--basis', str(BASIS),
                                  '--out', str(out), '--split', 'diii_d_train', *FLAGS])
    STATE['finish'] = pd._finish_from(args)
    STATE['grid'] = score._grid()
    original_decode = pd.FluxBasis.decode

    def decode(basis, coordinates):
        result = original_decode(basis, coordinates)
        stage('decoded_after_smoothing', result)
        return result

    pd.FluxBasis.decode = decode
    for name, label in (('moved_onto', 'axis'), ('_contour_bent', 'contour'),
                        ('edited_x_points', 'x_points'), ('_kept_within_margin', 'final')):
        original = getattr(pd, name)

        def wrapped(*args, _function=original, _label=label, **kwargs):
            result = _function(*args, **kwargs)
            stage(_label, result)
            return result

        setattr(pd, name, wrapped)
    original_edit = pd.edited

    def edited(*args, **kwargs):
        result = original_edit(*args, **kwargs)
        if STATE['stage_active']:
            STATE['edit_count'] += 1
            stage('shape' if STATE['edit_count'] == 1 else 'reshape', result[0])
        return result

    pd.edited = edited


def stage(name, maps):
    if not STATE['stage_active']:
        return
    score = STATE['score']
    radius, height, coarse, mask_f = STATE['grid']
    reference = STATE['reference']
    contours, rgeo = reference[0], reference[3]
    values = np.full(len(maps), np.nan)
    failures = np.zeros(len(maps), dtype=bool)
    for i, (flux, truth) in enumerate(zip(maps, contours, strict=True)):
        if truth is None:
            continue
        predicted, _ = score.local_score.extract_lcfs_with_sign(
            np.asarray(flux, dtype=np.float32).astype(np.float64), radius, height,
            'DIII-D', coarse, mask_f, n_iter=score.local_score.N_ITER, axis_sign=-1.0)
        if predicted is None:
            values[i], failures[i] = 1., True
        else:
            distance = score.local_score.symmetric_hausdorff(predicted, np.asarray(truth, dtype=float))
            values[i] = min(1., distance / rgeo if rgeo > 0 else 1.)
    STATE['stage_values'][name] = (values, failures)


def load_fold(fold):
    if STATE['fold'] == fold:
        return
    pd, manifest = STATE['pd'], STATE['manifest']
    groups = {}
    for role in ('hx_shared', 'hc_p', 'hc_s'):
        groups[role] = [pd.Network.load(x['path']) for x in manifest['models']
                        if x['fold'] == fold and x['role'] == role]
    assert all(n.coordinates == 64 and n.feature_mean.size == 582 and
               n.feature_layout == 'discharge' for group in groups.values() for n in group)
    path = SEAT / f'd9_gbm_heads/kaggle_out/gbm_oof_fold{fold}.npz'
    assert digest(path) == manifest['trees'][fold]['sha256']
    with np.load(path, allow_pickle=False) as stored:
        rows, names = stored['rows'], stored['targets'].astype(str).tolist()
        tree = np.full((manifest['original_rows'], len(WEIGHTS)), np.nan, dtype=np.float32)
        tree[rows] = stored['pred'][:, [names.index(n) for n in WEIGHTS]]
    STATE.update(fold=fold, groups=groups, tree=tree)


def build_case(case):
    load_fold(case['fold'])
    pd, out, matrix = STATE['pd'], STATE['out'], STATE['matrix']
    path = Path(case['path'])
    row = STATE['raw_read'](path)
    shot = pd.aligned_shot(row, case['stem'], np.asarray(row['efit_times'], dtype=float))
    start, stop = case['span']
    assert shot.n_frames == stop - start
    features = np.asarray(matrix[start:stop])
    assert features.shape == (shot.n_frames, 582) and np.isfinite(features).all()
    if case['index'] == 0:
        live = STATE['raw_layout'](row, shot, {'discharge'})['discharge']
        assert live.shape == (shot.n_frames, 367)
        delta = float(np.max(np.abs(live.astype(np.float32) - features[:, :367])))
        assert np.allclose(live.astype(np.float32), features[:, :367], rtol=2e-6, atol=2e-6), delta
        write_json(out / 'feature_parity.json', {'stem': case['stem'], 'base_width': 367,
                    'max_abs_difference': delta, 'frame_times_sha256': hashlib.sha256(shot.times_ms.tobytes()).hexdigest()})
    answers = {}
    for arm, role in (('p', 'hc_p'), ('s', 'hc_s')):
        networks = STATE['groups']['hx_shared'] + STATE['groups'][role]
        coordinates, labels = STATE['raw_together'](networks, {'discharge': features})
        for j, (name, weight) in enumerate(WEIGHTS.items()):
            tree = STATE['tree'][start:stop, j].astype(float)
            assert np.isfinite(tree).all() and tree.shape == labels[name].shape
            labels[name] += weight * (tree - labels[name])
        answers[arm] = coordinates, labels
    STATE['reference'] = pickle.loads((out / 'references' / f"{case['stem']}.pkl").read_bytes())
    STATE['stage_values'], STATE['edit_count'] = {}, 0
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    predictions = []
    for arm in ARMS:
        networks = STATE['groups']['hx_shared'] + STATE['groups']['hc_p' if arm[0] == 'p' else 'hc_s']
        coordinates, labels = answers[arm[0]]
        pd.predict_together = lambda *_args: (coordinates.copy(), {n: v.copy() for n, v in labels.items()})
        STATE['stage_active'] = arm == 'p45'
        finish = dataclasses.replace(STATE['finish'], kept_coordinates=64 if arm == 'p64' else 45)
        prediction = pd.predict_shot(path, networks, STATE['basis'], finish)
        assert all(np.isfinite(x).all() for x in prediction)
        predictions.append(prediction)
    STATE['stage_active'] = False
    names = list(STATE['stage_values'])
    np.savez_compressed(out / 'diagnostics' / f"{case['stem']}.npz",
        stage_names=np.asarray(names), stage_distance=np.stack([STATE['stage_values'][n][0] for n in names]),
        stage_failures=np.stack([STATE['stage_values'][n][1] for n in names]),
        times_ms=shot.times_ms, ip_A=shot.plasma_current,
        ip_rate_A_s=np.gradient(shot.plasma_current, shot.times_ms / 1000.) if shot.n_frames > 1 else np.zeros(1),
        electron_pressure_mean_Pa=pressure_summary(shot)[0], electron_pressure_peak_Pa=pressure_summary(shot)[1],
        pressure_channel_coverage=pressure_summary(shot)[2])
    print(json.dumps({'built': case['index'], 'fold': case['fold'], 'stem': case['stem'],
                      'frames': shot.n_frames, 'stage_count': len(names)}), flush=True)
    return case['index'], predictions


def pressure_summary(shot):
    valid = np.isfinite(shot.thomson.ne) & np.isfinite(shot.thomson.te) & (shot.thomson.ne > 0) & (shot.thomson.te > 0)
    pressure = np.where(valid, shot.thomson.ne * shot.thomson.te * 1.602176634e-19, 0.)
    count = valid.sum(axis=1)
    mean = np.where(count > 0, pressure.sum(axis=1) / np.maximum(count, 1), np.nan)
    peak = np.where(count > 0, pressure.max(axis=1), np.nan)
    return mean, peak, count / max(1, valid.shape[1])


def bounded_results(pool, function, cases):
    iterator = iter(cases)
    pending = {}
    for case in iterator:
        pending[pool.submit(function, case)] = case
        if len(pending) == 6:
            break
    while pending:
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            del pending[future]
            yield future.result()
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(function, case)] = case


def reference_case(task):
    return scorer()._reference(task)


def scored_case(task):
    case, arm, out, means = task
    score = scorer()
    path, out = Path(case['path']), Path(out)
    truth = score._truth(path)
    reference = pickle.loads((out / 'references' / f"{case['stem']}.pkl").read_bytes())
    prediction = score._prediction(out / f'{arm}.npz', case['index'])
    radius, height, coarse, mask_f = score._grid()
    part = score.local_score.score_shot(truth, reference, prediction, radius, height,
                                       coarse, mask_f, 1., means)
    if arm == 'p45':
        np.savez_compressed(out / 'diagnostics' / f"{case['stem']}_errors.npz",
            flux_mse=np.mean((truth['psi'].astype(float) - prediction['psirz'])**2, axis=(1, 2)),
            q95_squared_error=(truth['q95'] - prediction['q95'])**2,
            betaN_squared_error=(truth['betaN'] - prediction['betaN'])**2)
    print(json.dumps({'scored': arm, 'index': case['index']}), flush=True)
    return case['index'], part


def unrounded_score(parts, statistics, weights):
    def total(key):
        return sum(w * np.asarray(x[key]) for w, x in zip(weights, statistics, strict=True))
    flux_total = total('psi_sumsq') - total('psi_sum')**2 / total('psi_n')
    flux_residual = sum(w * x['ss_res_psi'][1.] for w, x in zip(weights, parts, strict=True))
    flux = 1. - flux_residual / flux_total
    terms = []
    for prefix, residual_key in (('scal', 'ss_res_scal'), ('cons', 'ss_res_cons')):
        count = total(prefix + '_n')
        denom = total(prefix + '_sumsq') - total(prefix + '_sum')**2 / np.maximum(count, 1)
        residual = sum(w * np.asarray(x[residual_key]) for w, x in zip(weights, parts, strict=True))
        r2 = 1. - residual / np.where(denom > 0, denom, 1.)
        terms.append(float(np.mean(r2)) if prefix == 'scal' else float(np.mean(np.maximum(0., r2[denom > 0]))))
    valid = [(w, x['shot_dlcfs']) for w, x in zip(weights, parts, strict=True) if x['shot_dlcfs'] is not None]
    distance = sum(w * d for w, d in valid) / sum(w for w, _ in valid) if valid else 1.
    return .55 * max(0., float(flux)) + .15 * max(0., terms[0]) + .10 * (1. - min(1., distance)) + .20 * terms[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(exist_ok=False)
    for name in ('references', 'diagnostics'):
        (args.out / name).mkdir()
    cases, rows = population()
    if args.smoke:
        cases = cases[:1]
    models = qualified_models()
    manifest = {'status': 'frozen_execution_manifest', 'smoke_only': args.smoke,
        'cases': cases, 'models': models, 'original_rows': rows, 'arms': ARMS,
        'basis_sha256': digest(BASIS), 'builder_sha256': digest(PKG / 'entrypoints/predict_diii_d.py'),
        'script_sha256': digest(__file__), 'flags': FLAGS, 'tree_weights': WEIGHTS,
        'trees': [{'fold': k, 'sha256': digest(SEAT / f'd9_gbm_heads/kaggle_out/gbm_oof_fold{k}.npz')} for k in range(5)],
        'gate_protocol_sha256': digest(HERE / 'amendment4.json'),
        'diagnostic_stage_sign': 1, 'diagnostic_pressure': 'electron-only Thomson channel summaries; not vessel or total pressure',
        'bootstrap': 'paired shots; pooled reference moments resampled; failed-value imputation uses frozen population means'}
    write_json(args.out / 'manifest.json', manifest)
    print(json.dumps({'phase': 'reference', 'shots': len(cases), 'frames': sum(c['span'][1] - c['span'][0] for c in cases)}), flush=True)
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        statistics = list(pool.map(reference_case, [(Path(c['path']), args.out / 'references') for c in cases]))
    stats = {key: sum(one[key] for one in statistics) for key in statistics[0]}
    write_json(args.out / 'reference_statistics.json', serial(statistics))
    means = (stats['psi_sum'] / stats['psi_n'], stats['scal_sum'] / np.maximum(stats['scal_n'], 1),
             stats['cons_sum'] / np.maximum(stats['cons_n'], 1))
    archives = {arm: zipfile.ZipFile(args.out / f'{arm}.npz', 'x', compression=zipfile.ZIP_STORED, allowZip64=True) for arm in ARMS}
    print(json.dumps({'phase': 'build', 'shots': len(cases)}), flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 initializer=initialize, initargs=(manifest, args.out)) as pool:
            for index, predictions in bounded_results(pool, build_case, cases):
                for arm, prediction in zip(ARMS, predictions, strict=True):
                    for name, values in zip(('psirz', 'q95', 'betaN'), prediction, strict=True):
                        with archives[arm].open(f'shot_{index:04d}_{name}.npy', 'w', force_zip64=True) as stream:
                            np.lib.format.write_array(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
    finally:
        for archive in archives.values():
            archive.close()
    write_json(args.out / 'build_receipt.json', {'status': 'all_three_archives_built', 'shots': len(cases),
        'archives': {arm: {'sha256': digest(args.out / f'{arm}.npz'), 'bytes': (args.out / f'{arm}.npz').stat().st_size} for arm in ARMS}})
    score = scorer()
    parts, results = {}, {}
    for arm in ARMS:
        print(json.dumps({'phase': 'score', 'arm': arm}), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            numbered = list(pool.map(scored_case, [(c, arm, args.out, means) for c in cases]))
        part = [x[1] for x in sorted(numbered)]
        acc = score.Accum()
        acc.psi_sign = 1.
        for item in part:
            acc.add(item)
        assert acc.ss_res_psi[1.] <= acc.ss_res_psi[-1.], 'stage analysis assumes verified canonical prediction sign'
        assert acc.n_shots_missing == 0 and acc.psi_nonfinite == 0
        result = score.finalize_machine(acc, stats)
        result['S_unrounded'] = unrounded_score(part, statistics, np.ones(len(cases)))
        assert abs(result['S'] - result['S_unrounded']) <= 5.01e-7
        parts[arm], results[arm] = part, result
        write_json(args.out / f'{arm}_parts.json', serial(part))
        write_json(args.out / f'{arm}_score.json', serial(result))
        print(json.dumps({'arm': arm, 'result': result}), flush=True)
    comparisons = {}
    for arm in ('s45', 'p64'):
        rng = np.random.default_rng(990)
        boots = []
        for _ in range(2000):
            weights = np.bincount(rng.integers(0, len(cases), len(cases)), minlength=len(cases))
            boots.append(unrounded_score(parts[arm], statistics, weights) - unrounded_score(parts['p45'], statistics, weights))
        gain = results[arm]['S_unrounded'] - results['p45']['S_unrounded']
        failures = sum(x['lcfs_fails'] for x in parts[arm])
        frames = sum(x['lcfs_frames'] for x in parts[arm])
        base_failures = sum(x['lcfs_fails'] for x in parts['p45'])
        base_frames = sum(x['lcfs_frames'] for x in parts['p45'])
        p = float(np.mean(np.asarray(boots) > 0))
        guard = failures * base_frames <= base_failures * frames
        comparisons[arm] = {'delta_S': gain, 'ci95': np.percentile(boots, [2.5, 97.5]).tolist(),
            'P_positive': p, 'no_increased_lcfs_failures': guard,
            'admission_passes': bool(not args.smoke and len(cases) == 400 and gain >= 3e-5 and p >= .975 and guard)}
    write_json(args.out / 'comparison.json', {'status': 'smoke_complete' if args.smoke else 'finished_oof_gate_complete',
        'shots': len(cases), 'results': results, 'comparisons': comparisons, 'seconds': time.time() - started,
        'public_candidate_built': False, 'public_upload_or_submission': False})
    print(json.dumps({'complete': comparisons, 'seconds': time.time() - started}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
