# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — deferred common-score physics campaign evaluation
"""Evaluate every declared comparison only after all fitting tasks terminate."""

import json
import multiprocessing
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from campaign_common import (ARMS, D26, HERE, PKG, SEAT, base_maps, digest, extra_maps,
                             feature_block, write_json)
from physics_features import read_inputs, full_inputs
from fit_pressure_factor import predict_factor
from fec_adapter.cross_machine import map_scalars
from fec_adapter.cross_machine.thomson_isoflux import green
from fec_adapter.orientation import flux_orientation

sys.path.insert(0, str(D26))
import finish_and_score as previous

VARIANTS = ('control45', 'control64', 'pressure45', 'memory45', 'residual45', 'gradient64')
STATE = {}


def initialize(manifest, out):
    previous.initialize(manifest, out)
    STATE.update(base=base_maps(), extra=extra_maps(), pd=previous.STATE['pd'],
                 basis=previous.STATE['basis'], out=Path(out), manifest=manifest, fold=None)
    pd = STATE['pd']
    original_decode = pd.FluxBasis.decode

    def decode(basis, coordinates):
        maps = original_decode(basis, coordinates)
        if STATE.get('arm') == 'residual':
            maps = maps + .5 * STATE['coil_flux']
        return maps

    pd.FluxBasis.decode = decode


def load_fold(fold):
    if STATE['fold'] == fold:
        return
    models = STATE['manifest']['models']
    pd = STATE['pd']
    STATE['networks'] = {}
    for arm in STATE['manifest']['complete_arms']:
        paths = [m['path'] for m in models if m['fold'] == fold and m['role'] in ('hx_shared', arm)]
        assert len(paths) == 4
        STATE['networks'][arm] = [pd.Network.load(path) for path in paths]
    path = SEAT / f'd9_gbm_heads/kaggle_out/gbm_oof_fold{fold}.npz'
    with np.load(path, allow_pickle=False) as stored:
        names = stored['targets'].astype(str).tolist()
        tree = np.full((STATE['manifest']['original_rows'], 4), np.nan, dtype=np.float32)
        tree[stored['rows']] = stored['pred'][:, [names.index(n) for n in previous.WEIGHTS]]
    STATE.update(fold=fold, tree=tree)


def build_case(case):
    load_fold(case['fold'])
    pd, basis = STATE['pd'], STATE['basis']
    path = Path(case['path'])
    row = previous.STATE['raw_read'](path)
    shot = pd.aligned_shot(row, case['stem'], np.asarray(row['efit_times'], dtype=float))
    start, stop = case['span']
    assert shot.n_frames == stop - start
    rows = np.arange(start, stop)
    rr, zz = np.meshgrid(shot.r_grid, shot.z_grid)
    units = np.stack([green(rr, zz, r, z).ravel() for r, z in zip(shot.coils.r, shot.coils.z, strict=True)])
    STATE['coil_flux'] = (flux_orientation(shot.plasma_current) * shot.coil_current @ units).reshape(-1, 65, 65)
    pd.read_row = lambda _path: row
    previous.STATE['stage_active'] = False
    results = []
    for variant in STATE['manifest']['variants']:
        arm = variant[:-2]
        keep = int(variant[-2:])
        features = feature_block(STATE['base']['training'], STATE['extra']['training'], rows, arm)
        networks = STATE['networks'][arm]
        coordinates, labels = previous.STATE['raw_together'](networks, {'discharge': features})
        for j, (name, weight) in enumerate(previous.WEIGHTS.items()):
            head = STATE['tree'][start:stop, j].astype(float)
            assert np.isfinite(head).all()
            labels[name] += weight * (head - labels[name])
        pd.layout_features = lambda *_args: {'discharge': features}
        pd.predict_together = lambda *_args: (coordinates.copy(), {n: v.copy() for n, v in labels.items()})
        STATE['arm'] = arm
        from dataclasses import replace
        finish = replace(previous.STATE['finish'], kept_coordinates=keep)
        prediction = pd.predict_shot(path, networks, basis, finish)
        assert all(np.isfinite(value).all() for value in prediction)
        results.append(prediction)
    print(json.dumps({'physics_archive_case': case['index'], 'fold': case['fold']}), flush=True)
    return case['index'], results


def pressure_case(task):
    case, out = task
    out = Path(out)
    pd = previous.entrypoint()
    row = read_inputs(Path(case['path']))
    shot = pd.aligned_shot(row, case['stem'], np.asarray(row['efit_times'], dtype=float))
    from shape_edit_fixed_axis_volume import Reader
    reader = Reader('DIII-D', backend='native')
    old = previous.scorer()._prediction(D26 / 'full_score_v2/p45.npz', case['index'])
    beta_e = np.full(shot.n_frames, np.nan)
    for i, flux in enumerate(old['psirz']):
        reading = reader.read(flux)
        if reading.contour is None:
            continue
        axis = reading.axis
        axis_flux = float(map_scalars._at(flux, shot.r_grid, shot.z_grid,
                                         np.array([axis[0]]), np.array([axis[1]]))[0])
        edge_flux = float(np.mean(map_scalars._at(flux, shot.r_grid, shot.z_grid,
                                                reading.contour[:, 0], reading.contour[:, 1])))
        if abs(edge_flux - axis_flux) < 1e-12:
            continue
        beta_e[i] = map_scalars.normalised_beta(flux, shot.r_grid, shot.z_grid, axis,
            axis_flux, edge_flux, shot.thomson.r, shot.thomson.z, shot.thomson.te[i], shot.thomson.ne[i],
            float(shot.plasma_current[i]), map_scalars.vacuum_f('DIII-D', float(shot.tf_current[i]))) / map_scalars.PRESSURE_FACTOR
    start, stop = case['span']
    inputs = np.load(HERE / 'prepared/training/factor_features.npy', mmap_mode='r', allow_pickle=False)[start:stop]
    factor = predict_factor(HERE / f"pressure_factors/factor_fold{case['fold']}.npz", inputs)
    valid = np.isfinite(beta_e) & (beta_e > 0)
    fixed = np.where(valid, beta_e * 2.25, old['betaN'])
    adaptive = np.where(valid, beta_e * factor, old['betaN'])
    return case['index'], old['psirz'].astype(np.float32), old['q95'].astype(np.float32), fixed, adaptive


def scored_case(task):
    case, arm, out, means = task
    score = previous.scorer()
    path, out = Path(case['path']), Path(out)
    import pickle
    reference = pickle.loads((out / 'references' / f"{case['stem']}.pkl").read_bytes())
    truth = score._truth(path)
    prediction = score._prediction(out / f'{arm}.npz', case['index'])
    radius, height, coarse, mask_f = score._grid()
    part = score.local_score.score_shot(truth, reference, prediction, radius, height,
                                       coarse, mask_f, 1., means)
    return case['index'], part


def store_prediction(archive, index, prediction):
    for name, values in zip(('psirz', 'q95', 'betaN'), prediction, strict=True):
        assert np.isfinite(values).all()
        with archive.open(f'shot_{index:04d}_{name}.npy', 'w', force_zip64=True) as stream:
            np.lib.format.write_array(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)


def main():
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    factors = json.loads((HERE / 'pressure_factors/receipt.json').read_text())
    readouts = json.loads((HERE / 'readouts/receipt.json').read_text())
    assert neural['status'] == 'all_50_neural_tasks_terminal' and len(neural['tasks']) == 50
    assert factors['status'] == 'all_five_factor_fits_complete'
    assert readouts['status'] == 'all_readout_tasks_terminal' and len(readouts['tasks']) == 50
    out = HERE / 'evaluation'
    out.mkdir(exist_ok=False)
    (out / 'references').symlink_to(D26 / 'full_score_v2/references', target_is_directory=True)
    cases = json.loads((D26 / 'full_score_v2/manifest.json').read_text())['cases']
    assert len(cases) == 400
    models = [m for m in previous.qualified_models() if m['role'] == 'hx_shared']
    complete_arms, incomplete = [], {}
    for arm in ARMS:
        tasks = [t for t in readouts['tasks'] if t['name'].startswith(arm + '_seed')]
        if len(tasks) != 10 or any(t['status'] != 'geometry_readout_complete' for t in tasks):
            incomplete[arm] = 'missing successful fivefold paired readouts; no partial population substitution'
            continue
        complete_arms.append(arm)
        for task in tasks:
            path = HERE / 'readouts' / task['name'] / 'network.npz'
            assert digest(path) == task['network_sha256']
            models.append({'fold': task['fold'], 'role': arm, 'path': str(path), 'sha256': task['network_sha256']})
    variants = [v for v in VARIANTS if v[:-2] in complete_arms]
    manifest = {'cases': cases, 'models': models, 'variants': variants, 'complete_arms': complete_arms,
        'original_rows': json.loads((D26 / 'full_score_v2/manifest.json').read_text())['original_rows'],
        'all_fitting_tasks_terminal_before_evaluation': True, 'incomplete_arms': incomplete,
        'residual_reconstruction': 'Mean two full-map hx plus two coil-residual hc coefficients; after common learned-coordinate truncation/smoothing, add half the full known coil field before axis/shape/contour finish.'}
    write_json(out / 'manifest.json', manifest)
    context = multiprocessing.get_context('spawn')
    if variants:
        archives = {v: zipfile.ZipFile(out / f'{v}.npz', 'x', compression=zipfile.ZIP_STORED, allowZip64=True) for v in variants}
        try:
            with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=initialize,
                                     initargs=(manifest, out)) as pool:
                for index, predictions in previous.bounded_results(pool, build_case, cases):
                    for v, pred in zip(variants, predictions, strict=True):
                        store_prediction(archives[v], index, pred)
        finally:
            for archive in archives.values():
                archive.close()
    pressure_archives = {arm: zipfile.ZipFile(out / f'{arm}.npz', 'x', compression=zipfile.ZIP_STORED, allowZip64=True)
                         for arm in ('factor_fixed', 'factor_adaptive')}
    try:
        with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
            for index, maps, q95, fixed, adaptive in pool.map(pressure_case, [(c, out) for c in cases]):
                store_prediction(pressure_archives['factor_fixed'], index, (maps, q95, fixed))
                store_prediction(pressure_archives['factor_adaptive'], index, (maps, q95, adaptive))
    finally:
        for archive in pressure_archives.values():
            archive.close()
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        statistics = list(pool.map(previous.reference_case, [(Path(c['path']), out / 'references') for c in cases]))
    stats = {key: sum(x[key] for x in statistics) for key in statistics[0]}
    means = (stats['psi_sum'] / stats['psi_n'], stats['scal_sum'] / np.maximum(stats['scal_n'], 1),
             stats['cons_sum'] / np.maximum(stats['cons_n'], 1))
    score = previous.scorer()
    parts, results = {}, {}
    for variant in variants + ['factor_fixed', 'factor_adaptive']:
        with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
            numbered = list(pool.map(scored_case, [(c, variant, out, means) for c in cases]))
        part = [p for _, p in sorted(numbered)]
        acc = score.Accum()
        for row in part:
            acc.add(row)
        acc.psi_sign = 1.
        assert acc.ss_res_psi[1.] <= acc.ss_res_psi[-1.]
        result = score.finalize_machine(acc, stats)
        result['S_unrounded'] = previous.unrounded_score(part, statistics, np.ones(400))
        assert abs(result['S_unrounded'] - result['S']) <= 5.01e-7
        parts[variant], results[variant] = part, result
        write_json(out / f'{variant}_parts.json', previous.serial(part))
        write_json(out / f'{variant}_score.json', result)
        print(json.dumps({'scoring_arm_complete': variant, 'shots': 400}), flush=True)
    comparisons = {}
    for new, control in [('pressure45', 'control45'), ('memory45', 'control45'),
                         ('residual45', 'control45'), ('gradient64', 'control64'),
                         ('factor_adaptive', 'factor_fixed')]:
        if new not in parts or control not in parts:
            comparisons[new] = {'status': 'incomplete_no_admission', 'admission_passes': False}
            continue
        rng = np.random.default_rng(990)
        boots = []
        for _ in range(2000):
            weights = np.bincount(rng.integers(0, 400, 400), minlength=400)
            boots.append(previous.unrounded_score(parts[new], statistics, weights) -
                         previous.unrounded_score(parts[control], statistics, weights))
        delta = results[new]['S_unrounded'] - results[control]['S_unrounded']
        p = float(np.mean(np.asarray(boots) > 0))
        failures = sum(r['lcfs_fails'] for r in parts[new])
        old_failures = sum(r['lcfs_fails'] for r in parts[control])
        comparisons[new] = {'status': 'compared_after_all_fits', 'control': control,
            'delta_S': delta, 'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'P_positive': p,
            'no_increased_lcfs_failures': failures <= old_failures,
            'admission_passes': bool(delta >= 3e-5 and p >= .99 and failures <= old_failures),
            'MAST_score_inferred': False}
    write_json(out / 'summary.json', {'status': 'all_declared_comparisons_processed',
        'results': results, 'comparisons': comparisons, 'incomplete_arms': incomplete,
        'public_upload_or_submission': False, 'combined_candidate_built': False})
    print(json.dumps({'campaign_evaluation_complete': True}), flush=True)


if __name__ == '__main__':
    main()
