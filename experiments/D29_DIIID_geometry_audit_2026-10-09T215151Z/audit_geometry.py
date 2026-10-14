# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D geometry stage attribution
"""Compare learned heads and all deployment stages on unchanged predictions."""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D26 = HERE.parent / 'D26_topology_aux_2026-10-08T213532Z'
sys.path.insert(0, str(D26))
import finish_and_score as original

NAMES = ('R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li')
HEADS = ('r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored')
STATE = {}


def write_json(path: Path, value: dict) -> None:
    """Create immutable evidence without replacing an earlier result."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def population() -> tuple[dict, list, dict]:
    """Read the original manifest and complete reference eligibility masks."""
    manifest = json.loads((D26 / 'full_score_v2/manifest.json').read_text())
    parts = json.loads((D26 / 'full_score_v2/p45_parts.json').read_text())
    total, square, count = np.zeros(7), np.zeros(7), np.zeros(7, dtype=np.int64)
    for item in manifest['cases']:
        ref = pickle.loads((D26 / 'full_score_v2/references' / (item['stem'] + '.pkl')).read_bytes())
        values, valid = ref[1], ref[2]
        for j in range(7):
            selected = values[valid[:, j], j]
            total[j] += selected.sum()
            square[j] += (selected**2).sum()
            count[j] += len(selected)
    assert np.all(count > 0)
    return manifest, parts, {'mean': (total / count).tolist(),
                            'denominator': (square - total**2 / count).tolist(),
                            'count': count.tolist()}


def initialize(manifest: dict, statistics: dict, parity: bool) -> None:
    """Bind the original predictor and native scalar reader once per worker."""
    original.initialize(manifest, HERE)
    sys.path.insert(0, str(original.PKG / 'entrypoints'))
    from shape_edit_fixed_axis_volume import Reader
    STATE.update(reader=Reader('DIII-D', backend='native'), mean=np.asarray(statistics['mean']),
                 archive=zipfile.ZipFile(D26 / 'full_score_v2/p45.npz'), parity=parity)
    original.stage = capture


def capture(name: str, maps: np.ndarray) -> None:
    """Measure archived-precision maps against the scorer's reference masks."""
    if not original.STATE['stage_active']:
        return
    maps = np.asarray(maps, dtype=np.float32).astype(float)
    rows = []
    for frame in maps:
        reading = STATE['reader'].read(frame)
        rows.append([reading.values[key] for key in NAMES])
    values = np.asarray(rows, dtype=float)
    truth, valid = STATE['reference'][1:3]
    finite = np.isfinite(values)
    safe = np.where(finite, values, STATE['mean'])
    errors = np.where(valid, (safe - truth)**2, 0.)
    STATE['stages'][name] = {'SSE_geometry': np.sum(errors, axis=0).tolist(),
                             'nonfinite_geometry': np.sum(valid & ~finite, axis=0).tolist()}
    STATE['events'].append(name)
    if STATE['parity']:
        radius, height, coarse, mask = original.STATE['grid']
        public = original.STATE['score'].local_score
        checks = [0, len(maps) // 2, len(maps) - 1]
        for index in checks:
            contour, _ = public.extract_lcfs_with_sign(maps[index], radius, height, 'DIII-D', coarse, mask,
                                                      n_iter=public.N_ITER, axis_sign=-1.)
            expected = public.derive_frame(maps[index], radius, height, 'DIII-D', coarse, mask,
                                           contour=contour, with_li=True, axis_sign=-1.)
            measured = np.asarray([expected[key] for key in NAMES])
            assert np.allclose(values[index], measured, rtol=1e-10, atol=1e-10, equal_nan=True), (name, index)


def case(item: dict) -> dict:
    """Reproduce the four-member p45 baseline and measure all seven stages."""
    file = HERE / 'shots' / (item['stem'] + '.json')
    if file.exists():
        record = json.loads(file.read_text())
        assert record['final_archive_bitwise_parity']
        return record
    original.load_fold(item['fold'])
    pd = original.STATE['pd']
    path = Path(item['path'])
    row = original.STATE['raw_read'](path)
    shot = pd.aligned_shot(row, item['stem'], np.asarray(row['efit_times'], dtype=float))
    a, b = item['span']
    features = np.asarray(original.STATE['matrix'][a:b])
    networks = original.STATE['groups']['hx_shared'] + original.STATE['groups']['hc_p']
    coordinates, labels = original.STATE['raw_together'](networks, {'discharge': features})
    for j, (name, weight) in enumerate(original.WEIGHTS.items()):
        tree = original.STATE['tree'][a:b, j].astype(float)
        assert np.isfinite(tree).all()
        labels[name] += weight * (tree - labels[name])
    reference = pickle.loads((D26 / 'full_score_v2/references' / (item['stem'] + '.pkl')).read_bytes())
    STATE.update(reference=reference, stages={}, events=[])
    target, valid = reference[1:3]
    head = np.column_stack([labels[name] for name in HEADS])
    assert head.shape == target.shape and np.isfinite(head).all()
    head_sse = np.sum(np.where(valid, (head - target)**2, 0.), axis=0)
    pd.read_row = lambda _path: row
    pd.layout_features = lambda *_args: {'discharge': features}
    pd.predict_together = lambda *_args: (coordinates.copy(), {key: value.copy() for key, value in labels.items()})
    original.STATE.update(stage_active=True, edit_count=0)
    finish = dataclasses.replace(original.STATE['finish'], kept_coordinates=45)
    prediction = pd.predict_shot(path, networks, original.STATE['basis'], finish)
    original.STATE['stage_active'] = False
    for name, values in zip(('psirz', 'q95', 'betaN'), prediction, strict=True):
        baseline = np.load(STATE['archive'].open(f"shot_{item['index']:04d}_{name}.npy"), allow_pickle=False)
        assert np.array_equal(np.asarray(values, dtype=np.float32), baseline), (item['index'], name)
    expected = STATE['parts'][item['index']]['ss_res_cons']
    assert np.allclose(STATE['stages']['final']['SSE_geometry'], expected, rtol=1e-8, atol=1e-10)
    assert list(STATE['stages']) == ['decoded_after_smoothing', 'axis', 'contour', 'shape', 'x_points', 'reshape', 'final']
    record = {'index': item['index'], 'stem': item['stem'], 'fold': item['fold'], 'frames': shot.n_frames,
              'final_archive_bitwise_parity': True, 'final_original_scorer_SSE_parity': True,
              'head_SSE_geometry': head_sse.tolist(), 'stages': STATE['stages'], 'stage_events': STATE['events']}
    write_json(file, record)
    print(json.dumps({'geometry_case_complete': item['index'], 'frames': shot.n_frames}), flush=True)
    return record


def worker_initialize(manifest: dict, parts: list, statistics: dict, parity: bool) -> None:
    """Retain the official baseline parts for per-case identity checks."""
    initialize(manifest, statistics, parity)
    STATE['parts'] = parts


def main() -> None:
    """Finish the complete diagnostic population before geometry attribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    manifest, parts, statistics = population()
    if args.qualify:
        worker_initialize(manifest, parts, statistics, True)
        record = case(manifest['cases'][0])
        write_json(HERE / 'qualification.json', {'status': 'real_baseline_all_stage_geometry_parity_qualified',
            'frames': record['frames'], 'all_seven_stages_three_real_frames_original_derivation_parity': True,
            'final_map_q95_betaN_bitwise_parity': True, 'final_fullshot_official_geometry_SSE_parity': True,
            'comparative_population_metrics_processed': False})
        return
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_initialize, initargs=(manifest, parts, statistics, False)) as pool:
        records = list(pool.map(case, manifest['cases'], chunksize=1))
    assert len(records) == 400 and sum(r['frames'] for r in records) == 89584
    denominator = np.asarray(statistics['denominator'])
    stage_names = list(records[0]['stages'])
    results = {}
    for stage in ['learned_heads', *stage_names]:
        key = 'head_SSE_geometry'
        residual = sum(np.asarray(r[key] if stage == 'learned_heads' else r['stages'][stage]['SSE_geometry']) for r in records)
        r2 = 1. - residual / denominator
        results[stage] = {'R2_geometry': dict(zip(NAMES, r2.tolist(), strict=True)),
                          'consistency': float(np.mean(np.maximum(0, r2))), 'SSE_geometry': residual.tolist()}
    write_json(HERE / 'summary.json', {'status': 'all_DIII_D_geometry_stages_attributed', 'shots': 400,
        'frames': 89584, 'statistics': statistics, 'results': results,
        'seconds': time.monotonic() - started, 'new_neural_training': False,
        'all_final_predictions_unchanged': True, 'public_upload_or_submission': False})


if __name__ == '__main__':
    main()
