# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — confirmed production X-point data preparation
"""Derive production targets only after source-bound confirmation approval."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
GENERATOR = None


def digest(path: Path) -> str:
    """Hash an input or output with bounded memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: dict) -> None:
    """Preserve receipts without overwriting earlier attempts."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify() -> dict:
    """Require completed confirmation and reviewed, source-bound activation."""
    activation = json.loads((HERE / 'production_activation_v1.json').read_text())
    assert activation['status'] == 'all_data_static8_authorized_after_reviewed_D43_357_PASS'
    assert activation['seeds'] == list(range(500, 508)) and activation['share'] == .5
    assert activation['result_review_pass'] and activation['trainer_review_pass']
    draft = json.loads((HERE / 'preparation_draft_v1.json').read_text())
    d43 = Path(draft['D43_root'])
    terminal = json.loads((d43 / 'pipeline_terminal.json').read_text())
    assert terminal['status'] == 'fixed_static_sole357_pass_candidate_requires_result_review'
    assert terminal['protocol_sha256'] == digest(d43 / 'protocol_frozen_v1.json')
    frozen = json.loads((d43 / 'protocol_frozen_v1.json').read_text())
    assert all(activation['sources_sha256'].get(path) == expected
               for path, expected in frozen['sources_sha256'].items())
    confirmation_required = [d43 / 'protocol_frozen_v1.json', d43 / 'pipeline_terminal.json',
                             d43 / 'confirmation357/native_qualification.json',
                             d43 / 'confirmation_trees/receipt.json',
                             d43 / 'confirmation_trees/study_receipt_v1.json']
    assert all(str(path) in activation['confirmation_sources_sha256'] for path in confirmation_required)
    review = json.loads(Path(activation['result_review_source']).read_text())
    assert review['status'] == 'D43_sole357_result_review_PASS'
    assert review['analysis_sha256'] == digest(d43 / 'confirmation357/analysis.json')
    for phase, shots, frames in [('validation238', 238, 52278), ('confirmation357', 357, 76251)]:
        path = d43 / phase / 'analysis.json'
        result = json.loads(path.read_text())
        assert result['shots'] == shots and result['frames'] == frames
        assert result['identity']['fixed_primary'] == 'static' and result['identity']['share'] == .5
        assert result['identity']['protocol_sha256'] == terminal['protocol_sha256']
        assert result['decisions']['static']['gate_pass']
        assert activation['confirmation_sources_sha256'][str(path)] == digest(path)
    required = [Path(__file__), HERE / 'metadata_row_plan_v1.json', HERE / 'preparation_draft_v1.json',
                HERE / 'train_production_static_X_v1.py', Path(activation['generator_source']),
                Path(activation['result_review_source']), Path(activation['trainer_review_source'])]
    d40, d38 = Path(draft['D40_root']), Path(draft['D38_root'])
    required += [Path(table) for table in json.loads((HERE / 'metadata_row_plan_v1.json').read_text())['table_paths'].values()]
    required += [d40 / name for name in ('train_X_specialists_v3.py', 'x_training_kernel_v2.py',
                 'x_temporal_model.py', 'shared_initial_parameters.py', 'training_recipe_frozen_v1.json',
                 'original6326_data/normalization.npz', 'shared_initial_parameters/receipt.json')]
    required += [d40 / 'shared_initial_parameters' / f'seed{seed}.npz' for seed in range(500, 508)]
    required += [d38 / name for name in ('campaign_common.py', 'training_data.py',
                 'candidate_state.py', 'temporal_model.py', 'prepare_temporal.py')]
    assert all(str(path) in activation['sources_sha256'] for path in required)
    for path, expected in activation['sources_sha256'].items():
        assert digest(Path(path)) == expected, path
    for path, expected in activation['confirmation_sources_sha256'].items():
        assert digest(Path(path)) == expected, path
    return activation


def derive(task: dict) -> dict:
    """Reuse a verified completed teacher or derive one complete missing shot."""
    destination = Path(GENERATOR.OUTPUT) / task['part'] / (task['stem'] + '.npz')
    receipt = destination.with_suffix('.json')
    if receipt.exists():
        record = json.loads(receipt.read_text())
        assert record['stem'] == task['stem'] and record['frames'] == task['frames']
        assert record['part'] == task['part'] and record['output_path'] == str(destination)
        assert digest(destination) == record['output_sha256']
        return record
    assert not destination.exists(), 'Unreceipted target file requires an integrity review.'
    record = GENERATOR.derive_shot(task)
    write(receipt, record)
    return record


def main() -> None:
    """Create native four-position targets and full-data feature/target moments."""
    started = time.monotonic()
    activation = verify()
    import numpy as np
    draft = json.loads((HERE / 'preparation_draft_v1.json').read_text())
    d38, d40 = Path(draft['D38_root']), Path(draft['D40_root'])
    sys.path.insert(0, str(d38))
    from campaign_common import mapped_array
    from training_data import Moments
    plan = json.loads((HERE / 'metadata_row_plan_v1.json').read_text())
    out = HERE / 'all7041_data'
    out.mkdir(exist_ok=True)
    assert not (out / 'receipt.json').exists(), 'Completed data are verified, never regenerated.'
    spec = importlib.util.spec_from_file_location('production_native_teacher', activation['generator_source'])
    global GENERATOR
    GENERATOR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GENERATOR)
    GENERATOR.OUTPUT = out / 'missing_teachers'
    for part in ('training', 'calibration'):
        (GENERATOR.OUTPUT / part).mkdir(parents=True, exist_ok=True)
    GENERATOR.initialise()
    tasks = [{'stem': row['stem'], 'part': part, 'frames': row['frames'],
              'input_path': str(Path(activation['data_root']) / (row['stem'] + '.parquet'))}
             for part, rows in plan['parts'].items() for row in rows if row['cached_teacher'] is None]
    assert len(tasks) == 715 and {t['stem'] for t in tasks} == set(plan['missing_native_teacher_stems'])
    generated = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('fork')) as pool:
        for record in pool.map(derive, tasks, chunksize=1):
            generated[record['stem']] = record
            print(json.dumps({'production_teacher_shots': len(generated), 'required': 715}), flush=True)
    labels = tuple(draft['label_names'])
    names = list(GENERATOR.NAMES)
    columns = [names.index(name) for name in labels]
    xmom, ymom = Moments(582), Moments(4)
    target_paths, teacher_sources, totals = {}, {}, {}
    for part, rows in plan['parts'].items():
        path = out / (part + '_X_targets.npy')
        assert not path.exists(), 'Partial combined targets require an integrity review.'
        count = plan['totals'][part]['frames']
        target = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, 4))
        features = mapped_array(Path(plan['table_paths'][part]), 'features')
        assert features.shape == (count, 582)
        at = 0
        for row in rows:
            old = row['cached_teacher']
            record = old if old is not None else generated[row['stem']]
            teacher_path = Path(record['path'] if old is not None else record['output_path'])
            expected = record['sha256'] if old is not None else record['output_sha256']
            assert digest(teacher_path) == expected
            teacher_sources[str(teacher_path)] = expected
            with np.load(teacher_path, allow_pickle=False) as saved:
                assert saved['label_names'].astype(str).tolist() == names
                assert str(saved['stem']) == row['stem'] and len(saved['times_ms']) == row['frames']
                values = saved['labels'][:, columns]
                assert values.dtype == np.float32 and values.shape == (row['frames'], 4)
            lo, hi = row['feature_span']
            assert lo == at and hi - lo == len(values) and not np.isinf(values).any()
            target[lo:hi] = values
            for offset in range(lo, hi, 4096):
                stop = min(hi, offset + 4096)
                raw = np.asarray(features[offset:stop], dtype=float)
                assert np.isfinite(raw).all()
                xmom.add(raw)
                ymom.add(target[offset:stop].astype(float))
            at = hi
        assert at == count
        target.flush()
        target_paths[part] = str(path)
        totals[part] = {'shots': len(rows), 'frames': count}
        print(json.dumps({'production_normalized_part': part, **totals[part]}), flush=True)
    assert sum(t['shots'] for t in totals.values()) == 7041
    assert sum(t['frames'] for t in totals.values()) == 1559340
    assert (xmom.count > 0).all() and (ymom.count > 0).all()
    common = float(np.sqrt(np.mean(ymom.m2 / ymom.count)))
    assert common > 1e-6
    normalization = out / 'normalization.npz'
    assert not normalization.exists()
    with np.load(d40 / 'original6326_data/normalization.npz', allow_pickle=False) as old:
        np.savez(normalization, feature_mean=xmom.mean, feature_scale=xmom.scale(),
                 output_mean=ymom.mean, output_scale=np.full(4, common),
                 temporal_mean=old['temporal_mean'], temporal_scale=old['temporal_scale'],
                 finite_target_counts=ymom.count.astype(np.int64), labels=np.asarray(labels))
    verify()
    write(out / 'receipt.json', {'status': 'all7041_native_X_targets_and_active_normalizers_complete',
          'activation_sha256': digest(HERE / 'production_activation_v1.json'), 'totals': totals,
          'label_names': labels, 'target_paths': target_paths,
          'target_sha256': {part: digest(Path(path)) for part, path in target_paths.items()},
          'normalization_sha256': digest(normalization), 'teacher_sources_sha256': teacher_sources,
          'missing_native_teachers_derived': len(generated), 'cached_teachers_verified': 6326,
          'shared_output_scale_m': common, 'finite_target_counts': ymom.count.astype(int).tolist(),
          'unused_static_temporal_normalizers_retained': True,
          'targets_opened_only_after_reviewed357_PASS': True, 'seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
