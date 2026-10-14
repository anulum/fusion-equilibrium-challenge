# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded pressure predictions for every frame
"""Write compact scalar predictions with unchanged map and q95 lineage."""

from __future__ import annotations

import json
import multiprocessing
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reliability import D26, D27, HERE, PKG, digest, factor, read_inputs, write_json
from fec_adapter.shot_reader import aligned_shot

sys.path.insert(0, str(D27 / 'repair_axis_flux_v1'))
from pressure_integral import electron_beta

ARMS = ('fixed_guarded', 'adaptive_guarded', 'refit_guarded')
ARCHIVE = D26 / 'full_score_v2/p45.npz'
STATE = {}


def initialize() -> None:
    """Open the immutable baseline and the existing physical-map reader."""
    sys.path.insert(0, str(PKG / 'entrypoints'))
    from shape_edit_fixed_axis_volume import Reader
    STATE['reader'] = Reader('DIII-D', backend='native')
    STATE['archive'] = zipfile.ZipFile(ARCHIVE)
    STATE['features'] = np.load(D27 / 'prepared/training/factor_features.npy', mmap_mode='r')


def case(item: dict) -> dict:
    """Keep the baseline on any frame without input and physical support."""
    stem, index, fold = item['stem'], item['index'], item['fold']
    out = HERE / 'predictions/shots'
    file, receipt = out / (stem + '.npz'), out / (stem + '.json')
    if receipt.exists():
        record = json.loads(receipt.read_text())
        assert digest(file) == record['sha256']
        return record
    support = HERE / 'inputs/shots/training' / (stem + '.npz')
    assert digest(support) == json.loads(support.with_suffix('.json').read_text())['sha256']
    with np.load(support, allow_pickle=False) as stored:
        reliable = stored['reliable']
    from pathlib import Path
    row = read_inputs(Path(item['path']))
    shot = aligned_shot(row, stem, np.asarray(row['efit_times'], dtype=float))
    archive = STATE['archive']
    maps = np.load(archive.open(f'shot_{index:04d}_psirz.npy'), allow_pickle=False)
    baseline = np.load(archive.open(f'shot_{index:04d}_betaN.npy'), allow_pickle=False)
    assert maps.shape == (shot.n_frames, 65, 65) and reliable.shape == baseline.shape == (shot.n_frames,)
    beta = np.full(shot.n_frames, np.nan)
    for i in np.flatnonzero(reliable):
        beta[i] = electron_beta(maps[i], STATE['reader'], shot, int(i))
    usable = reliable & np.isfinite(beta) & (beta > 0)
    a, b = item['span']
    features = STATE['features'][a:b].astype(float)
    assert len(features) == len(baseline)
    old = factor(D27 / f'pressure_factors/factor_fold{fold}.npz', features)
    new = factor(HERE / f'factors/factor_fold{fold}.npz', features)
    values = dict(zip(ARMS, [np.where(usable, beta * f, baseline).astype(np.float32)
                            for f in (2.25, old, new)], strict=True))
    assert all(np.isfinite(v).all() for v in values.values())
    assert all(np.array_equal(v[~usable], baseline[~usable]) for v in values.values())
    temporary = out / (stem + '.partial.npz')
    np.savez_compressed(temporary, baseline=baseline, electron_beta=beta.astype(np.float32),
                        input_supported=reliable, physical_applied=usable, **values)
    if file.exists():
        with np.load(file, allow_pickle=False) as saved, np.load(temporary, allow_pickle=False) as computed:
            assert set(saved.files) == set(computed.files)
            for key in saved.files:
                assert np.array_equal(saved[key], computed[key], equal_nan=True)
        temporary.unlink()
    else:
        temporary.replace(file)
    record = {'stem': stem, 'index': index, 'fold': fold, 'frames': len(baseline),
              'input_supported': int(reliable.sum()), 'physical_applied': int(usable.sum()),
              'all_predictions_finite': True, 'fallback_exact_baseline': True, 'sha256': digest(file)}
    write_json(receipt, record)
    print(json.dumps({'guarded_predictions_complete': index}), flush=True)
    return record


def main() -> None:
    """Complete every declared arm before processing any comparative score."""
    assert json.loads((HERE / 'factors/receipt.json').read_text())['status'] == 'all_five_supported_factors_complete'
    out = HERE / 'predictions'
    (out / 'shots').mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        return
    cases = json.loads((D27 / 'evaluation/manifest.json').read_text())['cases']
    assert len(cases) == 400
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        records = list(pool.map(case, cases, chunksize=1))
    assert len(records) == 400 and sum(r['frames'] for r in records) == 89584
    write_json(out / 'receipt.json', {'status': 'all_400_guarded_predictions_complete', 'arms': ARMS,
        'shots': 400, 'frames': 89584, 'records': records, 'baseline_archive': str(ARCHIVE),
        'baseline_archive_sha256': digest(ARCHIVE), 'maps_and_q95_unchanged': True,
        'comparative_metrics_computed': False, 'public_submission_archive_built': False})


if __name__ == '__main__':
    main()
