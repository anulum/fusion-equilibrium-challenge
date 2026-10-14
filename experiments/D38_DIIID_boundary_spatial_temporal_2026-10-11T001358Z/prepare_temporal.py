# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — temporal cache alignment and eligible normalization
"""Verify cached input clocks, fit channel statistics and prepare development inputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from campaign_common import HERE, ROOT, DATA, digest, metadata, write_json
from window_cache import mapped_array

SOURCE = ROOT / 'work/first_agent/d20_raw_window/raw_window_features.py'


def unpack(flat: np.ndarray) -> np.ndarray:
    """Restore the uniform 41-time grid and the original 21-channel order."""
    assert flat.shape[1] == 840
    values = np.concatenate([flat[:, :40, None], flat[:, 40:800].reshape(-1, 40, 19), flat[:, 800:840, None]], axis=2)
    return np.concatenate([values[:, :20], np.zeros((len(flat), 1, 21), dtype=values.dtype), values[:, 20:]], axis=1)


def main() -> None:
    """Keep all labels out of temporal normalization and cache qualification."""
    out = HERE / 'temporal'
    receipt = json.loads((out / 'window_download_receipt.json').read_text())
    assert receipt['status'] == 'exact_three_D20_input_window_files_recovered'
    cohort = json.loads((HERE / 'cohort.json').read_text())
    spec = importlib.util.spec_from_file_location('native_window_source', SOURCE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_read = pq.read_table

    def read(*args, **kwargs):
        kwargs['use_threads'] = False
        return original_read(*args, **kwargs)

    pq.read_table = read
    checks = []
    total, sums, squares = 0, np.zeros(21), np.zeros(21)
    for part in ('training', 'calibration', 'validation'):
        allowed = set(cohort['reserved_confirmation'] if part == 'validation' else cohort[part])
        meta = metadata(part)
        selected = [(stem, meta['bounds'][i + 1] - meta['bounds'][i]) for i, stem in enumerate(meta['stems']) if stem in allowed]
        assert [s for s, _ in selected] == sorted(allowed)
        bounds = np.r_[0, np.cumsum([n for _, n in selected])]
        path = out / 'windows' / f'rawwindow_{part}.npz'
        identities = mapped_array(path, 'shot_of_row')
        values = mapped_array(path, 'features')
        assert np.array_equal(identities, np.repeat(np.arange(len(selected)), np.diff(bounds)))
        assert values.shape == (int(bounds[-1]), 840)
        for index in sorted({0, len(selected) // 3, 2 * len(selected) // 3, len(selected) - 1}):
            stem = selected[index][0]
            generated = native.features_of(DATA / (stem + '.parquet'))
            assert np.array_equal(generated, values[bounds[index]:bounds[index + 1]]), (part, stem)
            checks.append({'part': part, 'stem': stem, 'native_clock_window_bitexact': True, 'frames': len(generated)})
        if part != 'validation':
            for start in range(0, len(values), 4096):
                window = unpack(values[start:start + 4096]).astype(float)
                assert np.isfinite(window).all()
                total += window.shape[0] * window.shape[1]
                sums += window.sum(axis=(0, 1))
                squares += (window * window).sum(axis=(0, 1))
        write_json(out / f'{part}_alignment.json', {'part': part, 'frames': len(values),
                   'stems': [s for s, _ in selected], 'offsets': bounds.tolist(),
                   'all_shot_row_identities_exact': True, 'cache_sha256': digest(path)})
    mean = sums / total
    variance = np.maximum(0., squares / total - mean * mean)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.)
    assert np.isfinite(mean).all() and np.isfinite(scale).all()
    np.savez(out / 'normalization.npz', mean=mean, scale=scale)
    development = sorted(cohort['development'])
    assert len(development) == 358
    folder = out / 'development'
    folder.mkdir(exist_ok=True)
    frames = 0
    for index, stem in enumerate(development):
        path = folder / (stem + '.npy')
        if not path.exists():
            np.save(path, native.features_of(DATA / (stem + '.parquet')), allow_pickle=False)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        assert values.ndim == 2 and values.shape[1] == 840 and np.isfinite(values).all()
        frames += len(values)
        if (index + 1) % 25 == 0:
            print(json.dumps({'development_input_shots_complete': index + 1, 'frames': frames}), flush=True)
    assert frames == 78875
    write_json(out / 'preparation_receipt.json', {'status': 'eligible_channel_statistics_and_all_development_windows_complete',
               'eligible_normalization_frames': total // 41, 'normalization_sha256': digest(out / 'normalization.npz'),
               'native_source_sha256': digest(SOURCE), 'real_native_bitexact_checks': checks,
               'development_shots': 358, 'development_frames': frames,
               'development_or_confirmation_labels_used': False, 'confirmation_inputs_only_alignment_checked': True})


if __name__ == '__main__':
    main()
