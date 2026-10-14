# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — checkpointed input support census
"""Prepare pressure reliability masks without opening any target columns."""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from reliability import DATA, HERE, PARTS, digest, input_support, metadata, write_json

MAST = DATA.parent / 'mast_public_test'


def case(task: tuple[str, str, int]) -> dict:
    """Save one real input-only result and resume only verified checkpoints."""
    part, stem, frames = task
    directory = HERE / 'inputs/shots' / part
    path = directory / (stem + '.npz')
    receipt = directory / (stem + '.json')
    if receipt.exists():
        value = json.loads(receipt.read_text())
        assert value['sha256'] == digest(path) and value['frames'] == frames
        return value
    arrays, value = input_support((MAST if part == 'mast' else DATA) / (stem + '.parquet'))
    assert value['frames'] == frames
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for key, array in arrays.items():
                assert np.array_equal(saved[key], array, equal_nan=True), key
    else:
        temporary = directory / (stem + '.partial.npz')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    value.update(part=part, sha256=digest(path))
    write_json(receipt, value)
    print(json.dumps({'input_support_complete': stem, 'part': part}), flush=True)
    return value


def main() -> None:
    """Finish all DIII-D and MAST input masks before downstream fitting."""
    out = HERE / 'inputs'
    if (out / 'receipt.json').exists():
        assert json.loads((out / 'receipt.json').read_text())['status'] == 'all_input_support_prepared'
        return
    started = time.monotonic()
    metas = {part: metadata(part) for part in PARTS}
    tasks = []
    for part, meta in metas.items():
        (out / 'shots' / part).mkdir(parents=True, exist_ok=True)
        tasks.extend((part, stem, frames) for stem, frames in zip(meta['stems'], meta['counts'], strict=True))
    (out / 'shots/mast').mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    mast_files = sorted(MAST.glob('*.parquet'))
    assert len(mast_files) == 1206
    for file in mast_files:
        times = pq.read_table(file, columns=['efit_times'], use_threads=False)['efit_times'][0].as_py()
        tasks.append(('mast', file.stem, len(times)))
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts = list(pool.map(case, tasks, chunksize=1))
    for part, meta in metas.items():
        path = out / (part + '.npy')
        if path.exists():
            saved = np.load(path, mmap_mode='r', allow_pickle=False)
            assert saved.shape == (meta['rows'],) and saved.dtype == bool
            continue
        array = np.lib.format.open_memmap(out / (part + '.partial.npy'), mode='w+', dtype=bool, shape=(meta['rows'],))
        for i, stem in enumerate(meta['stems']):
            a, b = meta['bounds'][i:i+2]
            with np.load(out / 'shots' / part / (stem + '.npz'), allow_pickle=False) as stored:
                array[a:b] = stored['reliable']
        array.flush()
        del array
        (out / (part + '.partial.npy')).replace(path)
    totals = {part: {key: sum(r[key] for r in receipts if r['part'] == part)
                     for key in ('frames', 'reliable', 'not_running', 'unsupported_time', 'too_few_pairs')}
              for part in (*PARTS, 'mast')}
    write_json(out / 'receipt.json', {'status': 'all_input_support_prepared', 'shots': len(tasks),
        'counts': totals, 'seconds': time.monotonic() - started, 'MAST_targets_used': False,
        'official_clock_parity_all_DIII_D': True, 'comparative_metrics_computed': False})


if __name__ == '__main__':
    main()
