# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D27 axis-flux discretisation repair
"""Regenerate pressure supervision only; preserve all trained predictor blocks."""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
from campaign_common import DATA, HERE, PARTS, PKG, metadata, write_json, digest
from physics_features import full_inputs, read_inputs
from pressure_integral import electron_beta
STATE = {}


def initialize(directory):
    sys.path.insert(0, str(PKG / 'entrypoints'))
    from shape_edit_fixed_axis_volume import Reader
    STATE['reader'] = Reader('DIII-D', backend='native')
    STATE['directory'] = Path(directory)


def case(task):
    part, stem, frames = task
    path = DATA / (stem + '.parquet')
    shot, positions, _ = full_inputs(read_inputs(path), stem)
    truth = np.asarray(pq.read_table(path, columns=['efit_psirz'], use_threads=False)['efit_psirz'][0].as_py(), dtype=float)
    assert truth.shape == (frames, 65, 65) and len(positions) == frames
    values = np.asarray([electron_beta(flux, STATE['reader'], shot, int(index))
                         for flux, index in zip(truth, positions, strict=True)], dtype=np.float32)
    output = STATE['directory'] / 'shots' / part / (stem + '.npy')
    with output.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    print(json.dumps({'beta_target_regenerated': stem, 'part': part, 'frames': frames}), flush=True)
    return {'stem': stem, 'part': part, 'frames': frames, 'finite': int(np.isfinite(values).sum()),
            'positive': int((np.isfinite(values) & (values > 0)).sum()), 'sha256': digest(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    start = time.monotonic()
    metas = {part: metadata(part) for part in PARTS}
    tasks = []
    for part, meta in metas.items():
        (args.out / 'shots' / part).mkdir(parents=True)
        tasks.extend((part, stem, frames) for stem, frames in zip(meta['stems'], meta['counts'], strict=True))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.out,)) as pool:
        receipts = list(pool.map(case, tasks, chunksize=1))
    write_json(args.out / 'shot_receipts.json', receipts)
    counts = {}
    for part, meta in metas.items():
        output = args.out / (part + '.npy')
        array = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(meta['rows'],))
        for i, stem in enumerate(meta['stems']):
            a, b = meta['bounds'][i:i+2]
            array[a:b] = np.load(args.out / 'shots' / part / (stem + '.npy'), allow_pickle=False)
        array.flush()
        counts[part] = int((np.isfinite(array) & (array > 0)).sum())
    assert counts['training'] > 0 and counts['calibration'] > 0
    write_json(args.out / 'receipt.json', {'status': 'all_beta_targets_regenerated',
        'shots': len(tasks), 'frames': sum(t[2] for t in tasks), 'positive_rows': counts,
        'seconds': time.monotonic()-start, 'comparative_metrics_computed': False,
        'predictor_blocks_modified': False, 'integration_source_sha256': digest(Path(__file__).parent / 'pressure_integral.py')})


if __name__ == '__main__':
    main()
