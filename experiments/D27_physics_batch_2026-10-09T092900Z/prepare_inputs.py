# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — single-pass preparation for all frozen physics experiments
"""Extract predictor blocks separately from pressure-factor training supervision."""

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from campaign_common import DATA, HERE, PARTS, PKG, digest, metadata, write_json
from physics_features import predictor_blocks, read_inputs
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from fec_adapter.cross_machine import map_scalars

STATE = {}


def initialize(out):
    STATE['out'] = Path(out)
    STATE['basis'] = FluxBasis.load(PKG / 'models/flux_basis.npz').truncated(64)
    sys.path.insert(0, str(PKG / 'entrypoints'))
    from shape_edit_fixed_axis_volume import Reader
    STATE['reader'] = Reader('DIII-D', backend='native')


def prepare_case(task):
    part, stem, expected_frames = task
    path = DATA / f'{stem}.parquet'
    row = read_inputs(path)
    blocks, shot, positions, orientation, units = predictor_blocks(row, stem, STATE['basis'])
    assert len(positions) == expected_frames
    target_table = pq.read_table(path, columns=['efit_psirz', 'efit_beta_n'], use_threads=False)
    truth = np.asarray(target_table['efit_psirz'][0].as_py(), dtype=float)
    beta_e = np.full(expected_frames, np.nan, dtype=np.float32)
    assert truth.shape == (expected_frames, 65, 65)
    for index, time_index in enumerate(positions):
        reading = STATE['reader'].read(truth[index])
        if reading.contour is None:
            continue
        axis = reading.axis
        axis_flux = float(map_scalars._at(truth[index], shot.r_grid, shot.z_grid,
                                         np.array([axis[0]]), np.array([axis[1]]))[0])
        edge_flux = float(np.mean(map_scalars._at(truth[index], shot.r_grid, shot.z_grid,
                                                reading.contour[:, 0], reading.contour[:, 1])))
        if not np.isfinite(edge_flux - axis_flux) or abs(edge_flux - axis_flux) < 1e-12:
            continue
        beta_e[index] = map_scalars.normalised_beta(
            truth[index], shot.r_grid, shot.z_grid, axis, axis_flux, edge_flux,
            shot.thomson.r, shot.thomson.z, shot.thomson.te[time_index], shot.thomson.ne[time_index],
            float(shot.plasma_current[time_index]),
            map_scalars.vacuum_f('DIII-D', float(shot.tf_current[time_index]))) / map_scalars.PRESSURE_FACTOR
    blocks['beta_e_truth'] = beta_e
    out = STATE['out'] / 'shots' / part / f'{stem}.npz'
    with out.open('xb') as stream:
        np.savez_compressed(stream, **blocks)
    receipt = {'part': part, 'stem': stem, 'frames': expected_frames,
               'sha256': digest(out), 'frame_times_sha256': hashlib.sha256(blocks['times_ms'].tobytes()).hexdigest(),
               'pressure_shape': list(blocks['pressure'].shape), 'memory_shape': list(blocks['memory'].shape),
               'beta_e_defined': int(np.isfinite(beta_e).sum()), 'orientation': orientation}
    print(json.dumps({'prepared': stem, 'part': part, 'frames': expected_frames}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    start = time.monotonic()
    metas = {part: metadata(part) for part in PARTS}
    tasks = []
    for part, meta in metas.items():
        (args.out / 'shots' / part).mkdir(parents=True)
        for stem, count in zip(meta['stems'], meta['counts'], strict=True):
            tasks.append((part, stem, count))
    if args.smoke:
        tasks = tasks[:1]
    write_json(args.out / 'manifest.json', {'smoke': args.smoke, 'metadata': metas,
        'tasks': tasks, 'basis_sha256': digest(PKG / 'models/flux_basis.npz'),
        'protocol_sha256': digest(HERE / 'protocol.json'),
        'inputs_read_separately_from_beta_e_training_targets': True,
        'pressure_factor_source': 'normalised_beta(true_training_map, allowed_Thomson_inputs)/2.25',
        'coil_turns_convention': 'existing machine.py ECOIL_EFFECTIVE_TURNS148.7; training-calibrated, not measured physical turns'})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.out,)) as pool:
        receipts = list(pool.map(prepare_case, tasks, chunksize=1))
    write_json(args.out / 'shot_receipts.json', receipts)
    if not args.smoke:
        sizes = {'pressure': 107, 'memory': 76, 'coil_coordinates': 64, 'factor_features': 8,
                 'beta_e_truth': None}
        for part, meta in metas.items():
            directory = args.out / part
            directory.mkdir()
            arrays = {name: np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+', dtype=np.float32,
                        shape=(meta['rows'],) if width is None else (meta['rows'], width))
                      for name, width in sizes.items()}
            for i, stem in enumerate(meta['stems']):
                start_row, stop_row = meta['bounds'][i:i + 2]
                with np.load(args.out / 'shots' / part / f'{stem}.npz', allow_pickle=False) as stored:
                    for name in arrays:
                        assert len(stored[name]) == stop_row - start_row
                        arrays[name][start_row:stop_row] = stored[name]
            for array in arrays.values():
                array.flush()
            write_json(directory / 'metadata.json', meta)
    write_json(args.out / 'receipt.json', {'status': 'input_smoke_complete' if args.smoke else 'all_physics_inputs_prepared',
        'shots': len(tasks), 'frames': sum(x[2] for x in tasks), 'seconds': time.monotonic() - start,
        'comparative_metrics_computed': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
