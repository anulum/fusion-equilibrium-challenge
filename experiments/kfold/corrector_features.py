# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — map-aware X-point corrector, stage A: per-frame features and targets
"""Read an archive of finished maps and the true maps, every 2nd frame, and write per frame:
the map-side features of protocol_mapaware_xpoint_corrector_20260929, the true raw-map
saddles (targets) and the v25-style reading of the map. Used for the OOF fold archives.

    python corrector_features.py --archive NPZ --stems-from RECEIPT_JSON --out NPZ [--workers 6]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA = Path('$FEC_DATA/data/diii_d_train')
TABLES = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge')
STRIDE = 2
ENDS = ('lower', 'upper')
MAP_COLUMNS = (['lower_r', 'lower_z', 'lower_hrr', 'lower_hrz', 'lower_hzz', 'lower_psin', 'lower_found',
                'upper_r', 'upper_z', 'upper_hrr', 'upper_hrz', 'upper_hzz', 'upper_psin', 'upper_found',
                'tip_low_r', 'tip_low_z', 'tip_high_r', 'tip_high_z', 'axis_r', 'axis_z',
                'R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li'])
_READER = None


def sample(phi, reader, pts):
    from scipy.ndimage import map_coordinates
    sr = float(reader.r_grid[1] - reader.r_grid[0])
    sz = float(reader.z_grid[1] - reader.z_grid[0])
    return map_coordinates(phi, [(pts[:, 1] - reader.z_grid[0]) / sz, (pts[:, 0] - reader.r_grid[0]) / sr],
                           order=1, mode='nearest')


def map_row(reader, flux):
    from fec_adapter.diii_d_accuracy.x_point import boundary_saddles
    from fec_adapter.scoring_kit.common import AXIS_SIGN, CONS_SCALARS
    reading = reader.read(flux)
    row = np.full(len(MAP_COLUMNS), np.nan)
    if reading.contour is None:
        return row, None
    c = np.asarray(reading.contour, float)
    phi = AXIS_SIGN['DIII-D'] * flux
    level = float(np.mean(sample(phi, reader, c)))
    axis_phi = float(sample(phi, reader, np.asarray([reading.axis]))[0])
    span = axis_phi - level
    saddles = boundary_saddles(reader, flux, reading)
    for i, end in enumerate(ENDS):
        s = saddles[end]
        base = 7 * i
        if s is not None:
            row[base:base + 7] = [s.position[0], s.position[1], s.hessian[0, 0], s.hessian[0, 1], s.hessian[1, 1],
                                  (float(sample(phi, reader, s.position[None])[0]) - level) / span if span > 0 else np.nan, 1.0]
        else:
            row[base + 6] = 0.0
    low, high = c[np.argmin(c[:, 1])], c[np.argmax(c[:, 1])]
    row[14:20] = [low[0], low[1], high[0], high[1], reading.axis[0], reading.axis[1]]
    row[20:27] = [reading.values[n] for n in CONS_SCALARS]
    return row, saddles


def shot_task(task):
    global _READER
    from fec_adapter.diii_d_accuracy.shape_edit import Reader
    from fec_adapter.diii_d_accuracy.x_point import boundary_saddles
    if _READER is None:
        _READER = Reader('DIII-D')
    archive_path, index, stem = task
    with np.load(archive_path, allow_pickle=False) as archive:
        maps = archive[f'shot_{index:04d}_psirz']
    truth = np.asarray(pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])
                       ['efit_psirz'][0].as_py(), dtype=np.float64)
    assert truth.shape == maps.shape, stem
    frames = np.arange(0, len(maps), STRIDE)
    feats = np.full((len(frames), len(MAP_COLUMNS)), np.nan)
    target = np.full((len(frames), 4), np.nan)
    for k, f in enumerate(frames):
        feats[k], _ = map_row(_READER, np.asarray(maps[f], np.float64))
        st = boundary_saddles(_READER, truth[f])
        for i, end in enumerate(ENDS):
            if st[end] is not None:
                target[k, 2 * i:2 * i + 2] = st[end].position
    return stem, index, frames, feats, target, len(maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--stems-from', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    stems = json.loads(args.stems_from.read_text())['stems']
    tasks = [(str(args.archive), i, s) for i, s in enumerate(stems)]
    with Pool(args.workers) as pool:
        results = pool.map(shot_task, tasks, chunksize=2)
    with np.load(TABLES / 'table_training.npz', allow_pickle=False) as table:
        tstems = table['stems'].astype(str)
        counts = np.bincount(table['shot_of_row'], minlength=len(tstems))
        starts = np.concatenate(([0], np.cumsum(counts)))
        where = {s: i for i, s in enumerate(tstems)}
        for s, _, _, _, _, n in results:
            assert counts[where[s]] == n, s
        index = np.concatenate([starts[where[s]] + fr for s, _, fr, _, _, _ in results])
        inputs = table['features'][index].astype(np.float32)
    np.savez_compressed(args.out, stems=np.asarray([r[0] for r in results]),
                        shot_of_row=np.concatenate([np.full(len(r[2]), k) for k, r in enumerate(results)]),
                        archive_index=np.concatenate([np.full(len(r[2]), r[1]) for r in results]),
                        frame=np.concatenate([r[2] for r in results]),
                        map_features=np.concatenate([r[3] for r in results]), map_columns=np.asarray(MAP_COLUMNS),
                        target=np.concatenate([r[4] for r in results]), inputs=inputs)
    print(json.dumps({'shots': len(results), 'rows': int(sum(len(r[2]) for r in results))}), flush=True)


if __name__ == '__main__':
    main()
