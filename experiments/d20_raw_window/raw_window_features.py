# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D20: the raw 0.5 ms samples of the input signals around each reconstruction frame
"""For every frame the signals themselves at the recorder's own step: 41 instants from -10 ms to +10 ms, 0.5 ms
apart, for the plasma current, every coil circuit (one filament of each) and the toroidal-field current.

Each sample is the signal at that instant minus its value at the frame (the table already holds the frame value):
plasma current over the shot's peak, coil filaments in MA-turns, TF relative. Nothing is averaged or summarised:
the network is to read the sequence itself (entrant, 2026-10-06). Predictor inputs only; package clock handling;
one sampler call per signal. Stored as float16 (the values are small differences).

    PYTHONPATH=R/fec_adapter python raw_window_features.py SPLIT_DIR STEMS.json OUT.npz [first last]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fec_adapter import machine
from fec_adapter.shot_reader import input_columns

GRID = np.arange(-10.0, 10.25, 0.5)


def features_of(path):
    names = pq.read_schema(path).names
    columns = input_columns(dict.fromkeys(names))
    table = pq.read_table(path, columns=columns)
    row = {}
    for name in columns:
        value = table[name][0]
        flat = name.startswith('magnetics_') and str(table.schema.field(name).type) in ('list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(np.float64) if flat else value.as_py()
    times = np.asarray(row['efit_times'], dtype=np.float64)
    coils = machine.coil_set(row)
    instants = (times[:, None] + GRID[None, :]).ravel()
    zero = int(np.argmin(np.abs(GRID)))
    ip = machine.plasma_current_si(row, instants).reshape(times.size, GRID.size)
    first = [list(coils.columns).index(c) for c in dict.fromkeys(coils.columns)]  # one filament per circuit
    cc = machine.coil_currents_si(row, coils, instants)[:, first].reshape(times.size, GRID.size, -1)
    tf = machine.toroidal_current_si(row, instants).reshape(times.size, GRID.size)
    raw = np.abs(np.asarray(row['magnetics_plasma_current'], dtype=np.float64))
    peak = max(float(np.nanmax(np.abs(ip[:, zero]))), float(np.nanpercentile(raw[np.isfinite(raw)], 99.9)) * 1.0e3 if np.isfinite(raw).any() else 0.0, 5.0e4)
    sign = np.sign(np.nanmedian(ip[:, zero])) or 1.0
    d_ip = sign * (ip - ip[:, [zero]]) / peak
    d_cc = sign * (cc - cc[:, [zero], :]) / 1.0e6
    d_tf = (tf - tf[:, [zero]]) / np.maximum(np.abs(tf[:, [zero]]), 1.0)
    keep = np.arange(GRID.size) != zero
    out = np.concatenate([d_ip[:, keep], d_cc[:, keep, :].reshape(times.size, -1), d_tf[:, keep]], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def main():
    split, stems_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    stems = json.loads(stems_file.read_text())
    first, last = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, len(stems))
    started, blocks = time.time(), []
    for stem in stems[first:last]:
        blocks.append(features_of(split / f'{stem}.parquet'))
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    np.savez(out, features=np.concatenate(blocks), offsets=offsets, stems=np.asarray(stems[first:last]))
    print(json.dumps({'DONE': True, 'shots': len(blocks), 'frames': int(offsets[-1]), 'width': int(blocks[0].shape[1]),
                      'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
