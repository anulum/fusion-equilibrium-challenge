# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D16: what the 0.5 ms input signals do inside and around a 20 ms reconstruction frame
"""Sub-frame features of every reconstruction frame, from predictor inputs only.

The table's 367 features sample the inputs once per 20 ms frame. The plasma current and the coil currents are
recorded every 0.5 ms and the core Thomson laser fires every 7.8 ms. Here each frame gets
* the plasma current at offsets -20, -10, -5, -2, -1, +1, +2, +5, +10 ms minus its value at the frame, over the
  shot's peak current (9), its largest swing within +-10 ms (1) and the offset values' spread (1);
* the plasma current's mean over +-10 ms (21 instants, 1 ms apart) minus its value at the frame, over the peak (1);
* for every coil filament the change over +-10 ms and over +-5 ms, and its mean over +-10 ms minus its value at
  the frame, in MA-turns (3 per filament);
* the toroidal-field current's change over +-10 ms, relative (1);
* milliseconds since the last and until the next core and edge Thomson pulse, clipped to 100 (4).
The package's own clock handling is used (machine.plasma_current_si and neighbours).

    PYTHONPATH=R/fec_adapter python subframe_features.py SPLIT_DIR STEMS.json OUT.npz [first last]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from fec_adapter import machine
from fec_adapter.shot_reader import input_columns
import pyarrow.parquet as pq

OFFSETS = np.array([-20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0])


def pulse_gaps(times, pulses):
    pulses = np.sort(np.asarray(pulses, dtype=np.float64))
    pulses = pulses[np.isfinite(pulses)]
    if pulses.size == 0:
        return np.full(times.size, 100.0), np.full(times.size, 100.0)
    after = np.searchsorted(pulses, times, side='right')
    since = np.where(after > 0, times - pulses[np.maximum(after - 1, 0)], 100.0)
    until = np.where(after < pulses.size, pulses[np.minimum(after, pulses.size - 1)] - times, 100.0)
    return np.clip(since, 0.0, 100.0), np.clip(until, 0.0, 100.0)


def features_of(path):
    names = pq.read_schema(path).names
    columns = input_columns(dict.fromkeys(names))
    table = pq.read_table(path, columns=columns)
    row = {}
    for name in columns:  # numeric series as arrays once: the samplers convert their columns on every call
        value = table[name][0]
        kind = table.schema.field(name).type
        flat = name.startswith('magnetics_') and str(kind) in ('list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(np.float64) if flat else value.as_py()
    times = np.asarray(row['efit_times'], dtype=np.float64)
    coils = machine.coil_set(row)
    grid = np.concatenate([[-20.0], np.arange(-10.0, 10.5, 1.0)])  # one sampler call for every instant needed
    at = {float(d): k for k, d in enumerate(grid)}
    instants = (times[None, :] + grid[:, None]).ravel()
    ip = machine.plasma_current_si(row, instants).reshape(grid.size, times.size)
    cc = machine.coil_currents_si(row, coils, instants).reshape(grid.size, times.size, -1)
    tfc = machine.toroidal_current_si(row, instants).reshape(grid.size, times.size)
    ip0 = ip[at[0.0]]
    raw = np.abs(np.asarray(row['magnetics_plasma_current'], dtype=np.float64))
    peak = max(float(np.nanmax(np.abs(ip0))), float(np.nanpercentile(raw[np.isfinite(raw)], 99.9)) * 1.0e3 if np.isfinite(raw).any() else 0.0, 5.0e4)
    sign = np.sign(np.nanmedian(ip0)) or 1.0
    steps = sign * (np.column_stack([ip[at[float(d)]] for d in OFFSETS]) - ip0[:, None]) / peak
    near = steps[:, (np.abs(OFFSETS) <= 10.0)]
    swing = near.max(axis=1).clip(min=0.0) - near.min(axis=1).clip(max=0.0)
    coil10 = sign * (cc[at[10.0]] - cc[at[-10.0]]) / 1.0e6
    coil5 = sign * (cc[at[5.0]] - cc[at[-5.0]]) / 1.0e6
    ip_mean = sign * (ip[1:].mean(axis=0) - ip0) / peak
    coil_mean = sign * (cc[1:].mean(axis=0) - cc[at[0.0]]) / 1.0e6
    tf = (tfc[at[10.0]] - tfc[at[-10.0]]) / np.maximum(np.abs(tfc[at[0.0]]), 1.0)
    gaps = [g for key in ('thomson_core_times', 'thomson_edge_times') for g in pulse_gaps(times, row[key])]
    out = np.column_stack([steps, swing, steps.std(axis=1), ip_mean, coil10, coil5, coil_mean, tf, *gaps])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main():
    split, stems_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    stems = json.loads(stems_file.read_text())
    first, last = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, len(stems))
    started, blocks = time.time(), []
    for i, stem in enumerate(stems[first:last]):
        blocks.append(features_of(split / f'{stem}.parquet'))
        if i % 200 == 199:
            print(json.dumps({'shots': i + 1, 'seconds': round(time.time() - started)}), flush=True)
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    np.savez(out, features=np.concatenate(blocks), offsets=offsets, stems=np.asarray(stems[first:last]))
    print(json.dumps({'DONE': True, 'shots': len(blocks), 'frames': int(offsets[-1]), 'width': int(blocks[0].shape[1]),
                      'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
