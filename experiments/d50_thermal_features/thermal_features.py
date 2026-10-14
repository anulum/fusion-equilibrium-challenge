# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D50: electron-pressure features of every reconstruction frame, from the Thomson pulses
"""Thermal features of every reconstruction frame, from predictor inputs only.

The table gives the networks electron temperature and density separately, as means over 8 cm bins of the one
pulse nearest to the frame. Normalised beta is a pressure integral over the field and the current, and the
Shafranov shift and the internal inductance follow the poloidal beta. Here every pulse of the vertical core laser
gives the electron pressure p = n T along the chord and its integrals; every frame gets them from the nearest
pulse and as means over the pulses within +-10 ms, +-50 ms and the 150 ms and 500 ms before it, and the same
integrals over the toroidal-field current, the plasma current and their products.

    PYTHONPATH=PKG/source/fec_adapter python thermal_features.py SPLIT_DIR STEMS.json OUT.npz WORKERS
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fec_adapter import machine
from fec_adapter import shot_reader as reader

WINDOWS = ((-10.0, 10.0), (-50.0, 50.0), (-150.0, 0.0), (-500.0, 0.0))
QUANTITIES = ('line_p', 'volume_p', 'peak_p', 'width', 'centre', 'line_n', 'line_t', 'edge_peak_p', 'edge_mean_p')
FLOOR_MA = 0.05
TRAPEZOID = getattr(np, 'trapezoid', None) or np.trapz


def integral(values, positions, valid):
    """Trapezoid integral along the chord over the valid channels of every pulse; nan below three channels."""
    out = np.full(values.shape[0], np.nan)
    for k in range(values.shape[0]):
        pick = valid[k]
        if pick.sum() >= 3:
            out[k] = TRAPEZOID(values[k, pick], positions[pick])
    return out


def core_pulses(te, ne, z):
    """Per-pulse integrals of the vertical laser; te in keV, ne in 1e19 m^-3, channels sorted by height."""
    order = np.argsort(z)
    te, ne, z = te[:, order], ne[:, order], z[order]
    valid = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    p = np.where(valid, te * ne, 0.0)
    line = integral(p, z, valid)
    moment = integral(p * z[None, :], z, valid)
    with np.errstate(invalid='ignore', divide='ignore'):
        centre = moment / line
    volume = integral(p * np.abs(z[None, :] - np.nan_to_num(centre)[:, None]), z, valid)
    spread = integral(p * (z[None, :] - np.nan_to_num(centre)[:, None]) ** 2, z, valid)
    with np.errstate(invalid='ignore', divide='ignore'):
        width = np.sqrt(np.maximum(spread / line, 0.0))
    peak = np.where(valid.any(axis=1), p.max(axis=1), np.nan)
    return {'line_p': line, 'volume_p': volume, 'peak_p': peak, 'width': width, 'centre': centre,
            'line_n': integral(np.where(valid, ne, 0.0), z, valid), 'line_t': integral(np.where(valid, te, 0.0), z, valid)}


def edge_pulses(te, ne):
    valid = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    p = np.where(valid, te * ne, 0.0)
    some = valid.any(axis=1)
    return {'edge_peak_p': np.where(some, p.max(axis=1), np.nan),
            'edge_mean_p': np.where(some, p.sum(axis=1) / np.maximum(valid.sum(axis=1), 1), np.nan)}


def nearest(times, pulses, values):
    if pulses.size == 0:
        return np.full(times.size, np.nan)
    if pulses.size == 1:
        return np.full(times.size, values[0])
    pick = np.clip(np.searchsorted(pulses, times), 1, pulses.size - 1)
    nearer = np.where(np.abs(times - pulses[pick - 1]) <= np.abs(pulses[pick] - times), pick - 1, pick)
    return values[nearer]


def window_mean(times, pulses, values, lo, hi):
    """Mean over the finite pulse values with lo <= pulse - frame <= hi; nan where there is none."""
    good = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(good, values, 0.0))])
    count = np.concatenate([[0.0], np.cumsum(good.astype(float))])
    first = np.searchsorted(pulses, times + lo, side='left')
    last = np.searchsorted(pulses, times + hi, side='right')
    n = count[last] - count[first]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (total[last] - total[first]) / n, np.nan), n


def sorted_block(times, *blocks):
    times = np.asarray(times, dtype=np.float64)
    keep = np.isfinite(times)
    order = np.argsort(times[keep], kind='stable')
    return times[keep][order], [np.asarray(b, dtype=np.float64)[keep][order] for b in blocks]


def features_of(path):
    names = pq.read_schema(path).names
    columns = reader.input_columns(dict.fromkeys(names))
    table = pq.read_table(path, columns=columns, use_threads=False)
    row = {}
    for name in columns:
        value = table[name][0]
        kind = table.schema.field(name).type
        flat = name.startswith('magnetics_') and str(kind) in ('list<element: double>', 'list<element: float>')
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(np.float64) if flat else value.as_py()
    times = np.asarray(row['efit_times'], dtype=np.float64)
    chord_names = [str(name) for name in row['thomson_chord_name']]
    chord_r = np.asarray(row['thomson_chord_R'], dtype=np.float64)
    chord_z = np.asarray(row['thomson_chord_Z'], dtype=np.float64)
    series = {}
    core_te = np.asarray(row['thomson_core_Te'], dtype=np.float64)
    core_ne = np.asarray(row['thomson_core_ne'], dtype=np.float64)
    if core_te.ndim == 2 and core_te.size and len(row['thomson_core_times']) == core_te.shape[0]:
        if core_te.shape[1] == len(row['thomson_core_R']) + 1:
            core_te, core_ne = core_te[:, 1:], core_ne[:, 1:]
        _, z = reader._core_positions(chord_names, chord_r, chord_z, core_te.shape[1], row)
        pulses, (te, ne) = sorted_block(row['thomson_core_times'], core_te, core_ne)
        if np.ptp(z) > 0:  # heights known: the fallback puts every channel at zero, where no integral exists
            for key, value in core_pulses(te / 1.0e3, ne / 1.0e19, z).items():
                series[key] = (pulses, value)
    edge_te = np.asarray(row['thomson_edge_Te'], dtype=np.float64)
    edge_ne = np.asarray(row['thomson_edge_ne'], dtype=np.float64)
    if edge_te.ndim == 2 and edge_te.size and len(row['thomson_edge_times']) == edge_te.shape[0]:
        pulses, (te, ne) = sorted_block(row['thomson_edge_times'], edge_te, edge_ne)
        for key, value in edge_pulses(te / 1.0e3, ne / 1.0e19).items():
            series[key] = (pulses, value)
    empty = (np.zeros(0), np.zeros(0))
    columns_out, by_window = [], {}
    for key in QUANTITIES:
        pulses, value = series.get(key, empty)
        by_window[key, 'nearest'] = nearest(times, pulses, value)
        columns_out.append(by_window[key, 'nearest'])
        for lo, hi in WINDOWS:
            mean, _ = window_mean(times, pulses, value, lo, hi) if pulses.size else (np.full(times.size, np.nan), None)
            by_window[key, lo] = mean
            columns_out.append(mean)
    pulses, value = series.get('line_p', empty)
    counts = [window_mean(times, pulses, value, lo, hi)[1] if pulses.size else np.zeros(times.size) for lo, hi in WINDOWS]
    if pulses.size:
        mean, _ = window_mean(times, pulses, value, -50.0, 50.0)
        square, _ = window_mean(times, pulses, value ** 2, -50.0, 50.0)
        spread = np.sqrt(np.maximum(square - mean ** 2, 0.0))
    else:
        spread = np.full(times.size, np.nan)
    current = np.maximum(np.abs(machine.plasma_current_si(row, times)) / 1.0e6, FLOOR_MA)
    toroidal = np.maximum(np.abs(machine.toroidal_current_si(row, times)) / 1.0e5, 1.0e-3)
    ratios = []
    for window in ('nearest', -10.0, -50.0, -150.0):
        line, volume = by_window['line_p', window], by_window['volume_p', window]
        ratios += [line / (toroidal * current), volume / (toroidal * current), line / current ** 2,
                   volume / current ** 2, volume / toroidal ** 2]
    out = np.column_stack([*columns_out, *counts, spread, *ratios])
    flags = np.column_stack([np.isfinite(by_window['line_p', 'nearest']), np.isfinite(by_window['line_p', -10.0]),
                             np.isfinite(by_window['line_p', -500.0]), np.isfinite(by_window['edge_peak_p', -10.0])])
    out = np.column_stack([np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), flags.astype(float)])
    assert out.shape == (times.size, WIDTH), out.shape
    basics = np.column_stack([current, toroidal, toroidal / current, 1.0 / current])
    return np.column_stack([basics, out]).astype(np.float32)


WIDTH = len(QUANTITIES) * (1 + len(WINDOWS)) + len(WINDOWS) + 1 + 20 + 4


def main():
    split, stems_file, out, workers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    stems = json.loads(stems_file.read_text())
    if len(sys.argv) > 5:
        stems = stems[:int(sys.argv[5])]
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        blocks = list(pool.map(features_of, [split / f'{stem}.parquet' for stem in stems], chunksize=8))
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    stacked = np.concatenate(blocks)
    basics, features = stacked[:, :4], stacked[:, 4:]
    shot_of_row = np.repeat(np.arange(len(blocks)), [len(b) for b in blocks])
    assert np.isfinite(features).all() and np.isfinite(basics).all() and features.shape[1] == WIDTH
    np.savez(out, features=features, basics=basics, offsets=offsets, shot_of_row=shot_of_row, stems=np.asarray(stems))
    print(json.dumps({'DONE': True, 'shots': len(blocks), 'frames': int(offsets[-1]), 'width': int(features.shape[1]),
                      'zero_share_by_column': np.round((features == 0).mean(axis=0), 3).tolist(),
                      'column_sd': np.round(features.std(axis=0), 4).tolist(),
                      'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
