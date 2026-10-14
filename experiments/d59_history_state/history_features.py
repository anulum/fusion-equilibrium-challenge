# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D59: slow-memory features of the whole discharge
"""What the discharge did over seconds before a frame, from predictor inputs only.

The table reaches back 0.3 s and 1.3 s by window means. The current profile, which sets the flux in the plasma core
and the internal inductance, relaxes over a resistive time of seconds and remembers how the current was ramped.
Per frame, on a 5 ms grid of the whole discharge record: exponentially weighted means of the plasma current over
0.3, 1, 3 and 10 s relative to its present value; its relative change over 0.3, 1 and 3 s; time since it first
passed 10, 50 and 90 percent of the discharge peak, the ramp duration, the present value over the peak, time since
the peak; its time integral over (present value x elapsed time); the solenoid current's change since the start,
alone and over the plasma current; the same weighted means for the core Thomson temperature integral and peak
pressure; and a resistive clock, the integral of temperature^-1.5 since the start, relative to elapsed time.
"""
import numpy as np
import pyarrow.parquet as pq

from fec_adapter import machine
from fec_adapter import shot_reader as reader
import thermal_features as thermal

STEP_MS, TAUS_S = 5.0, (0.3, 1.0, 3.0, 10.0)
WIDTH = 4 + 3 + 3 + 3 + 1 + 2 + 2 * 3 + 2


def ema(values, tau_s):
    alpha = 1.0 - np.exp(-STEP_MS / 1.0e3 / tau_s)
    out = np.empty_like(values)
    state = values[0]
    for i, v in enumerate(values):
        state = state + alpha * (v - state)
        out[i] = state
    return out


def at(grid, series, times):
    return np.interp(times, grid, series)


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
    base = np.asarray(row['magnetics_time'], dtype=np.float64)
    base = base[np.isfinite(base)]
    grid = np.arange(min(base.min(), times.min()), times.max() + STEP_MS, STEP_MS)
    current = np.abs(machine.plasma_current_si(row, grid)) / 1.0e6
    current = np.nan_to_num(current)
    coils = machine.coil_set(row)
    solenoid = np.nan_to_num(machine.coil_currents_si(row, coils, grid)[:, -1]) / 1.0e6
    peak = max(float(current.max()), 0.05)
    now = np.maximum(at(grid, current, times), 0.02)
    blocks = [at(grid, ema(current, tau), times) / now - 1.0 for tau in TAUS_S]
    blocks += [(now - at(grid, current, times - 1.0e3 * tau)) / now for tau in TAUS_S[:3]]
    first = lambda share: grid[np.argmax(current >= share * peak)]
    t10, t50, t90, t_peak = first(0.1), first(0.5), first(0.9), grid[np.argmax(current)]
    blocks += [(times - t10) / 1.0e3, (times - t50) / 1.0e3, (times - t90) / 1.0e3]
    blocks += [np.full(times.size, (t90 - t10) / 1.0e3), now / peak, (times - t_peak) / 1.0e3]
    integral = np.cumsum(current) * STEP_MS
    elapsed = np.maximum(times - t10, STEP_MS)
    blocks += [(at(grid, integral, times) - at(grid, integral, np.full(times.size, t10))) / (now * elapsed)]
    change = at(grid, solenoid, times) - solenoid[0]
    blocks += [change, change / now]
    # Thomson memory on the same grid (pulse values held between pulses)
    chord_names = [str(n) for n in row['thomson_chord_name']]
    chord_r, chord_z = np.asarray(row['thomson_chord_R'], dtype=np.float64), np.asarray(row['thomson_chord_Z'], dtype=np.float64)
    te, ne = np.asarray(row['thomson_core_Te'], dtype=np.float64), np.asarray(row['thomson_core_ne'], dtype=np.float64)
    held = {}
    if te.ndim == 2 and te.size and len(row['thomson_core_times']) == te.shape[0]:
        if te.shape[1] == len(row['thomson_core_R']) + 1:
            te, ne = te[:, 1:], ne[:, 1:]
        _, z = reader._core_positions(chord_names, chord_r, chord_z, te.shape[1], row)
        pulses, (te, ne) = thermal.sorted_block(row['thomson_core_times'], te, ne)
        if np.ptp(z) > 0 and pulses.size:
            series = thermal.core_pulses(te / 1.0e3, ne / 1.0e19, z)
            for key in ('line_t', 'peak_p'):
                value = series[key]
                good = np.isfinite(value)
                if good.sum() >= 2:
                    pick = np.clip(np.searchsorted(pulses[good], grid, side='right') - 1, 0, good.sum() - 1)
                    held[key] = value[good][pick]
    for key in ('line_t', 'peak_p'):
        if key in held:
            present = np.maximum(at(grid, held[key], times), 1e-6)
            blocks += [at(grid, ema(held[key], tau), times) / present - 1.0 for tau in TAUS_S[:3]]
        else:
            blocks += [np.zeros(times.size)] * 3
    if 'line_t' in held:
        clock = np.cumsum(np.maximum(held['line_t'], 1e-3) ** -1.5) * STEP_MS
        started = at(grid, clock, np.full(times.size, t10))
        blocks += [(at(grid, clock, times) - started) / elapsed, np.log(np.maximum(at(grid, clock, times) - started, 1e-6))]
    else:
        blocks += [np.zeros(times.size)] * 2
    out = np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape == (times.size, WIDTH), out.shape
    return np.clip(out, -1e3, 1e3).astype(np.float32)
