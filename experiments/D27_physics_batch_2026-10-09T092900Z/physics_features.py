# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — input-only pressure and full-discharge memory features
"""Build the frozen blocks without giving target reconstructions to a predictor."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from campaign_common import PKG
from fec_adapter.cross_machine.thomson_isoflux import green
from fec_adapter.diii_d_accuracy.features import (
    HEIGHT_EDGES, RADIUS_EDGES, VERTICAL_TOLERANCE, binned_profile,
)
from fec_adapter.discharge import discharge_window
from fec_adapter.input_features import profile_summary
from fec_adapter.orientation import flux_orientation
from fec_adapter.shot_reader import aligned_shot, input_columns

TAUS = (.05, .2, 1., 3.)


def read_inputs(path):
    columns = input_columns(dict.fromkeys(pq.read_schema(path).names))
    assert 'efit_psirz' not in columns and 'magnetics_dsep' not in columns
    table = pq.read_table(path, columns=columns, use_threads=False)
    row = {}
    for name in columns:
        kind, value = table.schema.field(name).type, table[name][0]
        flat = name.startswith('magnetics_') and (pa.types.is_list(kind) or pa.types.is_large_list(kind)) and (
            pa.types.is_floating(kind.value_type) or pa.types.is_integer(kind.value_type))
        row[name] = value.values.to_numpy(zero_copy_only=False).astype(float) if flat else value.as_py()
    return row


def full_inputs(row, stem):
    target_times = np.asarray(row['efit_times'], dtype=float)
    window = discharge_window(row)
    regular = np.arange(min(window.start_ms, target_times[0]), target_times[-1] + .001, 20.)
    times = np.unique(np.r_[regular, target_times])
    positions = np.searchsorted(times, target_times)
    assert np.array_equal(times[positions], target_times)
    return aligned_shot(row, stem, times), positions, window


def pressure_features(shot):
    te, ne = shot.thomson.te, shot.thomson.ne
    valid = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    pressure = np.where(valid, te * ne * 1.602176634e-19, np.nan)
    on_vertical = np.abs(shot.thomson.r - np.median(shot.thomson.r)) < VERTICAL_TOLERANCE
    blocks, summaries = [], []
    for member, positions, edges in ((on_vertical, shot.thomson.z, HEIGHT_EDGES),
                                     (~on_vertical, shot.thomson.r, RADIUS_EDGES)):
        values, fired = binned_profile(pressure, positions, member, edges)
        blocks.extend([values / 1e4, fired])
        summaries.append((values, fired, (edges[1:] + edges[:-1]) / 2))
    for values, fired, centres in summaries:
        good = fired[:, 1:] * fired[:, :-1]
        slope = np.diff(values, axis=1) / np.diff(centres)
        blocks.extend([slope * good / 1e4, good])
    for values, fired, _ in summaries:
        if shot.n_frames > 1:
            rate = np.gradient(values, shot.times_ms / 1000., axis=0)
            good = fired.copy()
            good[1:] *= fired[:-1]
            good[:-1] *= fired[1:]
        else:
            rate, good = np.zeros_like(values), np.zeros_like(fired)
        blocks.extend([rate * good / 1e5, good])
    global_summary = profile_summary(pressure)
    blocks.append(global_summary / np.array([1e4, 1e4, 1.]))
    result = np.column_stack(blocks)
    assert result.shape == (shot.n_frames, 107) and np.isfinite(result).all()
    return result, global_summary


def coil_projection(shot, basis, orientation):
    rr, zz = np.meshgrid(shot.r_grid, shot.z_grid)
    units = np.stack([green(rr, zz, r, z).ravel() for r, z in zip(shot.coils.r, shot.coils.z, strict=True)])
    projection = units @ basis.directions[:64].T
    result = orientation * shot.coil_current @ projection
    assert result.shape == (shot.n_frames, 64) and np.isfinite(result).all()
    return result, units


def memory_features(shot, coil_coordinates, summary):
    driving = np.column_stack([coil_coordinates[:, :8], shot.plasma_current, shot.tf_current,
        profile_summary(shot.thomson.te), profile_summary(shot.thomson.ne), summary])
    assert driving.shape[1] == 19 and np.isfinite(driving).all()
    blocks = []
    for tau in TAUS:
        filtered = np.empty_like(driving)
        filtered[0] = driving[0]
        for i in range(1, len(driving)):
            decay = np.exp(-(shot.times_ms[i] - shot.times_ms[i - 1]) / (1000. * tau))
            filtered[i] = decay * filtered[i - 1] + (1. - decay) * driving[i]
        blocks.append(filtered - driving)
    result = np.column_stack(blocks)
    assert result.shape == (shot.n_frames, 76) and np.isfinite(result).all()
    return result


def factor_features(shot, summary, window):
    te, ne = profile_summary(shot.thomson.te), profile_summary(shot.thomson.ne)
    phase = window.phase(shot.times_ms, shot.plasma_current)
    mean_r = float(np.mean(shot.r_grid))
    from fec_adapter.cross_machine.map_scalars import vacuum_f
    field = np.array([vacuum_f(shot.source, value) / mean_r for value in shot.tf_current])
    beta_proxy = 2 * 4e-7 * np.pi * summary[:, 1] / np.maximum(field**2, 1e-12)
    result = np.column_stack([
        np.log(np.maximum(te[:, 0] / np.maximum(te[:, 1], 1e-12), 1)),
        np.log(np.maximum(ne[:, 0] / np.maximum(ne[:, 1], 1e-12), 1)),
        np.log(np.maximum(summary[:, 0] / np.maximum(summary[:, 1], 1e-12), 1)),
        summary[:, 2], phase[:, 1], np.log(np.maximum(beta_proxy, 1e-12)),
        np.log(np.maximum(te[:, 1] / 1000., 1e-12)),
        np.log(np.maximum(ne[:, 1] / 1e19, 1e-12))])
    assert result.shape == (shot.n_frames, 8) and np.isfinite(result).all()
    return result


def predictor_blocks(row, stem, basis):
    shot, positions, window = full_inputs(row, stem)
    orientation = flux_orientation(shot.plasma_current[positions])
    pressure, summary = pressure_features(shot)
    coils, units = coil_projection(shot, basis, orientation)
    memory = memory_features(shot, coils, summary)
    factors = factor_features(shot, summary, window)
    return {'pressure': pressure[positions].astype(np.float32),
            'memory': memory[positions].astype(np.float32),
            'coil_coordinates': coils[positions].astype(np.float32),
            'factor_features': factors[positions].astype(np.float32),
            'times_ms': shot.times_ms[positions]}, shot, positions, orientation, units
