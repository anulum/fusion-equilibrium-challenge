# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — measured pressure support
"""Mark physical-pressure support using allowed measurements only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
D27 = HERE.parent / 'D27_physics_batch_2026-10-09T092900Z'
sys.path.insert(0, str(D27))
from campaign_common import DATA, D26, PARTS, PKG, metadata, digest, write_json
from physics_features import read_inputs
from fec_adapter import machine
from fec_adapter.discharge import RUNNING_SHARE, discharge_window
from fec_adapter.shot_reader import aligned_shot
from official_data_fixes import fix_d3d_ip_times


def pulse_support(times: NDArray[np.float64], targets: NDArray[np.float64]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Check the recorded span and distance to the nearest laser pulse."""
    finite = np.sort(times[np.isfinite(times)])
    if not len(finite):
        return np.zeros(len(targets), dtype=bool), np.full(len(targets), np.inf)
    pick = np.searchsorted(finite, targets)
    left = finite[np.clip(pick - 1, 0, len(finite) - 1)]
    right = finite[np.clip(pick, 0, len(finite) - 1)]
    age = np.minimum(np.abs(targets - left), np.abs(targets - right))
    spacing = np.diff(finite)
    positive = spacing[spacing > 0]
    limit = 2 * float(np.median(positive)) if len(positive) else 0.
    return (targets >= finite[0]) & (targets <= finite[-1]) & (age <= limit), age


def input_support(path: Path) -> tuple[dict[str, NDArray], dict]:
    """Read a real shot through the production reader and audit its clocks."""
    row = read_inputs(path)
    targets = np.asarray(row['efit_times'], dtype=float)
    shot = aligned_shot(row, path.stem, targets)
    window = discharge_window(row)
    corrected = machine.plasma_current_times(row)
    is_d3 = str(row['source']) == 'DIII-D'
    if is_d3:
        assert np.array_equal(corrected, fix_d3d_ip_times(row)), path
    core_channels = np.asarray(row['thomson_core_Te']).shape[1]
    ghost = core_channels == len(row['thomson_core_R']) + 1
    if ghost:
        raw_te = np.asarray(row['thomson_core_Te'], dtype=float)
        raw_ne = np.asarray(row['thomson_core_ne'], dtype=float)
        assert np.isnan(raw_te[:, 0]).all() and np.isnan(raw_ne[:, 0]).all()
        core_channels -= 1
    paired = np.isfinite(shot.thomson.te) & np.isfinite(shot.thomson.ne)
    paired &= (shot.thomson.te > 0) & (shot.thomson.ne > 0)
    core = paired[:, :core_channels].sum(axis=1)
    edge = paired[:, core_channels:].sum(axis=1)
    core_ok, core_age = pulse_support(np.asarray(row['thomson_core_times'], dtype=float), targets)
    edge_ok, edge_age = pulse_support(np.asarray(row['thomson_edge_times'], dtype=float), targets)
    temporal = ((core == 0) | core_ok) & ((edge == 0) | edge_ok)
    magnitude = np.abs(shot.plasma_current)
    share = magnitude / window.peak_current
    running = np.isfinite(share) & (share >= RUNNING_SHARE)
    running &= (targets >= window.start_ms) & (targets <= window.end_ms)
    toroidal = np.isfinite(shot.tf_current) & (shot.tf_current != 0)
    reliable = running & toroidal & temporal & ((core + edge) >= 5)
    arrays = {'reliable': reliable, 'running': running, 'temporal_supported': temporal,
              'times_ms': targets, 'Ip_A': shot.plasma_current, 'current_share': share,
              'TF_A': shot.tf_current, 'core_age_ms': core_age, 'edge_age_ms': edge_age,
              'paired_channels': core + edge}
    original = np.asarray(row['magnetics_plasma_current_times'], dtype=float) if is_d3 else corrected
    record = {'stem': path.stem, 'source': str(row['source']), 'frames': len(targets),
              'reliable': int(reliable.sum()), 'not_running': int((~running).sum()),
              'unsupported_time': int((~temporal).sum()), 'too_few_pairs': int(((core + edge) < 5).sum()),
              'clock_parity_official': True if is_d3 else None,
              'clock_shift_ms': float(corrected[0] - original[0]), 'core_ghost_channel_dropped': ghost,
              'peak_current_A': window.peak_current, 'window_ms': [window.start_ms, window.end_ms]}
    return arrays, record


def factor(path: Path, features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the existing eight-feature pressure factor representation."""
    with np.load(path, allow_pickle=False) as model:
        z = np.clip((features - model['feature_mean']) / model['feature_scale'], -8, 8)
        y = np.column_stack([np.ones(len(z)), z]) @ model['coefficients']
        return np.exp(np.clip(y, np.log(model['factor_bounds'][0]), np.log(model['factor_bounds'][1])))
