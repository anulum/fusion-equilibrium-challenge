# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — four-position specialist native integration
"""Blend already tree-adjusted X targets and restore native state after each call."""
from contextlib import contextmanager

import numpy as np

LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')


@contextmanager
def use_x_specialist(pd, predictions, share, half_window):
    share = float(share)
    if not np.isfinite(share) or not 0. <= share <= 1.:
        raise ValueError('X contribution must be finite and lie in [0,1]')
    if half_window < 0:
        raise ValueError('Resolve the native axis smoothing window before intervention')
    state = {'calls': 0, 'zero_contribution_bypass': share == 0.}
    if share == 0.:
        yield state
        return
    original = pd.edited_x_points

    def wrapped(maps, targets, source, **options):
        state['calls'] += 1
        assert state['calls'] == 1, 'one actual native X edit expected per shot'
        assert set(targets) == set(LABELS)
        supplied = predictions(targets) if callable(predictions) else predictions
        assert set(supplied) == set(LABELS)
        mixed = {}
        for name in LABELS:
            base = np.asarray(targets[name], dtype=float)
            expert = np.asarray(supplied[name], dtype=float)
            assert expert.shape == base.shape and expert.ndim == 1 and np.isfinite(expert).all()
            # Native target smoothing has already been applied to the baseline.
            expert = pd._smoothed(expert, half_window)
            if np.array_equal(expert, base):
                mixed[name] = targets[name]
            elif share == 1.:
                mixed[name] = expert
            else:
                mixed[name] = (1.-share)*base+share*expert
        return original(maps, mixed, source, **options)

    pd.edited_x_points = wrapped
    try:
        yield state
        assert state['calls'] == 1
    finally:
        pd.edited_x_points = original
