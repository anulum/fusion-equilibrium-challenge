# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — one shot solved and corrected in a single call

"""Solve a shot and add the fitted correction to it, without a stored base in between.

:mod:`fec_adapter.cross_machine.build` works from bases stored by an earlier
run, which is the economical route for a whole test set. This is the same
prediction in one call, for a shot that has no stored base, and the place where
the two routes are held to the same answer.
"""
from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from fec_adapter import batched_equilibrium, profiles
from fec_adapter.orientation import flux_orientation
from fec_adapter.physics_base import PhysicsBase, physics_fingerprint
from fec_adapter.predict import DEFAULT_SMOOTH_FRAMES, report_statuses
from fec_adapter.shot_reader import Shot
from fec_adapter.temporal import smooth_in_time

from .correction import CorrectionBundle


def predict_shot_corrected(
    shot: Shot,
    model: profiles.ProfileModel,
    bundle: CorrectionBundle,
    smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(psi, q95, betaN)``: the physics map plus the learned difference.

    The two scalars come from the bundle's own heads. The bundle refuses a
    solve made with other physics settings than it was fitted on.
    """
    started = time.monotonic()
    settings = batched_equilibrium.PipelineSettings()
    solution = batched_equilibrium.solve_shot(shot, model, settings)
    base = PhysicsBase.from_solution(solution, flux_orientation(shot.plasma_current))
    psi, q95, beta_n = bundle.apply(
        shot, base, physics_fingerprint(model, settings, shot.source)
    )
    report_statuses(shot, solution, started)
    return smooth_in_time(psi, smooth_frames), q95, beta_n
