# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — synthetic observation and calibration controls
"""Exercise physical channels, calibration floors and fixed-history compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from tools.constructive_observations import (
    DENSITY_AXIS,
    TEMPERATURE_AXIS,
    channels,
    compare_observations,
    minimum_allowance,
    observe,
)


def test_fixed_channels_on_analytic_surface() -> None:
    """Recover a quadratic flux surface's prescribed density and temperature."""
    r, z = np.linspace(0.1, 2.8, 129), np.linspace(-1.8, 1.8, 129)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    s = ((rr - 1.7) / 0.6) ** 2 + (zz / 1.0) ** 2
    points = channels()
    actual = observe(r, z, 0.3 * (1 - s), np.ones_like(s), 0.3, 0)
    channel_s = ((points[:, 0] - 1.7) / 0.6) ** 2 + points[:, 1] ** 2
    distance = np.maximum(1 - channel_s, 0)
    np.testing.assert_allclose(actual[:98], DENSITY_AXIS * distance, rtol=1e-12, atol=1e5)
    np.testing.assert_allclose(actual[98:], TEMPERATURE_AXIS * distance**2, atol=1e-10)
    assert np.count_nonzero(actual[:98]) > 30
    outside = observe(r, z, 0.3 * (1 - s), np.zeros_like(s), 0.3, 0)
    assert not outside.any()


@pytest.mark.parametrize("fault", ["grid", "map", "order", "mask", "span", "coverage", "axis"])
def test_observation_refusal(fault: str) -> None:
    """Reject invalid maps without silently dropping any measurement channel."""
    r, z = np.linspace(0.1, 2.8, 9), np.linspace(-1.8, 1.8, 9)
    flux, mask = np.ones((9, 9)), np.ones((9, 9))
    axis, boundary = 1.0, 0.0
    if fault == "grid":
        r = r[:3]
    elif fault == "map":
        flux[0, 0] = np.nan
    elif fault == "order":
        z = z[::-1]
    elif fault == "mask":
        mask[0, 0] = 2
    elif fault == "span":
        boundary = axis
    elif fault == "coverage":
        r = r + 1
    else:
        axis = 0.5
    with pytest.raises(ValueError):
        observe(r, z, flux, mask, axis, boundary)


def test_calibration_and_stationary_history() -> None:
    """A shared midpoint and persistent biases reproduce both available histories."""
    reference = np.concatenate([np.full(98, DENSITY_AXIS), np.full(98, TEMPERATURE_AXIS)])
    candidate = reference * 1.03
    result = compare_observations(reference, candidate)
    assert result.primary_compatible
    assert result.sensitivity == (False, False, False, True, True)
    assert result.minimum_relative == pytest.approx((0.015 - 0.001) / 1.015)
    assert result.minimum_relative_without_floor == pytest.approx(0.015 / 1.015)
    np.testing.assert_array_equal(reference + result.reference_bias, result.midpoint)
    np.testing.assert_array_equal(candidate + result.candidate_bias, result.midpoint)
    assert np.all(np.abs(result.reference_bias) <= result.halfwidth)
    for timestamp in (0, 1000, 2000):
        assert 0 <= timestamp * 0.0005 <= 1
        np.testing.assert_array_equal(reference + result.reference_bias, candidate + result.candidate_bias)
    assert compare_observations(reference, reference * 1.2).primary_compatible is False


def test_zero_mean_and_absolute_floor() -> None:
    """Distinguish zero-over-zero from a positive discrepancy at zero midpoint."""
    zero = np.zeros(3)
    assert minimum_allowance(zero, zero, zero) == 0
    assert minimum_allowance(np.ones(3), zero, zero) == float("inf")
    assert minimum_allowance(np.ones(3), zero, np.ones(3)) == 0
    empty = np.zeros(196)
    assert compare_observations(empty, empty).minimum_relative == 0


@pytest.mark.parametrize("fault", ["shape", "negative", "infinite"])
def test_calibration_refusal(fault: str) -> None:
    """Refuse malformed observations before constructing any bias vector."""
    reference, candidate = np.ones(196), np.ones(196)
    if fault == "shape":
        candidate = candidate[:195]
    elif fault == "negative":
        reference[0] = -1
    else:
        candidate[0] = np.inf
    with pytest.raises(ValueError):
        compare_observations(reference, candidate)
    with pytest.raises(ValueError):
        minimum_allowance(np.asarray([-1.0]), np.asarray([1.0]), np.asarray([0.0]))
