# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — physical current-profile controls
"""Exercise physical normalization, signs, release geometry and the native adapter."""

from __future__ import annotations

import numpy as np
import pytest
from freegs4e.jtor import ConstrainPaxisIp  # type: ignore[import-untyped]

from tools.constructive_model import (
    FixedPressureProfile,
    Profile,
    current_source,
    flux_functions,
    make_equilibrium,
)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("power", [1.0, 2.0, 4.0])
def test_pressure_current_and_flux_derivatives(sign: float, power: float) -> None:
    """Preserve pressure and signed Ip while changing only FF-prime shape."""
    r, z = np.meshgrid(np.linspace(1, 2.4, 65), np.linspace(-1, 1, 65), indexing="ij")
    s = ((r - 1.7) / 0.6) ** 2 + (z / 0.9) ** 2
    flux = sign * 0.3 * (1 - s)
    mask = (s < 1).astype(float)
    area = float((r[1, 0] - r[0, 0]) * (z[0, 1] - z[0, 0]))
    profile = Profile(current=sign * 1e6, power=power)
    current, evaluated = current_source(profile, r, flux, sign * 0.3, 0, mask, area)
    assert np.sum(current) * area == pytest.approx(profile.current, rel=2e-15)
    assert np.all(current[mask == 0] == 0)
    positions = np.linspace(0, 1, 1001)
    p, pprime, ffprime, fpol = flux_functions(evaluated, positions)
    np.testing.assert_allclose(p, 20_000 * (1 - positions) ** 3)
    np.testing.assert_allclose(np.gradient(p, positions)[1:-1] / evaluated.span, pprime[1:-1], atol=0.07)
    np.testing.assert_allclose(
        np.gradient(fpol**2 / 2, positions)[1:-1] / evaluated.span,
        ffprime[1:-1],
        atol=4e-5,
    )
    assert fpol[-1] == pytest.approx(profile.fvac)
    if power == 2:
        native = ConstrainPaxisIp(profile.paxis, profile.current, profile.fvac)
        control = native.Jtor_part2(r, z, flux, sign * 0.3, 0, mask)
        np.testing.assert_allclose(current, control, rtol=2e-14, atol=1e-8)


@pytest.mark.parametrize(
    "profile",
    [
        {"paxis": -1.0},
        {"current": 0.0},
        {"fvac": 0.0},
        {"power": float("nan")},
    ],
)
def test_physical_parameter_refusal(profile: dict[str, float]) -> None:
    """Reject invalid profiles before an upstream object can be initialized."""
    with pytest.raises(ValueError):
        Profile(**profile)


@pytest.mark.parametrize("fault", ["shape", "radius", "flux", "mask", "span", "area", "empty"])
def test_current_source_refusal(fault: str) -> None:
    """Refuse malformed maps and degenerate core support with actual arrays."""
    r = np.ones((4, 4))
    flux = np.zeros_like(r)
    mask = np.ones_like(r)
    axis, boundary, area = 1.0, 0.0, 0.01
    if fault == "shape":
        flux = np.zeros((3, 4))
    elif fault == "radius":
        r[0, 0] = 0
    elif fault == "flux":
        flux[0, 0] = np.inf
    elif fault == "mask":
        mask[0, 0] = 0.3
    elif fault == "span":
        boundary = axis
    elif fault == "area":
        area = -1
    else:
        mask[:] = 0
    with pytest.raises(ValueError):
        current_source(Profile(), r, flux, axis, boundary, mask, area)


def test_flux_function_refusal() -> None:
    """Reject unevaluated profiles, invalid coordinates and nonphysical F squared."""
    with pytest.raises(ValueError):
        flux_functions(Profile(), np.asarray([0.5]))
    with pytest.raises(ValueError):
        flux_functions(Profile(span=1, amplitude=0), np.asarray([-0.1]))
    with pytest.raises(ValueError, match="F squared"):
        flux_functions(Profile(span=1, amplitude=100), np.asarray([0.0]))


@pytest.mark.parametrize("initial", [0, 1])
def test_native_geometry_and_profile_adapter(initial: int) -> None:
    """Exercise the real external constructor and profile, without fitting a candidate."""
    eq = make_equilibrium(17, initial, np.zeros(18))
    profile = FixedPressureProfile(eq, Profile())
    r, z = eq.R, eq.Z
    s = ((r - 1.7) / 0.5) ** 2 + (z / 0.8) ** 2
    mask = (s < 1).astype(float)
    current = profile.Jtor_part2(r, z, 0.3 * (1 - s), 0.3, 0, mask)
    area = (r[1, 0] - r[0, 0]) * (z[0, 1] - z[0, 0])
    assert np.sum(current) * area == pytest.approx(1e6)
    knots = np.linspace(0, 1, 17)
    expected = flux_functions(profile.physical, knots)
    for actual, control in zip(
        (profile.pressure(knots), profile.pprime(knots), profile.ffprime(knots), profile.fpol(knots)),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, control)
    copied = profile.copy()
    copied.jtor[0, 0] = 123
    assert current[0, 0] == 0
    assert np.array_equal(eq.tokamak.current_vec, np.zeros(18))
    assert np.isfinite(eq.psi()).all()


@pytest.mark.parametrize("size,initial,currents", [(18, 0, 18), (17, 2, 18), (17, 0, 17)])
def test_native_constructor_refusal(size: int, initial: int, currents: int) -> None:
    """Refuse unsupported grid or coil state without starting a solve."""
    with pytest.raises(ValueError):
        make_equilibrium(size, initial, np.zeros(currents))


def test_ambient_equilibrium_path_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a real environment variable from overriding the declared initial guess."""
    monkeypatch.setenv("EQUILIBRIUM_PATH", "/not-an-authorized-initial-equilibrium")
    with pytest.raises(ValueError, match="EQUILIBRIUM_PATH"):
        make_equilibrium(17, 0, np.zeros(18))
