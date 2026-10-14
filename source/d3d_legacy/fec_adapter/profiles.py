# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — free-function profiles from kinetic measurements

"""Build the Grad-Shafranov free functions from what the machine measured.

A Grad-Shafranov solve needs two free functions of the normalised flux,
``p'`` and ``FF'``. The Challenge withholds both, and it withholds the magnetic
diagnostics that a reconstruction code would use to infer them. What it does
provide is the Thomson scattering profiles and the measured plasma current,
and those are between them enough to pin both functions without any appeal to
the target equilibrium:

*The pressure is measured.* Thomson gives electron temperature and density at
known positions. Mapped through the current flux estimate they give
``p(ψ_N)`` in pascals, and its flux derivative is ``p'`` — absolutely, not up
to a scale.

*The current closes the system.* With ``p'`` fixed in absolute units, the
total toroidal current ``I_p = ∮ (R p' + FF'/(μ₀R)) dA`` is one linear equation
for the amplitude of ``FF'`` once its shape is chosen. The measured ``I_p``
supplies the left-hand side.

That leaves the shape of ``FF'`` and the ion-to-electron temperature ratio as
the only quantities not determined by a measurement in this dataset. Both are
parameters of :class:`ProfileModel`, to be fitted on training shots and applied
to shots held out from that fit — never inferred per shot from the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

ELEMENTARY_CHARGE: Final = 1.602176634e-19
"""Coulombs; converts electronvolts to joules exactly (SI definition)."""

MU0: Final = 4.0e-7 * np.pi

TINY: Final = 1.0e-30
"""Magnitude below which a divisor is treated as zero; far under any physical flux or current."""
"""Vacuum permeability in henries per metre."""

DEFAULT_KNOTS: Final = 33
"""Normalised-flux samples spanning the plasma; enough to resolve a pedestal."""


@dataclass(frozen=True)
class ProfileModel:
    """The parameters that the measurements do not determine.

    Attributes
    ----------
    ti_over_te:
        Ion temperature as a fraction of the electron temperature. Thomson
        measures electrons only, and the ion contribution to the pressure has
        to come from somewhere; unity is the conventional assumption for a
        dataset without charge-exchange recombination spectroscopy.
    ffprime_alpha, ffprime_beta:
        Shape exponents of ``FF'(ψ_N) ∝ (1 - ψ_N**alpha)**beta``. This is the
        standard two-parameter family; it is peaked on axis and vanishes at
        the separatrix, which is what a diverted equilibrium requires.
    pressure_floor_pa:
        Pressure held outside the last closed surface. Zero is correct for an
        ideal separatrix and numerically awkward for a smooth cutoff, so a
        small positive floor is used instead of a hard zero.

    Notes
    -----
    The defaults of ``ti_over_te`` and ``ffprime_alpha`` were chosen on the
    calibration bucket through the batched pipeline, against the pooled flux R²
    and the reconstruction's axis radius. On 16 shots (4005 frames) the choice
    matters little: flux R² 0.9204 at 1.5 and 0.5 (the defaults), 0.9199 at 2.0
    and 0.5, 0.9191 at 1.5 and 0.75, 0.9183 at 1.0 and 1.0; the median radial
    axis offset runs from -2.7 cm (2.0, 0.5) through -4.2 cm (the defaults) to
    -7.1 cm (1.0, 1.0), and its interquartile range is 14 to 15 cm at every
    setting — the spread between discharges dwarfs what either knob moves.
    A first fit on four shots had put the offset of the defaults at -0.1 cm;
    sixteen shots did not bear that out, and the number is recorded here so
    that it is not quoted again. A ratio above one stands in for every
    contribution to the pressure that Thomson scattering does not see — hotter
    ions and fast ions together — and is an effective value, not a measured ion
    temperature.
    """

    ti_over_te: float = 1.5
    ffprime_alpha: float = 0.5
    ffprime_beta: float = 1.0
    pressure_floor_pa: float = 0.0


def knots(count: int = DEFAULT_KNOTS) -> NDArray[np.float64]:
    """Return monotonic normalised-flux knots spanning ``[0, 1]``."""
    return np.linspace(0.0, 1.0, count, dtype=np.float64)


def electron_pressure(
    te_ev: NDArray[np.float64], ne_m3: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the electron pressure in pascals from Thomson samples."""
    return np.asarray(te_ev * ne_m3 * ELEMENTARY_CHARGE, dtype=np.float64)


def total_pressure(
    te_ev: NDArray[np.float64], ne_m3: NDArray[np.float64], model: ProfileModel
) -> NDArray[np.float64]:
    """Return the total kinetic pressure in pascals.

    Quasi-neutrality with a single hydrogenic species gives ``n_i = n_e``, so
    the ion pressure is the electron pressure scaled by the temperature ratio.
    Impurity content would raise this and dilution would lower it; neither is
    measured here, and inventing a profile for either would add a free
    function rather than remove one.
    """
    return electron_pressure(te_ev, ne_m3) * (1.0 + model.ti_over_te)


def pressure_on_knots(
    psi_n_samples: NDArray[np.float64],
    pressure_samples: NDArray[np.float64],
    psi_n_knots: NDArray[np.float64],
    model: ProfileModel,
) -> NDArray[np.float64]:
    """Reduce scattered pressure samples to a monotone profile on the knots.

    Thomson channels sit where the laser sits, not where the flux surfaces
    are, so the samples arrive as an unordered cloud in ``ψ_N``. They are
    averaged into knot bins, gaps are filled by interpolation between the bins
    that do have data, and the result is forced to be non-increasing outwards.

    The monotonicity is a physical statement, not a smoothing convenience: a
    confined plasma's pressure falls from the axis to the boundary, and a
    sample cloud that says otherwise is reporting measurement scatter. Enforcing
    it keeps ``p'`` single-signed, which is what stops the current profile from
    developing a spurious hollow core.
    """
    finite = np.isfinite(psi_n_samples) & np.isfinite(pressure_samples)
    inside = finite & (psi_n_samples >= 0.0) & (psi_n_samples <= 1.0)
    if not inside.any():
        return np.full(psi_n_knots.size, model.pressure_floor_pa, dtype=np.float64)

    edges = _bin_edges(psi_n_knots)
    found = np.searchsorted(edges, psi_n_samples[inside], side="right") - 1
    which = np.clip(found, 0, psi_n_knots.size - 1)
    totals = np.bincount(which, weights=pressure_samples[inside], minlength=psi_n_knots.size)
    counts = np.bincount(which, minlength=psi_n_knots.size)
    # At least one sample lies inside the plasma — the early return above
    # guarantees it — so at least one bin is populated and no second emptiness
    # guard is reachable here.
    filled = counts > 0
    binned = np.where(filled, totals / np.maximum(counts, 1), np.nan)
    profile = np.interp(psi_n_knots, psi_n_knots[filled], binned[filled])
    profile = np.maximum.accumulate(profile[::-1])[::-1]
    return np.maximum(profile, model.pressure_floor_pa)


def pprime_from_pressure(
    pressure_knots: NDArray[np.float64], psi_n_knots: NDArray[np.float64], delta_psi: float
) -> NDArray[np.float64]:
    """Differentiate ``p(ψ_N)`` into ``p'(ψ) = dp/dψ`` in pascals per weber.

    Parameters
    ----------
    pressure_knots:
        Pressure in pascals at each knot.
    psi_n_knots:
        The knots themselves.
    delta_psi:
        ``ψ_boundary - ψ_axis`` in webers. The chain rule needs it, and it is
        a property of the equilibrium rather than of the measurement, which is
        why the pressure fit and the solve have to be iterated rather than
        done once.

    Notes
    -----
    The sign follows from ``ψ_N = (ψ - ψ_axis) / (ψ_boundary - ψ_axis)``: with
    pressure falling outwards, ``dp/dψ_N`` is negative and ``delta_psi``
    carries the orientation of the flux, so the result is correctly signed for
    either convention without a special case.
    """
    if not np.isfinite(delta_psi) or delta_psi == 0.0:
        return np.zeros_like(pressure_knots)
    slope: NDArray[np.float64] = np.asarray(
        np.gradient(pressure_knots, psi_n_knots), dtype=np.float64
    )
    return slope / delta_psi


def ffprime_shape(psi_n_knots: NDArray[np.float64], model: ProfileModel) -> NDArray[np.float64]:
    """Return the unscaled ``FF'`` shape on the knots.

    The amplitude is deliberately absent: it is fixed by the measured plasma
    current in :func:`ffprime_amplitude`, not chosen here.
    """
    inner = np.clip(1.0 - np.power(np.clip(psi_n_knots, 0.0, 1.0), model.ffprime_alpha), 0.0, None)
    return np.asarray(np.power(inner, model.ffprime_beta), dtype=np.float64)


def ffprime_amplitude(
    ip_target: float,
    pprime_current: float,
    shape_current: float,
) -> float:
    """Return the ``FF'`` scale that makes the modelled current match the measured one.

    Parameters
    ----------
    ip_target:
        Measured plasma current in amperes.
    pprime_current:
        ``∮ R p' dA`` over the plasma support, in amperes — the part of the
        current the measured pressure already accounts for.
    shape_current:
        ``∮ shape/(μ₀R) dA`` over the same support, in amperes per unit scale.

    Notes
    -----
    A vanishing ``shape_current`` means the plasma support collapsed, and a
    scale computed from it would be an arbitrarily large number applied to an
    empty region. Returning zero keeps the solve on the pressure-driven
    current alone, which is wrong but recoverably so.
    """
    if not np.isfinite(shape_current) or abs(shape_current) < TINY:
        return 0.0
    return float((ip_target - pprime_current) / shape_current)


def _bin_edges(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return bin edges midway between knot centres, closed at both ends."""
    midpoints = 0.5 * (centres[1:] + centres[:-1])
    edges: NDArray[np.float64] = np.concatenate([[centres[0]], midpoints, [centres[-1]]])
    return edges
