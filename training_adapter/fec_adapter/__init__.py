# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — Fusion Equilibrium Challenge adapter

"""Predict tokamak equilibria for the Fusion Equilibrium Challenge.

The Challenge asks for a flux map and two scalars from actuators and kinetic
profiles, with the magnetic diagnostics withheld. This package answers it by
solving the free-boundary Grad-Shafranov equation rather than by regressing
the answer: the coil currents drive the vacuum field, the Thomson profiles fix
the pressure, the measured plasma current closes the current profile, and the
equilibrium follows.
"""
from __future__ import annotations

__all__ = ["derived", "equilibrium", "machine", "profiles", "shot_reader"]
