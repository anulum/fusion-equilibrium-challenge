# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the names of the accuracy entry's feature layouts

"""Name what a network's feature columns are measured against.

Two layouts have the same columns in the same order and differ in what the
phase columns and the trailing means are measured against: the reconstruction
record alone (:data:`RECORD_LAYOUT`) or the discharge around it
(:data:`DISCHARGE_LAYOUT`, :mod:`.history`). A network reads the numbers of the
layout it was fitted on and no other, so the layout travels with the table a
network is fitted from and is stored in the network; a network file without it
predates the second layout and is record-relative.
"""
from __future__ import annotations

from typing import Final, Literal

FeatureLayout = Literal["record", "discharge"]
"""What a network's feature columns are measured against."""

RECORD_LAYOUT: Final[FeatureLayout] = "record"
"""Phase and means over the reconstruction frames alone: every network fitted before 22 Sep 2026."""

DISCHARGE_LAYOUT: Final[FeatureLayout] = "discharge"
"""Phase against the discharge window, means over the discharge record's base."""

LAYOUTS: Final[tuple[FeatureLayout, ...]] = (RECORD_LAYOUT, DISCHARGE_LAYOUT)
"""Every layout a network or a table may declare."""


def layout_of(value: object) -> FeatureLayout:
    """Return ``value`` as a feature layout, refusing anything that is not one.

    Raises
    ------
    ValueError
        If ``value`` names no known layout.
    """
    for layout in LAYOUTS:
        if value == layout:
            return layout
    raise ValueError(f"unknown feature layout {value!r}; known: {LAYOUTS}")
