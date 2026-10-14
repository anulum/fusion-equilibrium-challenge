# SPDX-License-Identifier: MIT
# Vendored from the Fusion Equilibrium Challenge starter kit
# (© 2026 Sophelio and the Fusion Equilibrium Challenge authors, MIT — see LICENSE beside this file)

"""The organisers' derivation of the seven ψ-derived scalars, kept verbatim.

The scorer reads ``R_axis``, ``Z_axis``, ``kappa``, ``tri_top``, ``tri_bot``, ``volume`` and
``li`` off the submitted map with these functions, and off the true map with the same ones. A
network output fitted to what these functions read off the true maps is fitted to the quantity
that is scored, and an edit of a map that is measured with them lands where the scorer will
read it. Only the imports were changed, to relative ones; the sources are the starter kit's
``fusion_scoring/{common,contour,derive,lcfs,o_point}.py`` with sha256 prefixes 4c9618aa,
20bebb75, 0484e633, 71130f63 and 531f8dc2. ``metrics.py`` is not needed here and not copied.
"""
from .common import AXIS_SIGN, CONS_SCALARS
from .derive import _points_in_polygon, derive_frame, internal_inductance
from .lcfs import extract_lcfs

points_in_polygon = _points_in_polygon
"""The scorer's own membership test, the one its volume and inductance integrate over."""

__all__ = [
    "AXIS_SIGN", "CONS_SCALARS", "derive_frame", "extract_lcfs", "internal_inductance",
    "points_in_polygon",
]
