# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the cross-machine entry (Challenge 2)

"""The entry for the cross-machine award: one method, scored on DIII-D and on MAST.

The award ranks ``S(MAST) / S(DIII-D)`` among entries whose DIII-D composite
reaches 0.85, and MAST has no training set at all. What carries over to a
machine nobody has labels for is the physics solve, so this entry is built on
it: the solve is the same on both machines, and on DIII-D a correction fitted
to that machine's reconstructions lifts the composite over the gate.

Nothing here may import :mod:`fec_adapter.diii_d_accuracy`, and nothing there
may import this package; ``tests/test_package_boundaries.py`` holds both to it.
What the two entries share sits in the package root.
"""
