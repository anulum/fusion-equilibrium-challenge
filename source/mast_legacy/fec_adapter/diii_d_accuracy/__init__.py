# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the DIII-D accuracy entry (Challenge 1)

"""The entry for the DIII-D award: the highest composite score on one machine.

DIII-D comes with thousands of labelled discharges, so this entry learns the
reconstruction from the allowed inputs directly and is free to use every
labelled shot, which a model that needs a physics solve per shot cannot afford.

Nothing here may import :mod:`fec_adapter.cross_machine`, and nothing there may
import this package; ``tests/test_package_boundaries.py`` holds both to it.
What the two entries share sits in the package root.
"""
