# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — package shim for the vendored solver modules

"""Namespace shim for the vendored equilibrium solver subpackage.

This file is **not** a copy of the upstream package initialiser. The real one
imports the whole of SCPN Fusion Core, which would drag in every dependency
that package has for the sake of eleven modules. The solver modules themselves
are byte-identical copies, so re-syncing them is a plain file copy; only these
two shims are ours.
"""
