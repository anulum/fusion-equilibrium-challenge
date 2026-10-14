# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — vessel envelope for the Challenge machines

"""Load the region of the grid a confined plasma can occupy.

A flux surface that leaves the vacuum vessel is not a flux surface of the
confined plasma, so finding the boundary needs to know where the vessel is.
The Challenge publishes that envelope per machine as part of its scorer, which
makes it machine geometry rather than an answer: it is the same mask for every
shot and it says nothing about any particular equilibrium.

The mask is cached per machine because it is loaded once per shot otherwise,
and it never changes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

ENVELOPE_FILES: Final = {"DIII-D": "d3d_envelope.npz", "MAST": "mast_envelope.npz"}
"""Envelope file published for each machine."""

DEFAULT_MASK_DIR = Path(__file__).resolve().parents[1] / "vendor" / "challenge_masks"
"""The copies of the scorer's mask files kept with this package; see ``vendor/PROVENANCE.md``."""


@lru_cache(maxsize=4)
def envelope(machine: str) -> NDArray[np.bool_]:
    """Return the vessel envelope mask for one machine.

    Parameters
    ----------
    machine:
        Machine identifier as the dataset records it, such as ``"DIII-D"``.

    Raises
    ------
    KeyError
        If the machine has no published envelope.
    FileNotFoundError
        If the mask directory does not hold it. The location can be overridden
        with ``FEC_MASK_DIR`` so the adapter is not tied to one checkout layout.
    """
    directory = Path(os.environ.get("FEC_MASK_DIR", str(DEFAULT_MASK_DIR)))
    path = directory / ENVELOPE_FILES[machine]
    if not path.is_file():
        raise FileNotFoundError(f"vessel envelope not found: {path}")
    with np.load(path) as bundle:
        return np.asarray(bundle["mask_coarse"], dtype=bool)


@lru_cache(maxsize=4)
def grids(machine: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``R`` and ``Z`` grid coordinates the envelope of a machine is given on."""
    directory = Path(os.environ.get("FEC_MASK_DIR", str(DEFAULT_MASK_DIR)))
    path = directory / ENVELOPE_FILES[machine]
    if not path.is_file():
        raise FileNotFoundError(f"vessel envelope not found: {path}")
    with np.load(path) as bundle:
        return (np.asarray(bundle["grid_R"], dtype=np.float64),
                np.asarray(bundle["grid_Z"], dtype=np.float64))
