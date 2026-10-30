# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — original scored-map numerical dependencies
"""Load the published frozen operators without changing their numerical source.

The maintained boundary checks source identities and restores historical import
aliases. The imported experiment implementations retain their archival scope.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Mask = NDArray[np.bool_]
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "experiments/d71_finisher_physics/physics.py": (
        "31094bce283aeafd5b32c27ef2cba83ebf6de7144fa23b97126b0b350b60f249"
    ),
    "experiments/d71b_physics_amendment/critical.py": (
        "e83d80df73ee3fdc00724519e7ed00faa2f373584f0de25857c576abd8f977b7"
    ),
    "experiments/d71b_physics_amendment/physics_b.py": (
        "428dd72e7a624a9a44454b6aecd7da4481d6b81ddb5a4d226cb3e9f63c083c86"
    ),
    "source/d3d_legacy/fec_adapter/scoring_kit/derive.py": (
        "11856266015ebec9e8fd96a82d110107cceedcce5085cd4b0ba9dd437b00df9c"
    ),
    "source/d3d_legacy/fec_adapter/scoring_kit/o_point.py": (
        "531f8dc24f54e7cd1516fa365fd412e4421d1c086d170682c00ba964304315e4"
    ),
}


Derivatives = Callable[[F64, F64, F64], tuple[F64, F64, F64]]
Inside = Callable[[F64 | None, F64, F64, object], Mask]
Dilate = Callable[[Mask, int], Mask]
Density = Callable[[F64, F64, F64, float], F64]
Measure = Callable[
    [F64, F64, F64, F64 | None, F64 | None, float, Mask, dict[str, object], Mask, float],
    dict[str, float],
]


@dataclass(frozen=True)
class Physics:
    """Original finite-difference and region operations, without wrappers."""

    derivatives: Derivatives
    inside: Inside
    dilate: Dilate
    MIN_PIXELS: int
    EDGE_CELLS: int


@dataclass(frozen=True)
class PhysicsB:
    """Original corrected measurements, retaining their numerical domains."""

    density: Density
    measure_b: Measure


@dataclass(frozen=True)
class Operators:
    """Bound legacy modules and the exact published scorer primitives."""

    physics: Physics
    corrected: PhysicsB
    kit: dict[str, object]
    source_sha256: dict[str, str]


def load() -> Operators:
    """Load the original public measurement dependencies under checked identities.

    Returns
    -------
    Operators
        Original operators and their source checksums. Historical top-level
        import names are restored before returning, including on load failure.

    Raises
    ------
    OSError, ValueError
        A required published dependency is absent or differs from its binding.
    """
    for name, checksum in SOURCES.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Published measurement dependency differs")
    derive = import_module("source.d3d_legacy.fec_adapter.scoring_kit.derive")
    o_point = import_module("source.d3d_legacy.fec_adapter.scoring_kit.o_point")
    names = ("physics", "critical", "physics_b")
    previous = {name: sys.modules.get(name) for name in names}
    modules: dict[str, ModuleType] = {}
    try:
        for name, path in zip(names, list(SOURCES)[:3], strict=True):
            module = ModuleType(name)
            module.__file__ = str(ROOT / path)
            modules[name] = sys.modules[name] = module
            SourceFileLoader(name, module.__file__).exec_module(module)
    finally:
        for name in names:
            saved = previous[name]
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
    return Operators(
        Physics(
            cast(Derivatives, modules["physics"].derivatives),
            cast(Inside, modules["physics"].inside),
            cast(Dilate, modules["physics"].dilate),
            int(modules["physics"].MIN_PIXELS),
            int(modules["physics"].EDGE_CELLS),
        ),
        PhysicsB(
            cast(Density, modules["physics_b"].density),
            cast(Measure, modules["physics_b"].measure_b),
        ),
        {
            "points_in_polygon": derive._points_in_polygon,
            "bilinear": derive._bilinear,
            "find_o_point": o_point.find_o_point,
            "erode": o_point._erode,
        },
        SOURCES.copy(),
    )
