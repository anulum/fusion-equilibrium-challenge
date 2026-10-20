# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — observe the unchanged native finisher
"""Capture maps at actual native call boundaries without replacing their arithmetic."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import cast

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
STAGES = ("raw_64coords", "kept_45coords", "smoothed", "axis", "contour",
          "shape", "x_points", "reshape", "margin")


def _recording(
    original: Callable[..., object], maps: dict[str, F64], kind: str,
    counts: dict[str, int],
) -> Callable[..., object]:
    def observed(*args: object, **kwargs: object) -> object:
        output = original(*args, **kwargs)
        counts[kind] = counts.get(kind, 0) + 1
        name = kind
        if kind == "shape":
            name = "shape" if counts[kind] == 1 else "reshape"
        if name in maps:
            raise ValueError(f"Unexpected repeated native stage: {name}")
        value = output[0] if isinstance(output, tuple) else output
        maps[name] = np.asarray(value, dtype=np.float64).copy()
        return output
    return observed


@contextmanager
def observe(
    module: ModuleType, basis: object, coordinates: F64,
) -> Iterator[dict[str, F64]]:
    """Observe every native map stage and restore all functions on exit.

    Parameters
    ----------
    module : types.ModuleType
        The exact qualified predict_diii_d namespace.
    basis : object
        Its loaded 64-coordinate basis.
    coordinates : numpy.ndarray
        Unedited ensemble coordinates, copied before use.

    Yields
    ------
    dict
        Maps in the frozen chronological stage order.
    """
    decode = cast(Callable[[F64], F64], getattr(basis, "decode"))
    kept = coordinates.copy()
    kept[:, 45:] = 0.
    maps = {"raw_64coords": decode(coordinates.copy()), "kept_45coords": decode(kept)}
    hooks = [(type(basis), "decode", "smoothed"), (module, "moved_onto", "axis"),
             (module, "_contour_bent", "contour"), (module, "edited", "shape"),
             (module, "edited_x_points", "x_points"),
             (module, "_kept_within_margin", "margin")]
    saved: list[tuple[object, str, Callable[..., object]]] = []
    counts: dict[str, int] = {}
    try:
        for owner, attribute, name in hooks:
            function = cast(Callable[..., object], getattr(owner, attribute))
            saved.append((owner, attribute, function))
            setattr(owner, attribute, _recording(function, maps, name, counts))
        yield maps
        if tuple(maps) != STAGES or counts.get("shape") != 2:
            raise ValueError(f"Native stage sequence changed: {tuple(maps)}, {counts}")
    finally:
        for owner, attribute, function in reversed(saved):
            setattr(owner, attribute, function)
