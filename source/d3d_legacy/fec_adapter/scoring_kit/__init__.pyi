# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typing surface of the vendored derivation: the names this package calls."""
import numpy as np
from numpy.typing import NDArray

AXIS_SIGN: dict[str, int]
CONS_SCALARS: list[str]

def extract_lcfs(
    psi: NDArray[np.float64], R: NDArray[np.float64], Z: NDArray[np.float64], machine: str,
    mask_coarse: NDArray[np.bool_] | None = ..., mask_f: NDArray[np.float64] | None = ...,
    n_points: int = ..., n_iter: int = ...,
) -> NDArray[np.float64] | None: ...
def derive_frame(
    psi: NDArray[np.float64], R: NDArray[np.float64], Z: NDArray[np.float64], machine: str,
    mask_coarse: NDArray[np.bool_] | None = ..., mask_f: NDArray[np.float64] | None = ...,
    contour: NDArray[np.float64] | str | None = ..., n_points: int = ...,
    with_li: bool = ..., axis_sign: int | None = ...,
) -> dict[str, float]: ...
def points_in_polygon(
    rq: NDArray[np.float64], zq: NDArray[np.float64], C: NDArray[np.float64],
) -> NDArray[np.bool_]: ...
def internal_inductance(
    psi: NDArray[np.float64], R: NDArray[np.float64], Z: NDArray[np.float64], machine: str,
    C: NDArray[np.float64] | None,
) -> float: ...
