# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — a linear basis for flux maps, fitted on every training frame

"""Fit a linear basis for the flux maps from every frame of the training part.

A flux map has 4225 cells, and a model that predicts them one by one learns the
same smooth field 4225 times. The maps of one machine live close to a space of
a few dozen dimensions, so the model predicts coordinates in that space and the
map is put back together from them.

The basis is the principal directions of the training maps. With a million and
a half frames the usual route — hold the frames, decompose them — does not fit
in memory and does not need to: the directions are the eigenvectors of the
4225 by 4225 second-moment matrix, and that matrix is a sum over frames, so it
is accumulated shot by shot, in parallel, and never more than one shot is held.
The result is exact, not a randomised or incremental approximation.

The coordinates are not whitened. With orthonormal directions a squared error
in coordinates *is* the squared error over the map, so a model fitted on the
coordinates minimises the quantity the score pools.

Only the training part goes in. The basis is a fitted thing like any other, and
a direction learned from a validation map would flatter every number quoted on
validation afterwards.
"""
from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Moments:
    """Sums over frames from which the mean map and the second moments follow.

    Attributes
    ----------
    frames:
        Number of frames summed.
    total:
        Sum of the flattened maps, shape ``(P,)``.
    outer:
        Sum of the outer products of the flattened maps, shape ``(P, P)``.
    """

    frames: int
    total: NDArray[np.float64]
    outer: NDArray[np.float64]

    def __add__(self, other: Moments) -> Moments:
        """Combine the sums of two disjoint sets of frames."""
        return Moments(
            self.frames + other.frames, self.total + other.total, self.outer + other.outer
        )


def moments_of(files: Sequence[Path]) -> Moments:
    """Sum the moments of the flux maps stored in ``files``, one shot in memory at a time."""
    frames, total, outer = 0, None, None
    for path in files:
        with np.load(path, allow_pickle=False) as stored:
            maps = stored["psi"].astype(np.float64)
        flat = maps.reshape(maps.shape[0], -1)
        if total is None or outer is None:
            total = np.zeros(flat.shape[1])
            outer = np.zeros((flat.shape[1], flat.shape[1]))
        frames += flat.shape[0]
        total += flat.sum(axis=0)
        outer += flat.T @ flat
    if total is None or outer is None:
        raise ValueError("no shot files to take moments of")
    return Moments(frames, total, outer)


def moments_in_parallel(files: Sequence[Path], workers: int) -> Moments:
    """Split the files between ``workers`` processes and add their moments up."""
    if not files:
        raise ValueError("no shot files to take moments of")
    share = [list(files[start::workers]) for start in range(min(workers, len(files)))]
    fresh = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(share), mp_context=fresh) as pool:
        parts = list(pool.map(moments_of, share))
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + part
    return combined


@dataclass(frozen=True)
class FluxBasis:
    """The mean map and the leading principal directions of the training maps.

    Attributes
    ----------
    mean:
        Mean flattened map, shape ``(P,)``.
    directions:
        Orthonormal directions, shape ``(K, P)``, by falling variance.
    variance:
        Variance along each direction, shape ``(K,)``.
    total_variance:
        Variance of the maps summed over all cells; ``variance.sum()`` over it
        is the share a truncation keeps, on the frames the basis was fitted on.
    grid_shape:
        ``(NZ, NR)`` of the maps.
    """

    mean: NDArray[np.float64]
    directions: NDArray[np.float64]
    variance: NDArray[np.float64]
    total_variance: float
    grid_shape: tuple[int, int]

    @classmethod
    def from_moments(cls, moments: Moments, size: int, grid_shape: tuple[int, int]) -> FluxBasis:
        """Take the ``size`` leading directions of the covariance the moments imply."""
        cells = moments.total.size
        if not 0 < size <= cells or grid_shape[0] * grid_shape[1] != cells:
            raise ValueError(
                f"cannot take {size} directions on a {grid_shape} grid of {cells} cells"
            )
        mean = moments.total / moments.frames
        covariance = moments.outer / moments.frames - np.outer(mean, mean)
        values, vectors = np.linalg.eigh(covariance)
        leading = np.argsort(values)[::-1][:size]
        return cls(
            mean=mean,
            directions=vectors[:, leading].T.copy(),
            variance=np.maximum(values[leading], 0.0),
            total_variance=float(np.trace(covariance)),
            grid_shape=grid_shape,
        )

    @property
    def kept_share(self) -> float:
        """Share of the fitted maps' variance inside the basis."""
        return float(self.variance.sum() / self.total_variance)

    def truncated(self, size: int) -> FluxBasis:
        """Return the same basis cut to its ``size`` leading directions."""
        if not 0 < size <= self.variance.size:
            raise ValueError(f"the basis has {self.variance.size} directions, asked for {size}")
        return FluxBasis(
            self.mean, self.directions[:size], self.variance[:size],
            self.total_variance, self.grid_shape,
        )

    def encode(self, maps: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the coordinates of maps ``(T, NZ, NR)``, shape ``(T, K)``."""
        flat = maps.reshape(maps.shape[0], -1).astype(np.float64)
        coordinates: NDArray[np.float64] = (flat - self.mean) @ self.directions.T
        return coordinates

    def decode(self, coordinates: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the maps of coordinates ``(T, K)``, shape ``(T, NZ, NR)``."""
        flat = coordinates.astype(np.float64) @ self.directions + self.mean
        maps: NDArray[np.float64] = flat.reshape(coordinates.shape[0], *self.grid_shape)
        return maps

    def save(self, path: Path) -> None:
        """Store the basis without pickling."""
        np.savez_compressed(
            path, allow_pickle=False, mean=self.mean, directions=self.directions,
            variance=self.variance, total_variance=np.float64(self.total_variance),
            grid_shape=np.asarray(self.grid_shape, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> FluxBasis:
        """Read a stored basis."""
        with np.load(path, allow_pickle=False) as stored:
            rows, columns = (int(value) for value in stored["grid_shape"])
            return cls(
                stored["mean"], stored["directions"], stored["variance"],
                float(stored["total_variance"]), (rows, columns),
            )
