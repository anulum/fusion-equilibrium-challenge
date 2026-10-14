# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — a learned correction on top of the physics flux map

"""Learn what the physics solve gets wrong, as a small linear model in a compressed space.

The physics pipeline reproduces the reconstructed flux to a pooled R² of about
0.92 and leaves an error that is smooth in space and slow in time: mostly the
current-profile shape of each discharge, which generic free functions cannot
know without magnetic measurements. A smooth error has few degrees of freedom,
so it is compressed by a principal-component basis fitted to the residual maps
``ψ_reconstruction - ψ_physics``, and each coefficient is regressed on
per-frame features by ridge regression.

Everything is linear on purpose. The base is a solved equilibrium and carries
the physics; the correction only has to be small, smooth and hard to overfit,
and a linear model in a truncated basis is all three. The model is stored as
plain arrays — no pickled objects — so a stored model can be read by anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FORMAT_VERSION = 1
"""Version of the stored-model layout; a file of another version is refused."""

FEATURE_AXES = 2
"""Features arrive as ``(frames, features)``."""

MAP_AXES = 3
"""Residual maps arrive as ``(frames, NZ, NR)``."""


@dataclass(frozen=True)
class ResidualModel:
    """A fitted correction: standardised features → basis coefficients → residual map.

    Attributes
    ----------
    feature_mean, feature_scale:
        Standardisation of the features, shape ``(F,)``. A feature that did not
        vary in the fit has scale one, so it contributes nothing instead of
        dividing by zero.
    residual_mean:
        Mean residual map, flattened, shape ``(P,)``.
    basis:
        Orthonormal principal directions of the centred residuals, ``(K, P)``.
    weights:
        Ridge coefficients from ``[1, standardised features]`` to the basis
        coefficients, shape ``(F + 1, K)``.
    map_shape:
        Shape ``(NZ, NR)`` the flattened maps are returned in.
    """

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    residual_mean: NDArray[np.float64]
    basis: NDArray[np.float64]
    weights: NDArray[np.float64]
    map_shape: tuple[int, int]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the predicted residual map of every frame, shape ``(T, NZ, NR)``.

        A non-finite feature is an input fault, not a frame to be guessed at,
        and is refused.
        """
        if features.ndim != FEATURE_AXES or features.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"expected features of shape (T, {self.feature_mean.size}), got {features.shape}"
            )
        if not bool(np.all(np.isfinite(features))):
            raise ValueError("features contain non-finite values")
        design = _design(features, self.feature_mean, self.feature_scale)
        maps = design @ self.weights @ self.basis + self.residual_mean
        return maps.reshape((features.shape[0], *self.map_shape))

    def save(self, path: Path) -> None:
        """Write the model as plain arrays."""
        np.savez_compressed(
            path, allow_pickle=False, format_version=np.int64(FORMAT_VERSION),
            feature_mean=self.feature_mean, feature_scale=self.feature_scale,
            residual_mean=self.residual_mean, basis=self.basis, weights=self.weights,
            map_shape=np.asarray(self.map_shape, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> ResidualModel:
        """Read a model written by :meth:`save`; refuse another layout version."""
        with np.load(path, allow_pickle=False) as stored:
            version = int(stored["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: stored-model version {version}, expected {FORMAT_VERSION}"
                )
            shape = stored["map_shape"]
            return cls(
                stored["feature_mean"], stored["feature_scale"], stored["residual_mean"],
                stored["basis"], stored["weights"], (int(shape[0]), int(shape[1])),
            )


def fit(
    features: NDArray[np.float64],
    residuals: NDArray[np.float64],
    n_components: int,
    ridge: float,
) -> ResidualModel:
    """Fit the basis and the regression.

    Parameters
    ----------
    features:
        Per-frame features, shape ``(T, F)``, finite.
    residuals:
        ``ψ_reconstruction - ψ_physics`` per frame, shape ``(T, NZ, NR)``, finite.
    n_components:
        Principal directions kept; at most ``min(T, NZ · NR)``.
    ridge:
        Penalty on the standardised-feature coefficients, relative to the
        number of frames; the intercept is not penalised. Must be positive:
        it is also what keeps collinear features from making the normal
        equations singular.
    """
    paired = (
        features.ndim == FEATURE_AXES
        and residuals.ndim == MAP_AXES
        and features.shape[0] == residuals.shape[0]
    )
    if not paired:
        raise ValueError(
            f"features {features.shape} and residuals {residuals.shape} do not pair up"
        )
    if not (bool(np.all(np.isfinite(features))) and bool(np.all(np.isfinite(residuals)))):
        raise ValueError("features and residuals must be finite")
    frames, (n_z, n_r) = features.shape[0], residuals.shape[1:]
    if not 1 <= n_components <= min(frames, n_z * n_r):
        raise ValueError(f"n_components must be between 1 and {min(frames, n_z * n_r)}")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    flat = residuals.reshape(frames, -1)
    residual_mean = flat.mean(axis=0)
    centred = flat - residual_mean
    # The right singular vectors come from the smaller Gram matrix: frames usually
    # outnumber grid cells, and a full SVD of the tall matrix would waste both time and memory.
    if frames >= centred.shape[1]:
        _, vectors = np.linalg.eigh(centred.T @ centred)
        basis = vectors[:, ::-1][:, :n_components].T
    else:
        _, vectors = np.linalg.eigh(centred @ centred.T)
        basis = (centred.T @ vectors[:, ::-1][:, :n_components]).T
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coefficients = centred @ basis.T

    feature_mean = features.mean(axis=0)
    # Constancy is tested on the range, which is exactly zero for identical values. The
    # standard deviation of a constant column is not — rounding leaves about 1e-14 — and
    # dividing by that would turn a feature that says nothing into the loudest one.
    varies = np.ptp(features, axis=0) > 0.0
    feature_scale = np.where(varies, features.std(axis=0), 1.0)
    design = _design(features, feature_mean, feature_scale)
    penalty = ridge * frames * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients)
    return ResidualModel(feature_mean, feature_scale, residual_mean, basis, weights, (n_z, n_r))


def _design(
    features: NDArray[np.float64], mean: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``[1, standardised features]``."""
    standardised = (features - mean) / scale
    return np.concatenate([np.ones((features.shape[0], 1)), standardised], axis=1)
