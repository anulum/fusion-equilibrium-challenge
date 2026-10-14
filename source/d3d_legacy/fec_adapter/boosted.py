# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — gradient-boosted regression kept as plain arrays

"""Fit a gradient-boosted regressor and keep it as plain arrays that evaluate without the library.

The fit is scikit-learn's histogram gradient boosting. What is kept is not the
fitted object — storing that means pickling it, and a pickle runs code when it
is read and breaks when the library moves on — but the trees themselves: for
every node its feature, threshold, children and value. Evaluating such trees is
a few lines of array code, written here and pinned against the library's own
prediction to rounding error, so a stored model can be read and applied with
NumPy alone.

Features are finite by construction in this package, so the library's handling
of missing values is not reproduced; a non-finite feature is refused instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

FORMAT_VERSION = 1
"""Version of the stored layout; a file of another version is refused."""

FEATURE_AXES = 2
"""Features arrive as ``(samples, features)``."""


@dataclass(frozen=True)
class BoostingSettings:
    """Settings of the fit; the defaults are the ones measured on the scalar heads."""

    max_depth: int = 6
    max_iter: int = 300
    learning_rate: float = 0.1
    l2_regularization: float = 1.0
    min_samples_leaf: int = 20
    seed: int = 0


@dataclass(frozen=True)
class BoostedModel:
    """A sum of regression trees over a constant, as flat node arrays.

    Attributes
    ----------
    baseline:
        The constant every prediction starts from.
    roots:
        Index of each tree's root node in the node arrays, shape ``(N_trees,)``.
    feature, threshold:
        Split of each node: a sample goes left when its feature value is at
        most the threshold. Unused in leaves.
    left, right:
        Children of each node, as indices into the node arrays.
    value:
        Contribution of each leaf, the learning rate already applied.
    is_leaf:
        Which nodes are leaves.
    n_features:
        Number of features the model was fitted on.
    """

    baseline: float
    roots: NDArray[np.int64]
    feature: NDArray[np.int64]
    threshold: NDArray[np.float64]
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    value: NDArray[np.float64]
    is_leaf: NDArray[np.bool_]
    n_features: int

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the prediction for every sample, shape ``(samples,)``."""
        if features.ndim != FEATURE_AXES or features.shape[1] != self.n_features:
            raise ValueError(
                f"expected features of shape (samples, {self.n_features}), got {features.shape}"
            )
        if not bool(np.all(np.isfinite(features))):
            raise ValueError("features contain non-finite values")
        samples = np.arange(features.shape[0])
        total = np.full(features.shape[0], self.baseline)
        for root in self.roots:
            node = np.full(features.shape[0], root)
            while True:
                inner = ~self.is_leaf[node]
                if not inner.any():
                    break
                at = node[inner]
                goes_left = features[samples[inner], self.feature[at]] <= self.threshold[at]
                node[inner] = np.where(goes_left, self.left[at], self.right[at])
            total += self.value[node]
        return total

    def save(self, path: Path) -> None:
        """Write the model as plain arrays."""
        np.savez_compressed(
            path, allow_pickle=False, format_version=np.int64(FORMAT_VERSION),
            baseline=np.float64(self.baseline), roots=self.roots, feature=self.feature,
            threshold=self.threshold, left=self.left, right=self.right, value=self.value,
            is_leaf=self.is_leaf, n_features=np.int64(self.n_features),
        )

    @classmethod
    def load(cls, path: Path) -> BoostedModel:
        """Read a model written by :meth:`save`; refuse another layout version."""
        with np.load(path, allow_pickle=False) as stored:
            version = int(stored["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: stored-model version {version}, expected {FORMAT_VERSION}"
                )
            return cls(
                float(stored["baseline"]), stored["roots"], stored["feature"],
                stored["threshold"], stored["left"], stored["right"], stored["value"],
                stored["is_leaf"], int(stored["n_features"]),
            )


def fit_boosted(
    features: NDArray[np.float64],
    values: NDArray[np.float64],
    settings: BoostingSettings | None = None,
) -> BoostedModel:
    """Fit on finite features and values and return the trees as arrays."""
    settings = settings or BoostingSettings()
    if features.ndim != FEATURE_AXES or values.shape != (features.shape[0],):
        raise ValueError(f"features {features.shape} and values {values.shape} do not pair up")
    if not (bool(np.all(np.isfinite(features))) and bool(np.all(np.isfinite(values)))):
        raise ValueError("features and values must be finite")
    fitted = HistGradientBoostingRegressor(
        max_depth=settings.max_depth, max_iter=settings.max_iter,
        learning_rate=settings.learning_rate, l2_regularization=settings.l2_regularization,
        min_samples_leaf=settings.min_samples_leaf, early_stopping=False,
        random_state=settings.seed,
    ).fit(features, values)
    return export_trees(fitted, features.shape[1])


def export_trees(fitted: HistGradientBoostingRegressor, n_features: int) -> BoostedModel:
    """Copy a fitted regressor's trees into flat node arrays.

    This reads the regressor's private tree tables; the equality test against
    the regressor's own prediction is what notices if a library release changes
    their meaning.
    """
    tables = [iteration[0].nodes for iteration in fitted._predictors]  # noqa: SLF001
    sizes = np.asarray([table.shape[0] for table in tables], dtype=np.int64)
    roots = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    nodes = np.concatenate(tables)
    offset = np.repeat(roots, sizes)
    return BoostedModel(
        baseline=float(np.ravel(fitted._baseline_prediction)[0]),  # noqa: SLF001
        roots=roots,
        feature=nodes["feature_idx"].astype(np.int64),
        threshold=nodes["num_threshold"].astype(np.float64),
        left=nodes["left"].astype(np.int64) + offset,
        right=nodes["right"].astype(np.int64) + offset,
        value=nodes["value"].astype(np.float64),
        is_leaf=nodes["is_leaf"].astype(bool),
        n_features=n_features,
    )
