# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the fitted correction as one bundle, tied to the physics it was fitted on

"""Fit, store and apply the learned correction as one bundle.

A bundle holds three models on the same features — a linear one for the flux
map and a boosted one each for ``q95`` and the normalised beta — together with
a fingerprint
of the physics settings its training base was generated with. The correction
is the difference between the reconstruction and *that* physics; applied to a
base solved with another pressure ratio or another cold support it would add
the wrong difference and nothing would fail. The fingerprint turns that into a
refusal.

The two scalars are learned directly rather than derived from the corrected
map: they depend on the pressure and the toroidal field as well as on the flux,
and the derived pair scored below zero against the reconstruction's own values.
Their heads are boosted trees rather than linear: the safety factor goes as a
ratio of currents, and on 64 held-out shots boosted heads reached R² 0.93 and
0.97 where linear ones, given the same ratio features, reached 0.78 and 0.92.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter import residual
from fec_adapter.boosted import BoostedModel, BoostingSettings, fit_boosted
from fec_adapter.physics_base import PhysicsBase
from fec_adapter.shot_reader import Shot

from .residual_features import DEFAULT_BLOCKS, frame_features

FLUX_FILE, Q95_FILE, BETA_FILE, SETTINGS_FILE = "flux.npz", "q95.npz", "beta_n.npz", "physics.json"
FEATURES_FILE = "feature_blocks.json"


@dataclass(frozen=True)
class TrainingShot:
    """One shot of the training set: inputs, physics base, and the reconstruction's answers.

    This is the only place in the package where reconstruction values are held,
    and they are held as targets: nothing here feeds them to the feature builder.
    """

    shot: Shot
    base: PhysicsBase
    psi: NDArray[np.float64]
    q95: NDArray[np.float64]
    beta_n: NDArray[np.float64]


@dataclass(frozen=True)
class CorrectionBundle:
    """The three fitted models and the physics they belong to."""

    flux: residual.ResidualModel
    q95: BoostedModel
    beta_n: BoostedModel
    physics: dict[str, object]
    feature_blocks: tuple[str, ...] = DEFAULT_BLOCKS

    def apply(
        self, shot: Shot, base: PhysicsBase, physics: dict[str, object]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(corrected flux, q95, betaN)`` for one shot.

        Parameters
        ----------
        shot, base:
            The shot's inputs and its physics base, in the written orientation.
        physics:
            Fingerprint of the settings ``base`` was solved with; it must equal
            the bundle's own.
        """
        if _canonical(physics) != _canonical(self.physics):
            raise ValueError(
                "the correction was fitted on a physics base solved with other settings: "
                f"bundle {_canonical(self.physics)} against current {_canonical(physics)}"
            )
        features = frame_features(shot, base, self.feature_blocks)
        return (
            base.psi + self.flux.predict(features),
            self.q95.predict(features),
            self.beta_n.predict(features),
        )

    def save(self, directory: Path) -> None:
        """Write the bundle as three plain-array files and one JSON file."""
        directory.mkdir(parents=True, exist_ok=True)
        self.flux.save(directory / FLUX_FILE)
        self.q95.save(directory / Q95_FILE)
        self.beta_n.save(directory / BETA_FILE)
        (directory / SETTINGS_FILE).write_text(json.dumps(self.physics, indent=2) + "\n")
        (directory / FEATURES_FILE).write_text(json.dumps(list(self.feature_blocks)) + "\n")

    @classmethod
    def load(cls, directory: Path) -> CorrectionBundle:
        """Read a bundle written by :meth:`save`."""
        return cls(
            residual.ResidualModel.load(directory / FLUX_FILE),
            BoostedModel.load(directory / Q95_FILE),
            BoostedModel.load(directory / BETA_FILE),
            json.loads((directory / SETTINGS_FILE).read_text()),
            _stored_blocks(directory),
        )


@dataclass(frozen=True)
class FitSettings:
    """What a fit of the bundle is free to choose.

    Attributes
    ----------
    n_components:
        Principal directions of the flux residual that are kept.
    ridge:
        Penalty of the flux regression.
    boosting:
        Settings of the two scalar heads; the package defaults when ``None``.
    feature_blocks:
        Optional feature blocks to fit with; the bundle records them.
    """

    n_components: int = 16
    ridge: float = 1.0e-3
    boosting: BoostingSettings | None = None
    feature_blocks: tuple[str, ...] = DEFAULT_BLOCKS


def fit_bundle(
    shots: Iterable[TrainingShot], physics: dict[str, object], settings: FitSettings
) -> CorrectionBundle:
    """Fit the flux correction and the two scalar heads on a set of shots.

    Frames whose reconstructed scalar is not finite are left out of that
    scalar's fit and kept for the flux, so a missing ``q95`` does not cost the
    frame its flux residual.
    """
    features, residuals, q95, beta_n = [], [], [], []
    for sample in shots:
        if sample.psi.shape != sample.base.psi.shape:
            raise ValueError(
                f"{sample.shot.name}: reconstruction {sample.psi.shape} and base "
                f"{sample.base.psi.shape} differ in shape"
            )
        features.append(frame_features(sample.shot, sample.base, settings.feature_blocks))
        residuals.append(sample.psi - sample.base.psi)
        q95.append(sample.q95)
        beta_n.append(sample.beta_n)
    if not features:
        raise ValueError("no shot to fit on")
    all_features = np.concatenate(features)
    return CorrectionBundle(
        flux=residual.fit(
            all_features, np.concatenate(residuals), settings.n_components, settings.ridge
        ),
        q95=_scalar_head(all_features, np.concatenate(q95), settings.boosting),
        beta_n=_scalar_head(all_features, np.concatenate(beta_n), settings.boosting),
        physics=physics,
        feature_blocks=settings.feature_blocks,
    )


def _scalar_head(
    features: NDArray[np.float64],
    values: NDArray[np.float64],
    boosting: BoostingSettings | None,
) -> BoostedModel:
    """Fit one scalar on the frames where it is finite."""
    known = np.isfinite(values)
    if not known.any():
        raise ValueError("a scalar head has no finite value to fit on")
    return fit_boosted(features[known], values[known], boosting)


def _stored_blocks(directory: Path) -> tuple[str, ...]:
    """Return the optional feature blocks a stored bundle was fitted with.

    A bundle written before the blocks were recorded has no such file; it was
    fitted before any optional block existed, which is the empty set.
    """
    path = directory / FEATURES_FILE
    return tuple(json.loads(path.read_text())) if path.exists() else ()


def _canonical(physics: dict[str, object]) -> str:
    """Return a fingerprint in a form in which equal settings compare equal."""
    return json.dumps(physics, sort_keys=True)
