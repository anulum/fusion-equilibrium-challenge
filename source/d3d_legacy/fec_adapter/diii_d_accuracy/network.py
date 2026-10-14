# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — a fitted network stored and evaluated as plain arrays

"""Hold a fitted network as plain arrays and evaluate it without the library that fitted it.

The network is fitted elsewhere, with a deep-learning library, on a machine
chosen for its memory. What is submitted is predicted here, and the prediction
must not depend on that library being installed, on its version, or on
unpickling anything: the stored file is arrays and a few numbers, read with
pickling switched off, and the forward pass is a dozen lines of matrix algebra.
The fitting side is held to this evaluation by a test, to single precision.

The form is a linear path plus a multilayer path, added: the flux of the coils
alone is exactly linear in their currents, so the linear path can carry it and
leave the rest — the plasma's answer — to the layers. The activation is
``x · sigmoid(x)``, which both sides can write down identically.

Inputs are standardised and then clipped. The ratio features grow without
bound as the plasma current falls to nothing at either end of a discharge, and
an unclipped input of fifty standard deviations sends the layers somewhere they
never were during fitting.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy.layout import RECORD_LAYOUT, FeatureLayout, layout_of

INPUT_CLIP = 8.0
"""Standard deviations beyond which a standardised input is held."""

FORMAT_VERSION = 1


def silu(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``x · sigmoid(x)``, evaluated without overflow for large negative ``x``."""
    gate = np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(values))),
        np.exp(-np.abs(values)) / (1.0 + np.exp(-np.abs(values))),
    )
    activated: NDArray[np.float64] = values * gate
    return activated


@dataclass(frozen=True)
class Network:
    """A fitted network: standardisation, a linear path, a multilayer path, output scaling.

    Attributes
    ----------
    feature_mean, feature_scale:
        Standardisation of the inputs, shape ``(F,)``.
    linear:
        Weight of the linear path, shape ``(O, F)``; it has no bias of its own.
    weights, biases:
        The layers of the multilayer path, first to last; the last one maps to ``O``.
    output_mean, output_scale:
        The network's raw output is multiplied by the scale and the mean added, shape ``(O,)``.
    coordinates:
        How many leading outputs are flux coordinates; the rest are the labels named below.
    label_names:
        Names of the outputs after the coordinates.
    feature_layout:
        What the feature columns the network was fitted on are measured against
        (:mod:`.layout`); it reads the numbers of that layout and no other.
    """

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    linear: NDArray[np.float64]
    weights: tuple[NDArray[np.float64], ...]
    biases: tuple[NDArray[np.float64], ...]
    output_mean: NDArray[np.float64]
    output_scale: NDArray[np.float64]
    coordinates: int
    label_names: tuple[str, ...]
    feature_layout: FeatureLayout = RECORD_LAYOUT

    def standardised(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the inputs as the layers see them."""
        if features.ndim != 2 or features.shape[1] != self.feature_mean.size:  # noqa: PLR2004
            raise ValueError(
                f"features of shape {features.shape} do not have "
                f"{self.feature_mean.size} columns"
            )
        scaled = (features.astype(np.float64) - self.feature_mean) / self.feature_scale
        clipped: NDArray[np.float64] = np.clip(scaled, -INPUT_CLIP, INPUT_CLIP)
        return clipped

    def predict(
        self, features: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        """Return the flux coordinates ``(T, K)`` and the labels by name, each ``(T,)``."""
        inputs = self.standardised(features)
        hidden = inputs
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            hidden = silu(hidden @ weight.T + bias)
        raw = hidden @ self.weights[-1].T + self.biases[-1] + inputs @ self.linear.T
        outputs = raw * self.output_scale + self.output_mean
        labels = {
            name: outputs[:, self.coordinates + index]
            for index, name in enumerate(self.label_names)
        }
        return outputs[:, : self.coordinates], labels

    def save(self, path: Path) -> None:
        """Store the network as arrays and a small description, without pickling."""
        arrays = {f"weight_{index}": weight for index, weight in enumerate(self.weights)}
        arrays.update({f"bias_{index}": bias for index, bias in enumerate(self.biases)})
        description = json.dumps({
            "format": FORMAT_VERSION,
            "layers": len(self.weights),
            "coordinates": self.coordinates,
            "label_names": list(self.label_names),
            "feature_layout": self.feature_layout,
        })
        np.savez_compressed(
            path, allow_pickle=False, description=np.asarray(description),
            feature_mean=self.feature_mean, feature_scale=self.feature_scale,
            linear=self.linear, output_mean=self.output_mean, output_scale=self.output_scale,
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> Network:
        """Read a stored network; a file of another format version is refused.

        A file that names no feature layout was written before the second layout
        existed, and is record-relative.
        """
        with np.load(path, allow_pickle=False) as stored:
            description = json.loads(str(stored["description"]))
            if description.get("format") != FORMAT_VERSION:
                raise ValueError(f"{path} has format {description.get('format')!r}")
            layers = range(int(description["layers"]))
            return cls(
                feature_mean=stored["feature_mean"],
                feature_scale=stored["feature_scale"],
                linear=stored["linear"],
                weights=tuple(stored[f"weight_{index}"] for index in layers),
                biases=tuple(stored[f"bias_{index}"] for index in layers),
                output_mean=stored["output_mean"],
                output_scale=stored["output_scale"],
                coordinates=int(description["coordinates"]),
                label_names=tuple(description["label_names"]),
                feature_layout=layout_of(description.get("feature_layout", RECORD_LAYOUT)),
            )


def predict_together(
    networks: Sequence[Network], features: Mapping[FeatureLayout, NDArray[np.floating]]
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Return the mean prediction of networks fitted alike from different starting weights.

    The fit reaches the training shots far more closely than unseen ones (li: R² 0.988
    against 0.930), and that gap is each network's own scatter around what the data
    support; the scatter of separately started networks is largely independent, so
    their mean keeps the signal and loses part of it. The flux coordinates are averaged
    as they are, which is averaging the maps, the basis being linear.

    Feature blocks are only ever appended to the feature matrix, so a network fitted
    before a block existed reads its own leading columns; networks of different input
    width can therefore answer together. A network wider than the matrix is refused
    by its own shape check. Likewise a label is averaged over the networks that have
    it: outputs are only ever added, so networks fitted before an output existed
    still answer the coordinates and the outputs they share with the newer ones.

    Each network reads the matrix of its own feature layout, so networks of both
    layouts can answer together; a network whose layout's matrix is missing is
    refused rather than handed another layout's numbers.

    Parameters
    ----------
    networks:
        The networks to average; they must give the same number of coordinates.
    features:
        One shot's feature matrix, ``(T, F)``, for every layout the networks read.
    """
    if not networks:
        raise ValueError("no network to predict with")
    missing = sorted({net.feature_layout for net in networks} - set(features))
    if missing:
        raise ValueError(f"no feature matrix of layout {missing} for the networks given")
    first = networks[0]
    for other in networks[1:]:
        if other.coordinates != first.coordinates:
            raise ValueError("the networks do not give the same number of coordinates")
    answers = [
        network.predict(features[network.feature_layout][:, : network.feature_mean.size])
        for network in networks
    ]
    coordinates: NDArray[np.float64] = np.mean([answer[0] for answer in answers], axis=0)
    names = dict.fromkeys(name for network in networks for name in network.label_names)
    labels = {
        name: np.asarray(np.mean([answer[1][name] for answer in answers if name in answer[1]],
                                 axis=0))
        for name in names
    }
    return coordinates, labels
