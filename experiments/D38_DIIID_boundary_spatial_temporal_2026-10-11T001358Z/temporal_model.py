# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — learned dilated temporal input encoder
"""Match JAX training with portable double-precision sequence inference."""
from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from prepare_temporal import unpack
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.network import Network


def window(flat: jax.Array) -> jax.Array:
    """Decode the original cache without changing channel or time order."""
    values = jnp.concatenate([flat[:, :40, None], flat[:, 40:800].reshape(-1, 40, 19), flat[:, 800:840, None]], axis=2)
    return jnp.concatenate([values[:, :20], jnp.zeros((len(flat), 1, 21), dtype=values.dtype), values[:, 20:]], axis=1)


def initial(key: jax.Array, settings) -> dict:
    """Initialize the identical static trunk and inactive control encoder."""
    parameters = fit.initial_parameters(key, 614, 2, settings)
    weights, biases = [], []
    for layer, before in enumerate((21, 32, 32)):
        bound = 1. / np.sqrt(5 * before)
        weights.append(jax.random.uniform(jax.random.fold_in(key, 100 + layer), (5, before, 32), jnp.float32, -bound, bound))
        biases.append(jnp.zeros(32, dtype=jnp.float32))
    parameters['conv_weights'], parameters['conv_biases'] = weights, biases
    return parameters


def pool(parameters: dict, raw: jax.Array, mean: jax.Array, scale: jax.Array, active: bool) -> jax.Array:
    """Apply three centered dilated convolutions and average all time positions."""
    if not active:
        return jnp.zeros((len(raw), 32), dtype=mean.dtype)
    hidden = (window(raw.astype(mean.dtype)) - mean) / scale
    for weights, bias, dilation in zip(parameters['conv_weights'], parameters['conv_biases'], (1, 2, 4), strict=True):
        padded = jnp.pad(hidden, ((0, 0), (2 * dilation, 2 * dilation), (0, 0)))
        convolved = sum(padded[:, k * dilation:k * dilation + 41] @ weights[k] for k in range(5)) + bias
        hidden = jax.nn.silu(convolved)
    return jnp.clip(hidden.mean(axis=1), -8., 8.)


def forward(parameters: dict, static: jax.Array, raw: jax.Array, mean: jax.Array, scale: jax.Array, active: bool) -> jax.Array:
    """Predict scaled scalar targets from current features and the learned history."""
    embedding = pool(parameters, raw, mean, scale, active)
    return fit.forward(parameters, jnp.concatenate([static, embedding], axis=1), 0., None)


class TemporalNetwork:
    """Read the exported scalar trunk and its complete convolutional encoder."""

    def __init__(self, folder: Path) -> None:
        self.network = Network.load(folder / 'network.npz')
        with np.load(folder / 'encoder.npz', allow_pickle=False) as stored:
            self.active = bool(stored['active'])
            self.mean, self.scale = stored['mean'].copy(), stored['scale'].copy()
            self.weights = tuple(stored[f'weights_{i}'].astype(float) for i in range(3))
            self.biases = tuple(stored[f'biases_{i}'].astype(float) for i in range(3))

    def predict(self, static: np.ndarray, raw: np.ndarray) -> dict:
        """Return unscaled q95 and beta_n using the exact declared input sequence."""
        if self.active:
            hidden = (unpack(raw).astype(float) - self.mean) / self.scale
            for weights, bias, dilation in zip(self.weights, self.biases, (1, 2, 4), strict=True):
                padded = np.pad(hidden, ((0, 0), (2 * dilation, 2 * dilation), (0, 0)))
                convolved = sum(padded[:, k * dilation:k * dilation + 41] @ weights[k] for k in range(5)) + bias
                hidden = convolved * expit(convolved)
            embedding = np.clip(hidden.mean(axis=1), -8., 8.)
        else:
            embedding = np.zeros((len(raw), 32))
        _, labels = self.network.predict(np.concatenate([static, embedding], axis=1))
        return labels
