# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — direct four-output temporal X specialist prototype
"""Share the static trunk while comparing mean and ordered temporal readouts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

D38 = Path(__file__).resolve().parent.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from temporal_model import initial as original_initial, pool as original_pool, window
from prepare_temporal import unpack
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.network import Network
import jax
import jax.numpy as jnp

LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')
MODES = ('static', 'mean', 'ordered')


def initial(key, settings):
    parameters = fit.initial_parameters(key, 614, 4, settings)
    original = original_initial(key, settings)
    for name in ('conv_weights', 'conv_biases'):
        parameters[name] = original[name]
    bound = 1. / np.sqrt(5*41*32)
    parameters['readout'] = jax.random.uniform(jax.random.fold_in(key, 900), (41*32, 32), jnp.float32, -bound, bound)
    parameters['readout_bias'] = jnp.zeros(32, dtype=jnp.float32)
    return parameters


def pool(parameters, raw, mean, scale, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if mode != 'ordered':
        return original_pool(parameters, raw, mean, scale, mode == 'mean')
    hidden = (window(raw.astype(mean.dtype))-mean)/scale
    for weights, bias, dilation in zip(parameters['conv_weights'], parameters['conv_biases'], (1, 2, 4), strict=True):
        padded = jnp.pad(hidden, ((0, 0), (2*dilation, 2*dilation), (0, 0)))
        hidden = jax.nn.silu(sum(padded[:, k*dilation:k*dilation+41] @ weights[k] for k in range(5))+bias)
    return jnp.clip(jax.nn.silu(hidden.reshape(len(raw), 41*32) @ parameters['readout']+parameters['readout_bias']), -8., 8.)


def forward(parameters, static, raw, mean, scale, mode):
    embedding = pool(parameters, raw, mean, scale, mode)
    return fit.forward(parameters, jnp.concatenate([static, embedding], axis=1), 0., None)


def loss(parameters, static, raw, targets, mean, scale, mode):
    predicted = forward(parameters, static, raw, mean, scale, mode)
    valid = jnp.isfinite(targets)
    error = jnp.where(valid, predicted-jnp.nan_to_num(targets), 0.)
    return jnp.mean(jnp.sum(error**2, axis=1))


def save(folder, parameters, scaling, settings, mean, scale, mode):
    if mode not in MODES:
        raise ValueError(mode)
    folder.mkdir(exist_ok=False)
    table = SimpleNamespace(label_names=LABELS, feature_layout='discharge')
    fit._export(parameters, scaling, settings, table).save(folder / 'network.npz')
    arrays = {'mode': np.asarray(mode), 'mean': np.asarray(mean), 'scale': np.asarray(scale),
              'readout': np.asarray(parameters['readout'], dtype=float),
              'readout_bias': np.asarray(parameters['readout_bias'], dtype=float)}
    for i in range(3):
        arrays[f'weights_{i}'] = np.asarray(parameters['conv_weights'][i], dtype=float)
        arrays[f'biases_{i}'] = np.asarray(parameters['conv_biases'][i], dtype=float)
    np.savez(folder / 'encoder.npz', **arrays)


class XTemporalNetwork:
    def __init__(self, folder):
        self.network = Network.load(folder / 'network.npz')
        assert self.network.coordinates == 0 and self.network.label_names == LABELS
        with np.load(folder / 'encoder.npz', allow_pickle=False) as stored:
            self.mode = str(stored['mode'])
            assert self.mode in MODES
            self.mean, self.scale = stored['mean'].copy(), stored['scale'].copy()
            self.weights = tuple(stored[f'weights_{i}'].astype(float) for i in range(3))
            self.biases = tuple(stored[f'biases_{i}'].astype(float) for i in range(3))
            self.readout = stored['readout'].astype(float)
            self.readout_bias = stored['readout_bias'].astype(float)

    def embedding(self, raw):
        if self.mode == 'static':
            return np.zeros((len(raw), 32))
        hidden = (unpack(raw).astype(float)-self.mean)/self.scale
        for weights, bias, dilation in zip(self.weights, self.biases, (1, 2, 4), strict=True):
            padded = np.pad(hidden, ((0, 0), (2*dilation, 2*dilation), (0, 0)))
            convolved = sum(padded[:, k*dilation:k*dilation+41] @ weights[k] for k in range(5))+bias
            hidden = convolved*expit(convolved)
        if self.mode == 'mean':
            return np.clip(hidden.mean(axis=1), -8., 8.)
        readout = hidden.reshape(len(raw), 41*32) @ self.readout+self.readout_bias
        return np.clip(readout*expit(readout), -8., 8.)

    def predict(self, static, raw):
        assert static.shape == (len(raw), 582) and raw.shape == (len(static), 840)
        _, labels = self.network.predict(np.concatenate([static, self.embedding(raw)], axis=1))
        return labels
