# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — grouped finite geometry readout residual moments
"""Fit physical geometry residuals with exact per-head target validity masks."""
from dataclasses import replace
import hashlib

import numpy as np

from li_readout_ridge import hidden_features


def validity_groups(truth):
    """Group only columns with identical complete-corpus finite-target masks."""
    masks = np.isfinite(truth)
    groups = {}
    for column in range(truth.shape[1]):
        key = hashlib.sha256(np.packbits(masks[:, column]).tobytes()).hexdigest()
        group = groups.setdefault(key, [])
        if group:
            np.testing.assert_array_equal(masks[:, column], masks[:, group[0]])
        group.append(column)
    return tuple(tuple(columns) for columns in groups.values())


def cpu_moments(network, features, truth, names, groups):
    """Use the original public predictor once and accumulate each validity group."""
    hidden = hidden_features(network, features)
    _, labels = network.predict(features)
    residual = np.asarray(truth, dtype=float)-np.column_stack([labels[name] for name in names])
    valid_hidden = np.isfinite(hidden).all(axis=1)
    if not np.isfinite(residual[np.isfinite(truth) & valid_hidden[:, None]]).all():
        raise ValueError('Original prediction is nonfinite at a valid target')
    moments = []
    for columns in groups:
        selected = np.asarray(columns)
        target_valid = np.isfinite(truth[:, selected])
        assert np.all(target_valid == target_valid[:, :1])
        finite = target_valid[:, 0] & valid_hidden
        h, r = hidden[finite], residual[finite][:, selected]
        moments.append((len(r), h.sum(0), r.sum(0), h.T@h, h.T@r, np.sum(r*r, axis=0)))
    return tuple(moments)


def corrected_from_moments(network, moments, names, groups, ridge=.001):
    """Solve standardized ridge residuals and change only named final rows."""
    assert sorted(column for group in groups for column in group) == list(range(len(names)))
    weights = [value.copy() for value in network.weights]
    biases = [value.copy() for value in network.biases]
    reports = []
    for values, columns in zip(moments, groups, strict=True):
        n, total_h, total_r, hh, hr, rr = values
        labels = [names[column] for column in columns]
        if n == 0:
            reports.append({'labels': labels, 'finite_rows': 0, 'unchanged_no_finite_targets': True})
            continue
        if n < 2:
            raise ValueError('Insufficient finite rows in a geometry group')
        mean, mean_r = np.asarray(total_h)/n, np.asarray(total_r)/n
        cov = np.asarray(hh)/n-np.outer(mean, mean)
        cov = (cov+cov.T)*.5
        spread = np.sqrt(np.maximum(np.diag(cov), 0))
        spread = np.where(spread > 0, spread, 1)
        z_cov = cov/spread[:, None]/spread[None, :]
        rhs = (np.asarray(hr)/n-mean[:, None]*mean_r[None, :])/spread[:, None]
        scaled = np.linalg.solve(z_cov+ridge*np.eye(len(mean)), rhs)
        delta, bias = scaled/spread[:, None], mean_r-mean@(scaled/spread[:, None])
        indices = np.asarray([network.coordinates+network.label_names.index(name) for name in labels])
        weights[-1][indices] += delta.T/network.output_scale[indices, None]
        biases[-1][indices] += bias/network.output_scale[indices]
        gradient = z_cov@scaled-rhs+ridge*scaled
        reports.append({'labels': labels, 'finite_rows': int(n), 'ridge': ridge,
                        'mean_residual_before': mean_r.tolist(), 'residual_MSE_before': (np.asarray(rr)/n).tolist(),
                        'normal_equation_gradient_max_abs': float(np.max(np.abs(gradient)))})
    return replace(network, weights=tuple(weights), biases=tuple(biases)), reports


def gpu_moment_function(network, names, groups):
    """Share one stable float64 forward across all exact validity-group moments."""
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise RuntimeError('Float64 JAX is required')
    indices = np.asarray([network.coordinates+network.label_names.index(name) for name in names])
    mean, scale = jnp.asarray(network.feature_mean), jnp.asarray(network.feature_scale)
    layers = [(jnp.asarray(w), jnp.asarray(b)) for w, b in zip(network.weights[:-1], network.biases[:-1], strict=True)]
    last, bias, linear = jnp.asarray(network.weights[-1][indices]), jnp.asarray(network.biases[-1][indices]), jnp.asarray(network.linear[indices])
    output_mean, output_scale = jnp.asarray(network.output_mean[indices]), jnp.asarray(network.output_scale[indices])

    @jax.jit
    def moments(features, truth):
        inputs = jnp.clip((jnp.asarray(features, dtype=jnp.float64)-mean)/scale, -8, 8)
        hidden = inputs
        for weight, offset in layers:
            values = hidden@weight.T+offset
            gate = jnp.exp(-jnp.abs(values))
            hidden = values*jnp.where(values >= 0, 1/(1+gate), gate/(1+gate))
        residual = jnp.asarray(truth, dtype=jnp.float64)-(hidden@last.T+bias+inputs@linear.T)*output_scale-output_mean
        valid_hidden = jnp.all(jnp.isfinite(hidden), axis=1)
        invalid_prediction = jnp.sum(jnp.isfinite(truth) & valid_hidden[:, None] & ~jnp.isfinite(residual))
        results = []
        invalid_group_masks = jnp.asarray(0)
        for columns in groups:
            selected = jnp.asarray(columns)
            target_valid = jnp.isfinite(truth[:, selected])
            invalid_group_masks += jnp.sum(target_valid != target_valid[:, :1])
            finite = target_valid[:, 0] & valid_hidden
            h = jnp.where(finite[:, None], hidden, 0)
            r = jnp.where(finite[:, None], residual[:, selected], 0)
            results.append((jnp.sum(finite), jnp.sum(h, axis=0), jnp.sum(r, axis=0), h.T@h, h.T@r, jnp.sum(r*r, axis=0)))
        return tuple(results), invalid_prediction, invalid_group_masks
    return moments
