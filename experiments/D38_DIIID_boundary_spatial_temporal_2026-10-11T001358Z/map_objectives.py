# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — axis-relative boundary supervision and spatial decoder
"""Use declared differentiable surrogates; exact contour metrics remain validation."""
from __future__ import annotations

import jax.numpy as jnp
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit


def interpolate(images: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """Bilinearly sample flattened image channels at grid-index coordinates."""
    xy = jnp.clip(jnp.nan_to_num(points), 0., 64.)
    low = jnp.clip(jnp.floor(xy).astype(jnp.int32), 0, 63)
    fraction = xy - low
    x, y = low[:, :, 0], low[:, :, 1]
    dx, dy = fraction[:, :, 0:1], fraction[:, :, 1:2]
    table = images.T
    return ((1 - dx) * (1 - dy) * table[y * 65 + x] +
            dx * (1 - dy) * table[y * 65 + x + 1] +
            (1 - dx) * dy * table[(y + 1) * 65 + x] +
            dx * dy * table[(y + 1) * 65 + x + 1])


def effective_decoder(parameters: dict, mask: jnp.ndarray) -> jnp.ndarray:
    """Restrict learned residuals to the declared vessel band and unit row norm."""
    values = parameters['decoder'] * mask[None, :]
    norm = jnp.sqrt(jnp.sum(values * values, axis=1, keepdims=True))
    return values / jnp.maximum(norm, jnp.asarray(1e-8, dtype=values.dtype))


def objective(parameters: dict, inputs: jnp.ndarray, targets: jnp.ndarray,
              teacher_coordinates: jnp.ndarray, points: jnp.ndarray, rows: jnp.ndarray,
              basis: jnp.ndarray, coordinate_mean: jnp.ndarray, coordinate_scale: jnp.ndarray,
              decoder_mask: jnp.ndarray, family: str) -> jnp.ndarray:
    """Keep pooled coordinate/label anchors while changing only declared map terms."""
    output = fit.forward(parameters, inputs, 0., None)
    wanted = targets[:, :142]
    present = jnp.isfinite(wanted)
    error = jnp.where(present, output[:, :142] - jnp.nan_to_num(wanted), 0.)
    weights = jnp.concatenate([jnp.ones(64, dtype=jnp.float32), jnp.full(78, .05, dtype=jnp.float32)])
    anchor = jnp.mean(jnp.sum(error * error * weights, axis=1))
    if family == 'control':
        return anchor
    sampled = interpolate(basis, points)
    differences = sampled[:, 1:, :] - sampled[:, :1, :]
    truth = jnp.einsum('nc,npc->np', teacher_coordinates / coordinate_scale, differences)
    predicted_coordinates = output[:, :45] + coordinate_mean[:45] / coordinate_scale
    prediction = jnp.einsum('nc,npc->np', predicted_coordinates, differences[:, :, :45])
    decoder = None
    if family == 'spatial':
        decoder = effective_decoder(parameters, decoder_mask)
        sample_decoder = interpolate(decoder, points)
        decoder_difference = sample_decoder[:, 1:, :] - sample_decoder[:, :1, :]
        prediction += jnp.einsum('nc,npc->np', output[:, 142:], decoder_difference)
    valid = jnp.isfinite(points[:, 1:, :]).all(axis=2) & jnp.isfinite(points[:, :1, :]).all(axis=2)
    ray_error = jnp.where(valid, prediction - truth, 0.)
    mean_ray_squared = jnp.sum(ray_error * ray_error, axis=1) / jnp.maximum(valid.sum(axis=1), 1)
    result = anchor + .2 * 4225 * jnp.mean(rows * mean_ray_squared)
    if decoder is not None:
        field_prediction = predicted_coordinates @ basis[:45] + output[:, 142:] @ decoder
        field_truth = (teacher_coordinates / coordinate_scale) @ basis
        result += .2 * jnp.mean(jnp.sum((field_prediction - field_truth) ** 2, axis=1))
    return result
