# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D57: a head-only network that reads a whole shot
"""Per-frame trunk, attention over the frames of one shot, per-frame heads.

A frame's 582 normalised features pass a 4 x 512 trunk. Its last activation is projected to 128, a sinusoidal
encoding of the frame's time is added, and two pre-norm blocks mix the frames of the shot by self-attention
(4 heads) and a 128-256-128 perceptron. The heads read the trunk activation and the mixed vector. The output
projections of both block parts start at zero, so at the start the network is the per-frame trunk alone.

``mix_frames`` False restricts every frame's attention to itself: the same parameters, layers and arithmetic,
with no information from other frames. That is the matched control.
"""
import jax
import jax.numpy as jnp

WIDTH, DEPTH, MIX, HEADS, BLOCKS = 512, 4, 128, 4, 2
PERIODS_S = (0.04, 0.1, 0.25, 0.6, 1.5, 4.0, 10.0, 25.0)


def dense(key, fan_in, fan_out, zero=False):
    scale = 0.0 if zero else (2.0 / fan_in) ** 0.5
    return {'w': scale * jax.random.normal(key, (fan_in, fan_out), jnp.float32), 'b': jnp.zeros((fan_out,), jnp.float32)}


def initial(key, inputs, outputs):
    keys = iter(jax.random.split(key, 64))
    trunk = [dense(next(keys), inputs if i == 0 else WIDTH, WIDTH) for i in range(DEPTH)]
    blocks = []
    for _ in range(BLOCKS):
        blocks.append({'q': dense(next(keys), MIX, MIX), 'k': dense(next(keys), MIX, MIX), 'v': dense(next(keys), MIX, MIX),
                       'o': dense(next(keys), MIX, MIX, zero=True), 'f1': dense(next(keys), MIX, 2 * MIX),
                       'f2': dense(next(keys), 2 * MIX, MIX, zero=True),
                       'g1': jnp.ones((MIX,), jnp.float32), 'g2': jnp.ones((MIX,), jnp.float32)})
    return {'trunk': trunk, 'project': dense(next(keys), WIDTH, MIX), 'time': dense(next(keys), 2 * len(PERIODS_S), MIX),
            'blocks': blocks, 'head': dense(next(keys), WIDTH + MIX, outputs)}


def apply(layer, x):
    return x @ layer['w'] + layer['b']


def norm(x, gain):
    centred = x - jnp.mean(x, axis=-1, keepdims=True)
    return gain * centred / jnp.sqrt(jnp.mean(centred ** 2, axis=-1, keepdims=True) + 1e-6)


def time_code(seconds):
    phase = 2 * jnp.pi * seconds[..., None] / jnp.asarray(PERIODS_S, jnp.float32)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def forward(parameters, x, seconds, valid, mix_frames):
    """x (B, L, F) normalised features, seconds (B, L), valid (B, L) bool -> (B, L, outputs)."""
    h = x
    for layer in parameters['trunk']:
        h = jax.nn.silu(apply(layer, h))
    z = apply(parameters['project'], h) + apply(parameters['time'], time_code(seconds))
    length = x.shape[1]
    allowed = valid[:, None, :] & valid[:, :, None]
    if not mix_frames:
        allowed = allowed & jnp.eye(length, dtype=bool)[None]
    allowed = allowed | jnp.eye(length, dtype=bool)[None]          # a padded frame attends to itself: no empty row
    bias = jnp.where(allowed, 0.0, -1e9)[:, None, :, :]
    size = MIX // HEADS
    for block in parameters['blocks']:
        y = norm(z, block['g1'])
        split = lambda t: t.reshape(t.shape[0], length, HEADS, size).transpose(0, 2, 1, 3)
        q, k, v = split(apply(block['q'], y)), split(apply(block['k'], y)), split(apply(block['v'], y))
        weight = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / size ** 0.5 + bias, axis=-1)
        mixed = (weight @ v).transpose(0, 2, 1, 3).reshape(z.shape)
        z = z + apply(block['o'], mixed)
        z = z + apply(block['f2'], jax.nn.silu(apply(block['f1'], norm(z, block['g2']))))
    return apply(parameters['head'], jnp.concatenate([h, z], axis=-1))
