# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — paired X specialist optimizer and checkpoint kernel
"""Train with shared initial arrays and retain immutable resumable state."""
import json
from functools import partial
from pathlib import Path

import numpy as np
import x_temporal_model as model
import shared_initial_parameters as initial
from candidate_state import save as save_state, restore as restore_state
from fec_adapter.diii_d_accuracy.fit_network import FitSettings
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent


def settings(seed):
    recipe = json.loads((HERE / 'training_recipe_frozen_v1.json').read_text())
    assert seed in recipe['paired_seeds']
    return FitSettings(**{**recipe['fit_settings_seed500'], 'seed': seed})


def initial_state(seed, device):
    receipt = json.loads((initial.OUT / 'receipt.json').read_text())
    item = next(r for r in receipt['models'] if r['seed'] == seed)
    assert initial.digest(item['file']) == item['sha256']
    parameters = jax.tree.map(lambda a: jax.device_put(a, device), initial.load(Path(item['file'])))
    with jax.default_device(device):
        state = (parameters, jax.tree.map(jnp.zeros_like, parameters),
                 jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))
        key = jax.random.fold_in(jax.random.key(seed), 700)
    return state, key


def epoch_step(data, normalization, mode, recipe):
    assert mode in model.MODES
    static, targets, raw = data
    assert static.shape == (len(raw), 582) and targets.shape == (len(raw), 4)
    assert raw.shape[1] == 840 and len(raw) >= recipe.batch
    mean, scale = normalization
    batches = len(raw)//recipe.batch
    rows = len(raw)

    @jax.jit
    def one_epoch(state, order_key, inputs, wanted, windows):
        def update(carry, chosen):
            value, gradient = jax.value_and_grad(model.loss)(carry[0], inputs[chosen], windows[chosen],
                                                            wanted[chosen], mean, scale, mode)
            return model.fit.decayed_adam_step(carry, gradient, recipe.epochs*batches, recipe), value

        order = jax.random.permutation(order_key, rows)[:batches*recipe.batch]
        state, losses = jax.lax.scan(update, state, order.reshape(batches, recipe.batch))
        return state, jnp.mean(losses)

    return partial(one_epoch, inputs=static, wanted=targets, windows=raw)


def advance(step, state, key, history, epochs):
    history = list(history)
    for _ in range(epochs):
        key, order = jax.random.split(key)
        state, value = step(state, order)
        value.block_until_ready()
        loss = float(value)
        assert np.isfinite(loss), 'nonfinite training loss; no numerical retry'
        history.append(loss)
    return state, key, history


def checkpoint(folder, state, key, history, frozen):
    folder.mkdir(exist_ok=True)
    epoch = len(history)
    path = folder / ('epoch'+str(epoch)+'.npz')
    record = path.with_suffix('.json')
    assert not path.exists() and not record.exists(), 'immutable checkpoint already exists'
    save_state(path, state, key, epoch, frozen)
    with record.open('x') as stream:
        json.dump({'epoch': epoch, 'loss_history': history, 'checkpoint_sha256': initial.digest(path),
                   'frozen': frozen}, stream, indent=2, allow_nan=False)
    return path


def resume(path, template, frozen):
    record = json.loads(path.with_suffix('.json').read_text())
    assert record['frozen'] == frozen and initial.digest(path) == record['checkpoint_sha256']
    state, key, epoch = restore_state(path, template, frozen)
    assert epoch == record['epoch'] == len(record['loss_history'])
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state))
    return state, key, record['loss_history']


def quality(history, parameters):
    assert len(history) == 60
    values = np.asarray(history, dtype=float)
    assert np.isfinite(values).all()
    final = float(values[-10:].mean())
    best = float(np.lib.stride_tricks.sliding_window_view(values, 10).mean(axis=1).min())
    finite = all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(parameters))
    return {'final10_mean_training_loss': final, 'lowest10_mean_training_loss': best,
            'final_parameters_finite': finite, 'seed_quality_pass': bool(finite and final <= 1.25*best)}
