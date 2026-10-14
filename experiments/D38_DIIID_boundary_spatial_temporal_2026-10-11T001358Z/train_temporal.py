# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — paired static and temporal scalar training
"""Fit every prospective temporal arm before comparing development scores."""
from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import HERE, digest, write_json
from training_data import COMPILE, initialize, scaling_for
from candidate_state import save, restore
from temporal_tables import upload
from temporal_model import initial, forward, window, TemporalNetwork
from prepare_temporal import unpack
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling

STATE = {}


def settings(seed: int, epochs: int = 60) -> FitSettings:
    """Use the frozen paired scalar schedule."""
    return FitSettings(coordinates=0, seed=seed, epochs=epochs, batch=2048, gradient_limit=1.)


def scalar_scaling(scaling, labels: tuple) -> _Scaling:
    """Give pooled encoder features unit scaling beside the original static normalizer."""
    ids = [64 + labels.index(n) for n in ('q95', 'beta_n')]
    return _Scaling(np.r_[scaling.feature_mean, np.zeros(32)], np.r_[scaling.feature_scale, np.ones(32)],
                    scaling.output_mean[ids], scaling.output_scale[ids])


def fit_one(family: str, seed: int, data: tuple, scaling: _Scaling, normalization: tuple,
            device: jax.Device, root: Path, epochs: int = 60) -> dict:
    """Retain fixed-final weights with full optimizer and convolutional state."""
    active = family == 'learned_conv_encoder'
    name = f'{family}_{seed}'
    out = root / name
    out.mkdir(exist_ok=True)
    if (out / 'fit_report.json').exists():
        receipt = json.loads((out / 'fit_report.json').read_text())
        assert digest(out / 'network.npz') == receipt['network_sha256']
        assert digest(out / 'encoder.npz') == receipt['encoder_sha256']
        return receipt
    recipe = settings(seed, epochs)
    frozen = {'family': family, 'settings': recipe.__dict__, 'protocol_sha256': digest(HERE / 'candidate_protocol_frozen.json')}
    static, targets, raw = data
    mean, scale = (jnp.asarray(a, dtype=jnp.float32) for a in normalization)
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params = initial(init_key, recipe)
    state = (params, jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params), jnp.zeros((), jnp.int32))
    start_epoch = 0
    checkpoint = out / 'checkpoint.npz'
    if checkpoint.exists():
        state, key, start_epoch = restore(checkpoint, state, frozen)
    batches = len(static) // recipe.batch

    def loss(parameters, x, y, windows):
        prediction = forward(parameters, x, windows, mean, scale, active)
        error = jnp.where(jnp.isfinite(y), prediction - jnp.nan_to_num(y), 0.)
        return jnp.mean(jnp.sum(error * error * .05, axis=1))

    @jax.jit
    def one_pass(carry, order_key, inputs, wanted, windows):
        def update(inner, chosen):
            value, gradient = jax.value_and_grad(loss)(inner[0], inputs[chosen], wanted[chosen], windows[chosen])
            return fit.decayed_adam_step(inner, gradient, recipe.epochs * batches, recipe), value
        order = jax.random.permutation(order_key, len(static))[:batches * recipe.batch]
        carry, losses = jax.lax.scan(update, carry, order.reshape(batches, recipe.batch))
        return carry, jnp.mean(losses)

    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        key, order = jax.random.split(key)
        if epoch == start_epoch:
            with COMPILE:
                state, value = one_pass(state, order, static, targets, raw)
                value.block_until_ready()
        else:
            state, value = one_pass(state, order, static, targets, raw)
            value.block_until_ready()
        assert np.isfinite(float(value)), 'nonfinite loss; no numeric retry'
        with (out / 'passes.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': float(value), 'seconds': time.monotonic() - started}) + '\n')
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            save(checkpoint, state, key, epoch + 1, frozen)
        print(json.dumps({'task': name, 'epoch_completed': epoch + 1, 'device': str(device)}), flush=True)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state[0]))
    network = fit._export(state[0], scaling, recipe, SimpleNamespace(label_names=('q95', 'beta_n'), feature_layout='discharge'))
    network.save(out / 'network.npz')
    encoded = {'active': np.asarray(active), 'mean': np.asarray(normalization[0], dtype=float), 'scale': np.asarray(normalization[1], dtype=float)}
    for block in ('weights', 'biases'):
        for i, values in enumerate(state[0]['conv_' + block]):
            encoded[f'{block}_{i}'] = np.asarray(values, dtype=float)
    np.savez(out / 'encoder.npz', **encoded)
    receipt = {'name': name, 'family': family, 'status': 'fit_complete', 'training_frames': len(static),
               'epochs': epochs, 'resumed_from_epoch': start_epoch, 'selection': 'fixed_final_epoch',
               'network_sha256': digest(out / 'network.npz'), 'encoder_sha256': digest(out / 'encoder.npz'),
               'seconds': time.monotonic() - started, 'held_out_metrics_computed': False, 'settings': recipe.__dict__}
    write_json(out / 'fit_report.json', receipt)
    return receipt


def worker(assignment: tuple) -> list:
    """Reuse each card's half-precision raw corpus for assigned scalar fits."""
    card, tasks, out = assignment
    device = jax.devices()[card]
    receipts = []
    with jax.default_device(device):
        data = upload(STATE['data'], STATE['scaling'], device)
        for family, seed in tasks:
            failure = out / f'{family}_{seed}_failure.json'
            if failure.exists():
                receipts.append(json.loads(failure.read_text()))
                continue
            try:
                receipts.append(fit_one(family, seed, data, STATE['scalar_scaling'], STATE['normalization'], device, out))
            except Exception as error:
                receipt = {'name': f'{family}_{seed}', 'status': 'failed_no_numeric_retry', 'type': type(error).__name__, 'error': str(error)}
                write_json(failure, receipt)
                receipts.append(receipt)
            gc.collect()
    return receipts


def qualify(out: Path) -> None:
    """Verify real decoded sequences, paired control loss and CPU/GPU predictions."""
    device = jax.devices()[0]
    assert device.platform in ('gpu', 'rocm')
    with jax.default_device(device):
        all_data = upload(STATE['data'], STATE['scaling'], device)
        batch = tuple(a[:2048] for a in all_data)
        assert np.array_equal(np.asarray(window(batch[2])), unpack(np.asarray(batch[2])))
        mean, scale = [jnp.asarray(a, dtype=jnp.float32) for a in STATE['normalization']]
        key, init_key = jax.random.split(jax.random.key(400))
        parameters = initial(init_key, settings(400, 1))
        old_loss, old_grad = jax.value_and_grad(fit._loss)(parameters, jnp.pad(batch[0], ((0, 0), (0, 32))), batch[1],
                             jnp.full(2, .05, dtype=jnp.float32), 0., key, jnp.eye(2, dtype=jnp.float32))

        def control_loss(params):
            errors = forward(params, batch[0], batch[2], mean, scale, False) - batch[1]
            return jnp.mean(jnp.sum(errors * errors * .05, axis=1))

        value, grad = jax.value_and_grad(control_loss)(parameters)
        assert np.allclose(np.asarray(value), np.asarray(old_loss), atol=1e-7, rtol=1e-6)
        assert all(np.allclose(np.asarray(a), np.asarray(b), atol=1e-7, rtol=1e-6) for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(old_grad), strict=True))
        results = []
        for family in ('static_matched_conv_control', 'learned_conv_encoder'):
            receipt = fit_one(family, 400, batch, STATE['scalar_scaling'], STATE['normalization'], device, out, epochs=1)
            model = TemporalNetwork(out / receipt['name'])
            raw_static = STATE['data']['base']['training']['features'][STATE['data']['rows']['training'][:128]].astype(float)
            raw_window = np.asarray(batch[2][:128])
            labels = model.predict(raw_static, raw_window)
            cpu = np.column_stack([labels[n] for n in ('q95', 'beta_n')])
            params = {block: [jnp.asarray(a, dtype=jnp.float64) for a in getattr(model.network, block)] for block in ('weights', 'biases')}
            params['linear'] = jnp.asarray(model.network.linear, dtype=jnp.float64)
            params['conv_weights'] = [jnp.asarray(a) for a in model.weights]
            params['conv_biases'] = [jnp.asarray(a) for a in model.biases]
            standardized = model.network.standardised(np.pad(raw_static, ((0, 0), (0, 32))))[:, :582]
            gpu = np.asarray(forward(params, jnp.asarray(standardized), jnp.asarray(raw_window), jnp.asarray(model.mean), jnp.asarray(model.scale), model.active))
            gpu = gpu * model.network.output_scale + model.network.output_mean
            difference = float(np.max(np.abs(cpu - gpu)))
            assert np.allclose(cpu, gpu, atol=1e-10, rtol=1e-10), difference
            recipe = settings(400, 1)
            frozen = {'family': family, 'settings': recipe.__dict__, 'protocol_sha256': digest(HERE / 'candidate_protocol_frozen.json')}
            original = initial(init_key, recipe)
            template = (original, jax.tree.map(jnp.zeros_like, original), jax.tree.map(jnp.zeros_like, original), jnp.zeros((), jnp.int32))
            restored, _, epoch = restore(out / receipt['name'] / 'checkpoint.npz', template, frozen)
            assert epoch == 1
            for block in ('weights', 'biases'):
                assert all(np.array_equal(np.asarray(a, dtype=float), b) for a, b in zip(restored[0][block], getattr(model.network, block), strict=True))
                assert all(np.array_equal(np.asarray(a, dtype=float), b) for a, b in zip(restored[0]['conv_' + block], getattr(model, block), strict=True))
            results.append({'family': family, 'cpu_gpu_double_max_abs': difference, 'checkpoint_exact': True})
        write_json(out / 'qualification.json', {'status': 'paired_temporal_real_ROCm_and_portable_double_qualified',
                    'input_window_reconstruction_exact': True, 'source_static_control_loss_gradient_parity': True,
                    'results': results, 'development_metrics_computed': False})


def main() -> None:
    """Qualify the real sequence implementation or finish all 16 declared fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    STATE['data'] = initialize()
    STATE['scaling'] = scaling_for(STATE['data'], True)
    STATE['scalar_scaling'] = scalar_scaling(STATE['scaling'], STATE['data']['metas']['training']['label_names'])
    with np.load(HERE / 'temporal/normalization.npz', allow_pickle=False) as stored:
        STATE['normalization'] = stored['mean'].copy(), stored['scale'].copy()
    if args.qualify:
        qualify(args.out)
        return
    assert json.loads((HERE / 'temporal_qualification/qualification.json').read_text())['status'] == 'paired_temporal_real_ROCm_and_portable_double_qualified'
    tasks = [(family, seed) for seed in range(400, 408) for family in ('static_matched_conv_control', 'learned_conv_encoder')]
    assert len(jax.devices()) == 3
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = [r for batch in pool.map(worker, [(i, tasks[i::3], args.out) for i in range(3)]) for r in batch]
    write_json(args.out / 'receipt.json', {'status': 'all16_temporal_candidate_fits_terminal', 'tasks': receipts,
                   'successful': sum(r['status'] == 'fit_complete' for r in receipts), 'development_metrics_computed': False})


if __name__ == '__main__':
    main()
