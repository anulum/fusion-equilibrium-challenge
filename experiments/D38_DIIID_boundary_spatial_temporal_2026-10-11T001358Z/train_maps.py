# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — paired boundary and spatial candidate fits
"""Fit all four frozen map arms without examining development metrics."""
from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_dilation

from campaign_common import HERE, PKG, digest, write_json
from training_data import COMPILE, initialize, scaling_for
from candidate_tables import upload
from candidate_state import save, restore
from map_objectives import effective_decoder, objective
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling
from fec_adapter.diii_d_accuracy.network import Network
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from fec_adapter.vessel import envelope

STATE = {}


def geometry(device: jax.Device, dtype=jnp.float32) -> tuple:
    """Load the existing orthonormal basis and one-cell vessel band."""
    basis = FluxBasis.load(PKG / 'models/flux_basis.npz')
    directions = np.asarray(basis.directions[:128]).reshape(128, 4225)
    mask = binary_dilation(envelope('DIII-D'), iterations=1).reshape(4225)
    return jax.device_put(jnp.asarray(directions, dtype=dtype), device), jax.device_put(jnp.asarray(mask, dtype=dtype), device)


def settings(seed: int, epochs: int = 60) -> FitSettings:
    """Return the prospective map schedule shared by every arm."""
    return FitSettings(seed=seed, epochs=epochs, gradient_limit=1., dropout=0.)


def initial(key: jax.Array, recipe: FitSettings, family: str, basis: jax.Array, mask: jax.Array) -> dict:
    """Share all original 142 output weights across paired arms."""
    parameters = fit.initial_parameters(key, 582, 142, recipe)
    if family == 'spatial':
        parameters['weights'][-1] = jnp.pad(parameters['weights'][-1], ((0, 16), (0, 0)))
        parameters['biases'][-1] = jnp.pad(parameters['biases'][-1], (0, 16))
        parameters['linear'] = jnp.pad(parameters['linear'], ((0, 16), (0, 0)))
        parameters['decoder'] = basis[45:61] * mask[None, :]
    return parameters


def export(parameters: dict, scaling: _Scaling, recipe: FitSettings, family: str, labels: tuple, out: Path, mask: jax.Array) -> None:
    """Export CPU double weights and the normalized learned dictionary."""
    extended = scaling
    names = labels
    if family == 'spatial':
        names = labels + tuple(f'spatial_{i:02d}' for i in range(16))
        extended = _Scaling(scaling.feature_mean, scaling.feature_scale,
                            np.r_[scaling.output_mean, np.zeros(16)],
                            np.r_[scaling.output_scale, np.full(16, scaling.output_scale[0])])
        np.save(out / 'decoder.npy', np.asarray(effective_decoder(parameters, mask), dtype=float), allow_pickle=False)
    network = fit._export(parameters, extended, recipe, SimpleNamespace(label_names=names, feature_layout='discharge'))
    network.save(out / 'network.npz')


def fit_one(family: str, seed: int, data: tuple, scaling: _Scaling, labels: tuple, device: jax.Device,
            root: Path, epochs: int = 60) -> dict:
    """Finish fixed-final training and retain failures without numeric retries."""
    name = f'{family}_{seed}'
    out = root / name
    out.mkdir(exist_ok=True)
    report = out / 'fit_report.json'
    if report.exists():
        receipt = json.loads(report.read_text())
        assert digest(out / 'network.npz') == receipt['network_sha256']
        return receipt
    recipe = settings(seed, epochs)
    frozen = {'family': family, 'settings': recipe.__dict__, 'protocol_sha256': digest(HERE / 'candidate_protocol_frozen.json')}
    x, y, teacher, points, frame, shot = data
    rows = shot if family in ('boundary_shot', 'spatial') else frame
    basis, mask = geometry(device)
    mean = jnp.asarray(scaling.output_mean[:64], dtype=jnp.float32)
    common_scale = jnp.asarray(scaling.output_scale[0], dtype=jnp.float32)
    key = jax.random.key(seed)
    key, initialization = jax.random.split(key)
    parameters = initial(initialization, recipe, family, basis, mask)
    state = (parameters, jax.tree.map(jnp.zeros_like, parameters), jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))
    start_epoch = 0
    checkpoint = out / 'checkpoint.npz'
    if checkpoint.exists():
        state, key, start_epoch = restore(checkpoint, state, frozen)
    batches = len(x) // recipe.batch

    @jax.jit
    def one_pass(carry, order_key, inputs, targets, teachers, positions, row_weights):
        def update(inner, chosen):
            loss, gradient = jax.value_and_grad(objective)(inner[0], inputs[chosen], targets[chosen], teachers[chosen],
                              positions[chosen], row_weights[chosen], basis, mean, common_scale, mask, family)
            return fit.decayed_adam_step(inner, gradient, recipe.epochs * batches, recipe), loss
        order = jax.random.permutation(order_key, len(x))[:batches * recipe.batch]
        carry, losses = jax.lax.scan(update, carry, order.reshape(batches, recipe.batch))
        return carry, jnp.mean(losses)

    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        key, order = jax.random.split(key)
        if epoch == start_epoch:
            with COMPILE:
                state, loss = one_pass(state, order, x, y, teacher, points, rows)
                loss.block_until_ready()
        else:
            state, loss = one_pass(state, order, x, y, teacher, points, rows)
            loss.block_until_ready()
        assert np.isfinite(float(loss)), 'nonfinite loss; no numeric retry'
        with (out / 'passes.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'loss': float(loss), 'seconds': time.monotonic() - started}) + '\n')
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            save(checkpoint, state, key, epoch + 1, frozen)
        print(json.dumps({'task': name, 'epoch_completed': epoch + 1, 'device': str(device)}), flush=True)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state[0]))
    export(state[0], scaling, recipe, family, labels, out, mask)
    receipt = {'name': name, 'family': family, 'status': 'fit_complete', 'epochs': epochs,
               'resumed_from_epoch': start_epoch, 'training_frames': len(x), 'selection': 'fixed_final_epoch',
               'network_sha256': digest(out / 'network.npz'), 'seconds': time.monotonic() - started,
               'held_out_metrics_computed': False, 'settings': recipe.__dict__}
    if family == 'spatial':
        receipt['decoder_sha256'] = digest(out / 'decoder.npy')
    write_json(report, receipt)
    return receipt


def worker(assignment: tuple) -> list:
    """Reuse each card's eligible corpus across its assigned paired fits."""
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
                receipts.append(fit_one(family, seed, data, STATE['scaling'], STATE['labels'], device, out))
            except Exception as error:
                receipt = {'name': f'{family}_{seed}', 'status': 'failed_no_numeric_retry',
                           'type': type(error).__name__, 'error': str(error)}
                write_json(failure, receipt)
                receipts.append(receipt)
            gc.collect()
    return receipts


def qualify(out: Path) -> None:
    """Exercise real batches, source loss gradients and portable double exports."""
    device = jax.devices()[0]
    assert device.platform in ('rocm', 'gpu')
    with jax.default_device(device):
        uploaded = upload(STATE['data'], STATE['scaling'], device)
        batch = tuple(a[:4096] for a in uploaded)
        basis, mask = geometry(device)
        scaling = STATE['scaling']
        recipe = settings(400, 1)
        key, init_key = jax.random.split(jax.random.key(400))
        params = initial(init_key, recipe, 'control', basis, mask)
        mean = jnp.asarray(scaling.output_mean[:64], dtype=jnp.float32)
        common = jnp.asarray(scaling.output_scale[0], dtype=jnp.float32)
        weight = jnp.r_[jnp.ones(64, dtype=jnp.float32), jnp.full(78, .05, dtype=jnp.float32)]
        ours, gradient = jax.value_and_grad(objective)(params, batch[0], batch[1], batch[2], batch[3], batch[4], basis, mean, common, mask, 'control')
        old, old_gradient = jax.value_and_grad(fit._loss)(params, batch[0], batch[1], weight, 0., key, jnp.eye(142, dtype=jnp.float32))
        assert np.array_equal(np.asarray(ours), np.asarray(old))
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(old_gradient), strict=True))
        results = []
        for family in ('control', 'boundary_frame', 'boundary_shot', 'spatial'):
            receipt = fit_one(family, 400, batch, scaling, STATE['labels'], device, out, epochs=1)
            network = Network.load(out / receipt['name'] / 'network.npz')
            raw = STATE['data']['base']['training']['features'][STATE['data']['rows']['training'][:128]].astype(float)
            params = {block: [jnp.asarray(a, dtype=jnp.float64) for a in getattr(network, block)] for block in ('weights', 'biases')}
            params['linear'] = jnp.asarray(network.linear, dtype=jnp.float64)
            gpu = np.asarray(fit.forward(params, jnp.asarray(network.standardised(raw)), 0., None)) * network.output_scale + network.output_mean
            coo, heads = network.predict(raw)
            cpu = np.column_stack([coo] + [heads[n] for n in network.label_names])
            difference = float(np.max(np.abs(cpu - gpu)))
            assert np.allclose(cpu, gpu, atol=1e-10, rtol=1e-10), difference
            frozen = {'family': family, 'settings': recipe.__dict__, 'protocol_sha256': digest(HERE / 'candidate_protocol_frozen.json')}
            original = initial(init_key, recipe, family, basis, mask)
            template = (original, jax.tree.map(jnp.zeros_like, original), jax.tree.map(jnp.zeros_like, original), jnp.zeros((), jnp.int32))
            restored, _, epoch = restore(out / receipt['name'] / 'checkpoint.npz', template, frozen)
            assert epoch == 1
            for block in ('weights', 'biases'):
                assert all(np.array_equal(np.asarray(a, dtype=float), b) for a, b in zip(restored[0][block], getattr(network, block), strict=True))
            assert np.array_equal(np.asarray(restored[0]['linear'], dtype=float), network.linear)
            if family == 'spatial':
                decoded = np.load(out / receipt['name'] / 'decoder.npy', allow_pickle=False)
                assert np.array_equal(decoded, np.asarray(effective_decoder(restored[0], mask), dtype=float))
                assert not np.any(decoded[:, np.asarray(mask) == 0])
                field_cpu = cpu[:, 142:] @ decoded
                field_gpu = np.asarray(jnp.asarray(cpu[:, 142:]) @ jnp.asarray(decoded))
                assert np.allclose(field_cpu, field_gpu, atol=1e-10, rtol=1e-10)
            results.append({'family': family, 'cpu_gpu_double_max_abs': difference, 'checkpoint_exact': True})
        write_json(out / 'qualification.json', {'status': 'all_four_map_objectives_real_ROCm_qualified',
                   'unmodified_control_loss_and_gradient_exact': True, 'results': results,
                   'development_metrics_computed': False})


def main() -> None:
    """Qualify implementations or finish all 32 predeclared map fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    STATE['data'] = initialize()
    STATE['scaling'] = scaling_for(STATE['data'], True)
    STATE['labels'] = tuple(STATE['data']['metas']['training']['label_names'])
    if args.qualify:
        qualify(args.out)
        return
    assert json.loads((HERE / 'map_qualification/qualification.json').read_text())['status'] == 'all_four_map_objectives_real_ROCm_qualified'
    tasks = [(family, seed) for seed in range(400, 408) for family in ('control', 'boundary_frame', 'boundary_shot', 'spatial')]
    assert len(jax.devices()) == 3
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [r for batch in pool.map(worker, [(i, tasks[i::3], args.out) for i in range(3)]) for r in batch]
    write_json(args.out / 'receipt.json', {'status': 'all32_map_candidate_fits_terminal', 'tasks': results,
                                        'successful': sum(r['status'] == 'fit_complete' for r in results),
                                        'development_metrics_computed': False})


if __name__ == '__main__':
    main()
