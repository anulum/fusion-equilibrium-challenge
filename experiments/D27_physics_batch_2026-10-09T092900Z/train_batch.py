# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — streamed local ROCm physics campaign
"""Run all frozen fits; do not compute validation comparisons during training."""

import argparse
import gc
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import (ARMS, FEATURES, HERE, PARTS, PKG, base_maps, digest,
                             extra_maps, feature_block, metadata, row_ids, write_json)
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from fec_adapter.vessel import envelope, grids

TRANSFER = threading.Lock()
COMPILE = threading.Lock()
CHUNK = 16384
STATE = {}


class Moments:
    def __init__(self, size):
        self.count = np.zeros(size)
        self.mean = np.zeros(size)
        self.m2 = np.zeros(size)

    def add(self, values):
        finite = np.isfinite(values)
        count = finite.sum(axis=0).astype(float)
        mean = np.where(finite, values, 0.).sum(axis=0) / np.maximum(count, 1.)
        m2 = np.where(finite, values - mean, 0.)**2
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2.sum(axis=0) + delta**2 * self.count * count / np.maximum(total, 1.)
        self.mean += delta * count / np.maximum(total, 1.)
        self.count = total

    def scale(self):
        spread = np.sqrt(np.maximum(0., self.m2 / np.maximum(self.count, 1.)))
        return np.where(spread > 1e-6, spread, 1.)


def fold_statistics(fold, base, extra, metas):
    rows = {part: row_ids(metas[part], fold) for part in ('training', 'calibration')}
    features = Moments(FEATURES)
    targets = Moments(142)
    coils = Moments(64)
    for part, ids in rows.items():
        for start in range(0, len(ids), CHUNK):
            take = ids[start:start + CHUNK]
            values = feature_block(base[part], extra[part], take, 'control')
            values[:, 582:689] = extra[part]['pressure'][take]
            values[:, 689:] = extra[part]['memory'][take]
            features.add(values)
            targets.add(np.column_stack([base[part]['coordinates'][take, :64], base[part]['labels'][take]]).astype(float))
            coils.add(extra[part]['coil_coordinates'][take].astype(float))
    assert np.all(targets.count > 0) and np.all(features.count > 0)
    coordinate_common = float(np.sqrt(np.sum(targets.m2[:64] / targets.count[:64])))
    assert coordinate_common > 0
    scale = targets.scale()
    scale[:64] = coordinate_common
    return rows, features, targets, coils, scale


def scaling_of(arm, features, targets, coils, output_scale):
    mean, scale = features.mean.copy(), features.scale()
    if arm != 'pressure':
        mean[582:689], scale[582:689] = 0., 1.
    if arm != 'memory':
        mean[689:], scale[689:] = 0., 1.
    output_mean = targets.mean.copy()
    if arm == 'residual':
        output_mean[:64] -= coils.mean
    return _Scaling(mean, scale, output_mean, output_scale.copy())


def geometry_mix():
    basis = FluxBasis.load(PKG / 'models/flux_basis.npz').truncated(64)
    radius, height = grids('DIII-D')
    directions = basis.directions.reshape(64, 65, 65)
    dz, dr = np.gradient(directions, height, radius, axis=(1, 2))
    inside = envelope('DIII-D')
    metric = dr[:, inside] @ dr[:, inside].T + dz[:, inside] @ dz[:, inside].T
    metric *= 64 / float(np.trace(metric))
    mix = np.eye(142)
    mix[:64, :64] = np.linalg.cholesky(np.eye(64) + .2 * metric)
    return mix.astype(np.float32)


@partial(jax.jit, donate_argnums=(0,))
def put_block(array, block, start, column):
    return jax.lax.dynamic_update_slice(array, block, (start, column))


def upload(rows, base, extra, scaling, arm, device, inputs=None, targets=None):
    n = sum(len(ids) for ids in rows.values())
    with TRANSFER, jax.default_device(device):
        if inputs is None:
            inputs = jnp.zeros((n, FEATURES), dtype=jnp.float32)
        if targets is None:
            targets = jnp.zeros((n, 142), dtype=jnp.float32)
        position = 0
        for part, ids in rows.items():
            for offset in range(0, len(ids), CHUNK):
                take = ids[offset:offset + CHUNK]
                x = feature_block(base[part], extra[part], take, arm)
                x = np.clip((x - scaling.feature_mean) / scaling.feature_scale, -8, 8).astype(np.float32)
                y = np.column_stack([base[part]['coordinates'][take, :64], base[part]['labels'][take]]).astype(float)
                if arm == 'residual':
                    y[:, :64] -= extra[part]['coil_coordinates'][take]
                y = ((y - scaling.output_mean) / scaling.output_scale).astype(np.float32)
                inputs = put_block(inputs, jax.device_put(x, device), position, 0)
                targets = put_block(targets, jax.device_put(y, device), position, 0)
                inputs.block_until_ready()
                targets.block_until_ready()
                position += len(take)
        assert position == n
    return inputs, targets


def fit_one(name, settings, inputs, targets, scaling, names, mix, device, out):
    destination = out / name
    destination.mkdir(exist_ok=False)
    n = len(inputs)
    batches = n // settings.batch
    steps = settings.epochs * batches
    weights = jnp.asarray(np.r_[np.ones(64), np.full(78, settings.label_weight)], dtype=jnp.float32)
    matrix = jnp.asarray(mix, dtype=jnp.float32)
    key = jax.random.key(settings.seed)
    key, initialization = jax.random.split(key)
    parameters = fit.initial_parameters(initialization, FEATURES, 142, settings)
    state = (parameters, jax.tree.map(jnp.zeros_like, parameters),
             jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))

    @jax.jit
    def one_pass(carry, order_key, table, wanted, loss_weights, loss_matrix):
        def update(inner, chosen):
            loss, gradient = jax.value_and_grad(fit._loss)(inner[0], table[chosen], wanted[chosen],
                loss_weights, 0., jax.random.fold_in(key, inner[3]), loss_matrix)
            return fit.decayed_adam_step(inner, gradient, steps, settings), loss
        order = jax.random.permutation(order_key, table.shape[0])[:batches * settings.batch]
        state, losses = jax.lax.scan(update, carry, order.reshape(batches, settings.batch))
        return state, jnp.mean(losses)

    start = time.monotonic()
    for epoch in range(settings.epochs):
        key, order = jax.random.split(key)
        if epoch == 0:
            with COMPILE:
                state, loss = one_pass(state, order, inputs, targets, weights, matrix)
                loss.block_until_ready()
        else:
            state, loss = one_pass(state, order, inputs, targets, weights, matrix)
            loss.block_until_ready()
        assert np.isfinite(float(loss)), 'nonfinite training loss; no automatic numeric retry'
        with (destination / 'passes.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, 'training_loss': float(loss),
                                     'elapsed_seconds': time.monotonic() - start}) + '\n')
        print(json.dumps({'epoch_completed': epoch + 1, 'task': name, 'device': str(device)}), flush=True)
    network = fit._export(state[0], scaling, settings,
                         SimpleNamespace(label_names=names, feature_layout='discharge'))
    for array in network.weights + network.biases + (network.linear,):
        assert np.isfinite(array).all()
    network.save(destination / 'network.npz')
    receipt = {'status': 'fixed_final_epoch_fit_complete', 'name': name, 'training_frames': n,
               'settings': settings.__dict__, 'device': str(device), 'epochs_completed': settings.epochs,
               'network_sha256': digest(destination / 'network.npz'), 'seconds': time.monotonic() - start,
               'calibration_selection': False, 'validation_metrics_computed': False}
    write_json(destination / 'fit_report.json', receipt)
    del state, parameters, network
    gc.collect()
    return receipt


def fold_worker(task):
    fold, card, out = task
    device = jax.devices()[card]
    base, extra, metas = STATE['base'], STATE['extra'], STATE['metas']
    rows, features, targets_moments, coils, output_scale = fold_statistics(fold, base, extra, metas)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    receipts = []
    with jax.default_device(device):
        inputs = targets = None
        for arm in ARMS:
            scaling = scaling_of(arm, features, targets_moments, coils, output_scale)
            inputs, targets = upload(rows, base, extra, scaling, arm, device, inputs, targets)
            matrix = geometry_mix() if arm == 'gradient' else np.eye(142, dtype=np.float32)
            for seed in protocol['seeds']:
                name = f'{arm}_seed{seed}_k{fold}'
                settings = FitSettings(**protocol['settings'], seed=seed)
                print(json.dumps({'task_start': name, 'training_frames': len(inputs), 'device': str(device)}), flush=True)
                try:
                    receipt = fit_one(name, settings, inputs, targets, scaling,
                                      tuple(metas['training']['label_names']), matrix, device, out)
                except Exception as error:
                    receipt = {'status': 'failed_no_numeric_retry', 'name': name,
                               'type': type(error).__name__, 'error': str(error)}
                    write_json(out / f'{name}_failure.json', receipt)
                    print(json.dumps({'task_failed': name, 'type': type(error).__name__}), flush=True)
                receipts.append(receipt)
        del inputs, targets
        gc.collect()
    return receipts


def qualification(out):
    device = jax.devices()[0]
    assert device.platform in ('gpu', 'rocm')
    rng = np.random.default_rng(200)
    features = rng.normal(size=(1024, FEATURES)).astype(np.float32)
    targets = rng.normal(size=(1024, 142)).astype(np.float32)
    targets[::3, 65] = np.nan
    rows = {'training': np.arange(1024)}
    source = {'training': {'features': features[:, :582], 'coordinates': targets[:, :64], 'labels': targets[:, 64:]}}
    extra = {'training': {'pressure': np.zeros((1024, 107), np.float32),
                         'memory': np.zeros((1024, 76), np.float32), 'coil_coordinates': np.zeros((1024, 64), np.float32)}}
    scaling = _Scaling(np.zeros(FEATURES), np.ones(FEATURES), np.zeros(142), np.ones(142))
    with jax.default_device(device):
        x, y = upload(rows, source, extra, scaling, 'control', device)
        assert np.array_equal(np.asarray(x)[:, :582], np.clip(features[:, :582], -8, 8))
        assert np.array_equal(np.asarray(y), targets, equal_nan=True)
        small = FitSettings(width=32, depth=2, batch=512, epochs=1, gradient_limit=1., seed=200)
        parameters = fit.initial_parameters(jax.random.key(200), FEATURES, 142, small)
        network = fit._export(parameters, scaling, small,
                             SimpleNamespace(label_names=tuple(f'label{i}' for i in range(78)), feature_layout='discharge'))
        host_x = np.asarray(x[:128], dtype=float)
        double_parameters = jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.float64), parameters)
        gpu = np.asarray(fit.forward(double_parameters, jnp.asarray(host_x), 0., None))
        coordinates, labels = network.predict(host_x)
        cpu = np.column_stack([coordinates, *[labels[n] for n in network.label_names]])
        difference = float(np.max(np.abs(cpu - gpu)))
        assert np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), difference
        receipt = fit_one('qualification_only', small, x, y, scaling,
                          tuple(f'label{i}' for i in range(78)), np.eye(142, dtype=np.float32), device, out)
    write_json(out / 'qualification.json', {'status': 'ROCm_transfer_export_optimizer_qualified',
               'device': str(device), 'forward_CPU_GPU_double_max_abs_difference': difference,
               'fit': receipt, 'scientific_campaign_fit': False})


def large_qualification(out):
    device = jax.devices()[0]
    assert device.platform in ('gpu', 'rocm')
    n = 1_100_000
    with jax.default_device(device):
        x = jnp.zeros((n, FEATURES), dtype=jnp.float32)
        y = jnp.zeros((n, 142), dtype=jnp.float32)
        rng = np.random.default_rng(200)
        x = put_block(x, jax.device_put(rng.normal(size=(CHUNK, FEATURES)).astype(np.float32), device), 0, 0)
        target = rng.normal(size=(CHUNK, 142)).astype(np.float32)
        target[::3, 65] = np.nan
        y = put_block(y, jax.device_put(target, device), 0, 0)
        x.block_until_ready()
        y.block_until_ready()
        scaling = _Scaling(np.zeros(FEATURES), np.ones(FEATURES), np.zeros(142), np.ones(142))
        settings = FitSettings(width=512, depth=4, batch=4096, epochs=1, gradient_limit=1., seed=200)
        receipt = fit_one('large_qualification_only', settings, x, y, scaling,
                          tuple(f'label{i}' for i in range(78)), np.eye(142, dtype=np.float32), device, out)
    write_json(out / 'qualification.json', {'status': 'full_size_ROCm_resource_path_qualified',
        'device': str(device), 'rows': n, 'input_columns': FEATURES,
        'GPU_table_bytes': n * (FEATURES + 142) * 4, 'fit': receipt,
        'synthetic_resource_test_only': True, 'scientific_campaign_fit': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--qualify-large', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    assert all(d.platform in ('gpu', 'rocm') for d in jax.devices())
    if args.qualify:
        qualification(args.out)
        return 0
    if args.qualify_large:
        large_qualification(args.out)
        return 0
    assert len(jax.devices()) == 3
    prepared = json.loads((HERE / 'prepared/receipt.json').read_text())
    assert prepared['status'] == 'all_physics_inputs_prepared'
    STATE.update(base=base_maps(), extra=extra_maps(), metas={p: metadata(p) for p in PARTS})
    write_json(args.out / 'manifest.json', {'protocol_sha256': digest(HERE / 'protocol.json'),
        'script_sha256': digest(__file__), 'devices': [str(d) for d in jax.devices()],
        'expected_tasks': 50, 'all_comparative_evaluation_deferred': True})
    receipts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        def card_jobs(card):
            return [receipt for fold in range(card, 5, 3)
                    for receipt in fold_worker((fold, card, args.out))]
        for part in pool.map(card_jobs, range(3)):
            receipts.extend(part)
    assert len(receipts) == 50
    write_json(args.out / 'receipt.json', {'status': 'all_50_neural_tasks_terminal', 'tasks': receipts,
        'successful': sum(r['status'] == 'fixed_final_epoch_fit_complete' for r in receipts),
        'comparative_metrics_computed': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
