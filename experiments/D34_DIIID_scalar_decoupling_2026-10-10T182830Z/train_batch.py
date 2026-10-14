# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — independent scalar-task training
"""Train thirty scalar-only models before any comparison."""

import argparse
import gc
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import ARMS, D27, FEATURES, HERE, PARTS, base_maps, digest, feature_block, metadata, row_ids, write_json
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling, label_loss_weights
from fec_adapter.diii_d_accuracy.network import Network

spec = importlib.util.spec_from_file_location('d27_streaming_training', D27 / 'train_batch.py')
stream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stream
spec.loader.exec_module(stream)
STATE = {}


def controls() -> list[dict]:
    """Verify the ten existing paired controls and retain their exact lineage."""
    fits = json.loads((D27 / 'fits/receipt.json').read_text())
    readouts = json.loads((D27 / 'readouts/receipt.json').read_text())
    original = {r['name']: r for r in fits['tasks']}
    rows = [r for r in readouts['tasks'] if r['arm'] == 'control']
    assert len(rows) == 10 and all(r['status'] == 'geometry_readout_complete' for r in rows)
    result = []
    for row in rows:
        name = row['name']
        initial, final = D27 / 'fits' / name / 'network.npz', D27 / 'readouts' / name / 'network.npz'
        assert digest(initial) == original[name]['network_sha256']
        assert digest(final) == row['network_sha256']
        result.append({'name': name, 'fold': row['fold'], 'fit_path': str(initial),
                       'fit_sha256': original[name]['network_sha256'], 'readout_path': str(final),
                       'readout_sha256': row['network_sha256']})
    return result


def fold_data(fold: int, device):
    """Reuse the existing control's normalization and exact permitted rows."""
    base, metas = STATE['base'], STATE['metas']
    rows = {part: row_ids(metas[part], fold) for part in ('training', 'calibration')}
    old = Network.load(D27 / f'fits/control_seed200_k{fold}/network.npz')
    other = Network.load(D27 / f'fits/control_seed201_k{fold}/network.npz')
    for name in ('feature_mean', 'feature_scale', 'output_mean', 'output_scale'):
        assert np.array_equal(getattr(old, name), getattr(other, name))
    assert old.coordinates == 64 and len(old.feature_mean) == FEATURES
    assert np.array_equal(old.feature_mean[582:], np.zeros(183))
    assert np.array_equal(old.feature_scale[582:], np.ones(183))
    scaling = _Scaling(old.feature_mean, old.feature_scale, old.output_mean, old.output_scale)
    receipt = json.loads((D27 / f'fits/control_seed200_k{fold}/fit_report.json').read_text())
    assert sum(len(r) for r in rows.values()) == receipt['training_frames']
    x, y = stream.upload(rows, base, {part: {} for part in PARTS}, scaling, 'control', device)
    selected = [64 + tuple(metas['training']['label_names']).index(n) for n in ('q95', 'beta_n')]
    scalar_scaling = _Scaling(scaling.feature_mean, scaling.feature_scale,
                              scaling.output_mean[selected], scaling.output_scale[selected])
    scalar_targets = y[:, selected]
    scalar_targets.block_until_ready()
    del y
    return rows, scalar_scaling, x, scalar_targets


def settings_for(arm: str, seed: int, epochs: int | None = None) -> FitSettings:
    """Use identical optimizer settings for every declared scalar task."""
    settings = FitSettings(**STATE['protocol']['settings'], seed=seed)
    return settings if epochs is None else replace(settings, epochs=epochs)


def save_checkpoint(path: Path, state, key, epoch: int, settings: FitSettings) -> None:
    """Save optimizer and permutation state atomically on Samsung."""
    arrays = {'epoch': np.asarray(epoch), 'key': np.asarray(jax.random.key_data(key)),
              'step': np.asarray(state[3]), 'settings': np.asarray(json.dumps(settings.__dict__, sort_keys=True))}
    for name, parameters in zip(('parameters', 'first', 'second'), state[:3], strict=True):
        arrays[name + '_linear'] = np.asarray(parameters['linear'])
        for block in ('weights', 'biases'):
            for i, value in enumerate(parameters[block]):
                arrays[f'{name}_{block}_{i}'] = np.asarray(value)
    temporary = path.with_name('checkpoint.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def restore_checkpoint(path: Path, settings: FitSettings):
    """Restore only an identical recipe, optimizer and random state."""
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored['settings']) == json.dumps(settings.__dict__, sort_keys=True)
        states = []
        for name in ('parameters', 'first', 'second'):
            states.append({block: [jnp.asarray(stored[f'{name}_{block}_{i}']) for i in range(settings.depth + 1)]
                           for block in ('weights', 'biases')})
            states[-1]['linear'] = jnp.asarray(stored[name + '_linear'])
        state = (*states, jnp.asarray(stored['step']))
        return state, jax.random.wrap_key_data(jnp.asarray(stored['key'])), int(stored['epoch'])


def fit_one(name: str, settings: FitSettings, x, y, scaling, names: tuple, device, root: Path) -> dict:
    """Fit the fixed final epoch; isolated nonfinite fits never retry numerically."""
    destination = root / name
    destination.mkdir(exist_ok=True)
    report = destination / 'fit_report.json'
    if report.exists():
        receipt = json.loads(report.read_text())
        assert digest(destination / 'network.npz') == receipt['network_sha256']
        return receipt
    batches = len(x) // settings.batch
    steps = settings.epochs * batches
    weights = jnp.full(len(names), settings.label_weight, dtype=jnp.float32)
    matrix = jnp.eye(len(names), dtype=jnp.float32)
    key = jax.random.key(settings.seed)
    key, initialization = jax.random.split(key)
    checkpoint = destination / 'checkpoint.npz'
    start_epoch = 0
    if checkpoint.exists():
        state, key, start_epoch = restore_checkpoint(checkpoint, settings)
    else:
        parameters = fit.initial_parameters(initialization, FEATURES, len(names), settings)
        state = (parameters, jax.tree.map(jnp.zeros_like, parameters),
                 jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))

    @jax.jit
    def one_pass(carry, order_key, inputs, targets):
        def update(inner, chosen):
            loss, gradient = jax.value_and_grad(fit._loss)(inner[0], inputs[chosen], targets[chosen],
                weights, 0., jax.random.fold_in(initialization, inner[3]), matrix)
            return fit.decayed_adam_step(inner, gradient, steps, settings), loss
        order = jax.random.permutation(order_key, inputs.shape[0])[:batches * settings.batch]
        carry, losses = jax.lax.scan(update, carry, order.reshape(batches, settings.batch))
        return carry, jnp.mean(losses)

    started = time.monotonic()
    for epoch in range(start_epoch, settings.epochs):
        key, order = jax.random.split(key)
        if epoch == start_epoch:
            with stream.COMPILE:
                state, loss = one_pass(state, order, x, y)
                loss.block_until_ready()
        else:
            state, loss = one_pass(state, order, x, y)
            loss.block_until_ready()
        assert np.isfinite(float(loss)), 'nonfinite training loss; isolated terminal failure'
        with (destination / 'passes.jsonl').open('a') as log:
            log.write(json.dumps({'epoch': epoch + 1, 'training_loss': float(loss),
                                  'elapsed_seconds_this_attempt': time.monotonic() - started}) + '\n')
        if (epoch + 1) % 10 == 0 or epoch + 1 == settings.epochs:
            save_checkpoint(checkpoint, state, key, epoch + 1, settings)
        print(json.dumps({'task': name, 'epoch_completed': epoch + 1, 'device': str(device)}), flush=True)
    network = fit._export(state[0], scaling, settings, SimpleNamespace(label_names=names, feature_layout='discharge'))
    assert all(np.isfinite(a).all() for a in network.weights + network.biases + (network.linear,))
    network.save(destination / 'network.npz')
    receipt = {'name': name, 'status': 'fixed_final_epoch_fit_complete', 'training_frames': len(x),
               'settings': settings.__dict__, 'epochs_completed': settings.epochs,
               'resumed_from_epoch': start_epoch, 'network_sha256': digest(destination / 'network.npz'),
               'seconds_this_attempt': time.monotonic() - started, 'validation_metrics_computed': False, 'targets': list(names)}
    write_json(report, receipt)
    del state, network
    gc.collect()
    return receipt


def fold_worker(fold: int, card: int, out: Path) -> list[dict]:
    """Reuse one fold upload for joint and separate scalar estimators."""
    device = jax.devices()[card]
    with jax.default_device(device):
        rows, scaling, x, y = fold_data(fold, device)
        receipts = []
        for arm, indices in (('joint', (0, 1)), ('q95', (0,)), ('beta_n', (1,))):
            names = tuple(('q95', 'beta_n')[i] for i in indices)
            subset = _Scaling(scaling.feature_mean, scaling.feature_scale,
                              scaling.output_mean[list(indices)], scaling.output_scale[list(indices)])
            target = y[:, list(indices)]
            for seed in (200, 201):
                name = f'{arm}_seed{seed}_k{fold}'
                failed = out / (name + '_failure.json')
                if failed.exists():
                    receipts.append(json.loads(failed.read_text()))
                    continue
                try:
                    receipt = fit_one(name, settings_for(arm, seed), x, target, subset, names, device, out)
                except Exception as error:
                    receipt = {'name': name, 'status': 'failed_no_numeric_retry',
                               'type': type(error).__name__, 'error': str(error)}
                    write_json(failed, receipt)
                receipt['fold'] = fold
                receipt['arm'] = arm
                receipt['seed'] = seed
                receipts.append(receipt)
            del target
        del rows, scaling, x, y
        gc.collect()
        return receipts


def qualify(out: Path) -> None:
    """Exercise every scalar output shape with the full real fold data."""
    device = jax.devices()[0]
    assert device.platform in ('gpu', 'rocm')
    with jax.default_device(device):
        rows, scaling, x, y = fold_data(0, device)
        raw = feature_block(STATE['base']['training'], {}, rows['training'][:128], 'control')
        results = []
        for arm, indices in (('joint', (0, 1)), ('q95', (0,)), ('beta_n', (1,))):
            names = tuple(('q95', 'beta_n')[i] for i in indices)
            subset = _Scaling(scaling.feature_mean, scaling.feature_scale,
                              scaling.output_mean[list(indices)], scaling.output_scale[list(indices)])
            settings = settings_for(arm, 200, epochs=1)
            name = 'qualification_' + arm
            receipt = fit_one(name, settings, x, y[:, list(indices)], subset, names, device, out)
            network = Network.load(out / name / 'network.npz')
            assert network.coordinates == 0 and network.label_names == names
            normalized = network.standardised(raw)
            parameters = {k: [jnp.asarray(a, dtype=jnp.float64) for a in getattr(network, k)]
                          for k in ('weights', 'biases')}
            parameters['linear'] = jnp.asarray(network.linear, dtype=jnp.float64)
            gpu = np.asarray(fit.forward(parameters, jnp.asarray(normalized), 0., None))
            gpu = gpu * network.output_scale + network.output_mean
            coordinates, labels = network.predict(raw)
            cpu = np.column_stack([labels[n] for n in names])
            assert coordinates.shape == (len(raw), 0)
            difference = float(np.max(np.abs(cpu - gpu)))
            assert np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), difference
            restored, restored_key, epoch = restore_checkpoint(out / name / 'checkpoint.npz', settings)
            assert epoch == 1
            for block in ('weights', 'biases'):
                for saved, exported in zip(restored[0][block], getattr(network, block), strict=True):
                    assert np.array_equal(np.asarray(saved), exported)
            assert np.array_equal(np.asarray(restored[0]['linear']), network.linear)
            assert np.isfinite(network.output_scale).all() and (network.output_scale > 0).all()
            results.append({'arm': arm, 'names': names, 'CPU_GPU_double_max_abs': difference,
                            'checkpoint_exact': True, 'fit': receipt})
        write_json(out / 'qualification.json', {'status': 'full_real_scalar_ROCm_path_qualified',
            'training_rows': len(x), 'results': results, 'campaign_candidate': False,
            'comparative_metrics_processed': False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    STATE.update(base=base_maps(), metas={p: metadata(p) for p in PARTS},
                 protocol=json.loads((HERE / 'protocol.json').read_text()))
    original_controls = controls()
    if args.qualify:
        qualify(args.out)
        return
    assert len(jax.devices()) == 3 and all(d.platform in ('gpu', 'rocm') for d in jax.devices())
    if not (args.out / 'manifest.json').exists():
        write_json(args.out / 'manifest.json', {'status': 'all_30_scalar_tasks_declared', 'expected_tasks': 30,
            'controls': original_controls, 'protocol_sha256': digest(HERE / 'protocol.json')})
    with ThreadPoolExecutor(max_workers=3) as pool:
        def card_jobs(card):
            return [r for fold in range(card, 5, 3) for r in fold_worker(fold, card, args.out)]
        receipts = [r for batch in pool.map(card_jobs, range(3)) for r in batch]
    assert len(receipts) == 30
    write_json(args.out / 'receipt.json', {'status': 'all_30_neural_tasks_terminal', 'tasks': receipts,
        'successful': sum(r['status'] == 'fixed_final_epoch_fit_complete' for r in receipts),
        'comparative_metrics_computed': False})


if __name__ == '__main__':
    main()
