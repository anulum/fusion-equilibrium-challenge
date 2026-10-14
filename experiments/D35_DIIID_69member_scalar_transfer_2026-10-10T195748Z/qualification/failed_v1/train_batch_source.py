# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — 69-core scalar transfer training
"""Refit every production recipe and eight independent scalar heads."""
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

from campaign_common import HERE, TABLES, digest, write_json
from training_data import COMPILE, Moments, initialize, scaling_for, upload
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, Table, _Scaling, label_loss_weights
from fec_adapter.diii_d_accuracy.network import Network

STATE = {}


def save_checkpoint(path: Path, state, key, epoch: int, settings: FitSettings, best, criterion: float, best_epoch: int) -> None:
    """Atomically save current optimizer and the selected control checkpoint."""
    arrays = {'epoch': np.asarray(epoch), 'key': np.asarray(jax.random.key_data(key)),
              'step': np.asarray(state[3]), 'settings': np.asarray(json.dumps(settings.__dict__, sort_keys=True)),
              'criterion': np.asarray(criterion), 'best_epoch': np.asarray(best_epoch)}
    for name, parameters in zip(('parameters', 'first', 'second', 'best'), (*state[:3], best), strict=True):
        arrays[name + '_linear'] = np.asarray(parameters['linear'])
        for block in ('weights', 'biases'):
            for i, value in enumerate(parameters[block]):
                arrays[f'{name}_{block}_{i}'] = np.asarray(value)
    temporary = path.with_name('checkpoint.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def restore_checkpoint(path: Path, settings: FitSettings):
    """Restore identical settings and both current and best parameter states."""
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored['settings']) == json.dumps(settings.__dict__, sort_keys=True)
        states = []
        for name in ('parameters', 'first', 'second', 'best'):
            states.append({block: [jnp.asarray(stored[f'{name}_{block}_{i}']) for i in range(settings.depth + 1)]
                           for block in ('weights', 'biases')})
            states[-1]['linear'] = jnp.asarray(stored[name + '_linear'])
        return ((*states[:3], jnp.asarray(stored['step'])), jax.random.wrap_key_data(jnp.asarray(stored['key'])),
                int(stored['epoch']), states[3], float(stored['criterion']), int(stored['best_epoch']))


def output_view(task: dict, scaling: _Scaling, labels: tuple):
    """Select the original family's output order and scalar-only padding."""
    settings = FitSettings(**task['fit_settings'])
    indices = list(range(settings.coordinates)) + [64 + labels.index(n) for n in task['label_names']]
    mean, scale = scaling.feature_mean, scaling.feature_scale
    if settings.coordinates == 0:
        mean, scale = np.r_[mean, np.zeros(183)], np.r_[scale, np.ones(183)]
    return settings, indices, _Scaling(mean, scale, scaling.output_mean[indices], scaling.output_scale[indices])


def fit_one(task: dict, x, all_y, held, scaling: _Scaling, data: dict, device, out: Path) -> dict:
    """Keep production cal-flux selection; scalar models keep fixed final epoch."""
    destination = out / task['name']
    destination.mkdir(exist_ok=True)
    report = destination / 'fit_report.json'
    if report.exists():
        receipt = json.loads(report.read_text())
        assert digest(destination / 'network.npz') == receipt['network_sha256']
        return receipt
    settings, indices, scaling = output_view(task, scaling, data['metas']['training']['label_names'])
    names = tuple(task['label_names'])
    batches = len(x) // settings.batch
    steps = settings.epochs * batches
    weights = jnp.asarray(np.r_[np.ones(settings.coordinates), label_loss_weights(names, settings)], dtype=jnp.float32)
    matrix = jnp.eye(len(indices), dtype=jnp.float32)
    key = jax.random.key(settings.seed)
    key, initialization = jax.random.split(key)
    start_epoch = 0
    checkpoint = destination / 'checkpoint.npz'
    criterion, best_epoch = np.inf, 0
    if checkpoint.exists():
        state, key, start_epoch, best, criterion, best_epoch = restore_checkpoint(checkpoint, settings)
    else:
        parameters = fit.initial_parameters(initialization, len(scaling.feature_mean), len(indices), settings)
        state = (parameters, jax.tree.map(jnp.zeros_like, parameters), jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))
        best = parameters
    chosen_outputs = jnp.asarray(indices)

    @jax.jit
    def one_pass(carry, order_key, inputs, targets):
        def update(inner, chosen):
            features = inputs[chosen]
            if settings.coordinates == 0:
                features = jnp.pad(features, ((0, 0), (0, 183)))
            wanted = targets[chosen][:, chosen_outputs]
            loss, gradient = jax.value_and_grad(fit._loss)(inner[0], features, wanted, weights, 0.,
                            jax.random.fold_in(initialization, inner[3]), matrix)
            return fit.decayed_adam_step(inner, gradient, steps, settings), loss
        order = jax.random.permutation(order_key, inputs.shape[0])[:batches * settings.batch]
        carry, losses = jax.lax.scan(update, carry, order.reshape(batches, settings.batch))
        return carry, jnp.mean(losses)

    @jax.jit
    def held_coordinates(parameters, inputs):
        return fit.forward(parameters, inputs, 0., None)[:, :64]

    truth = data['base']['calibration']['coordinates'][:, :64].astype(float)
    started = time.monotonic()
    for epoch in range(start_epoch, settings.epochs):
        key, order = jax.random.split(key)
        if epoch == start_epoch:
            with COMPILE:
                state, loss = one_pass(state, order, x, all_y)
                loss.block_until_ready()
                predicted = np.asarray(held_coordinates(state[0], held), dtype=float) if settings.coordinates else None
        else:
            state, loss = one_pass(state, order, x, all_y)
            loss.block_until_ready()
            predicted = np.asarray(held_coordinates(state[0], held), dtype=float) if settings.coordinates else None
        assert np.isfinite(float(loss)), 'nonfinite training loss; no numeric retry'
        log = {'epoch': epoch+1, 'training_loss': float(loss), 'seconds_this_attempt': time.monotonic()-started}
        if settings.coordinates:
            predicted = predicted * scaling.output_scale[:64] + scaling.output_mean[:64]
            squared = np.sort(np.sum((predicted-truth)**2, axis=1))
            value = float(squared[:-max(1, len(squared)//fit.TRIMMED_SHARE)].sum())
            assert np.isfinite(value), 'nonfinite cal criterion'
            log['calibration_trimmed_coordinate_SSE'] = value
            if value < criterion:
                criterion, best_epoch = value, epoch+1
                best = jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float64).copy(), state[0])
        else:
            best, criterion, best_epoch = state[0], 0., epoch+1
        with (destination / 'passes.jsonl').open('a') as stream:
            stream.write(json.dumps(log, allow_nan=False)+'\n')
        if (epoch+1) % 10 == 0 or epoch+1 == settings.epochs:
            save_checkpoint(checkpoint, state, key, epoch+1, settings, best, criterion, best_epoch)
        print(json.dumps({'task': task['name'], 'epoch_completed': epoch+1, 'device': str(device)}), flush=True)
    assert best_epoch > 0
    network = fit._export(best, scaling, settings, SimpleNamespace(label_names=names, feature_layout='discharge'))
    assert all(np.isfinite(a).all() for a in network.weights + network.biases + (network.linear,))
    network.save(destination / 'network.npz')
    receipt = {'name': task['name'], 'family': task['family'], 'status': 'fit_complete', 'training_frames': len(x),
               'settings': settings.__dict__, 'train_on_calibration': task['train_on_calibration'],
               'epochs_completed': settings.epochs, 'selected_epoch': best_epoch, 'criterion': criterion,
               'resumed_from_epoch': start_epoch, 'network_sha256': digest(destination / 'network.npz'),
               'seconds_this_attempt': time.monotonic()-started, 'held_out_metrics_computed': False,
               'selection': 'eligible_calibration_flux' if settings.coordinates else 'fixed_final_epoch'}
    write_json(report, receipt)
    return receipt


def worker(task: tuple) -> list:
    card, tasks, out = task
    device = jax.devices()[card]
    receipts = []
    with jax.default_device(device):
        for tc in (False, True):
            group = [t for t in tasks if t['train_on_calibration'] == tc]
            if not group:
                continue
            scaling = STATE['scaling'][tc]
            x, y, held = upload(STATE['data'], scaling, tc, device)
            for task in group:
                failed = out / (task['name']+'_failure.json')
                if failed.exists():
                    receipts.append(json.loads(failed.read_text()))
                    continue
                try:
                    receipts.append(fit_one(task, x, y, held, scaling, STATE['data'], device, out))
                except Exception as error:
                    receipt = {'name': task['name'], 'status': 'failed_no_numeric_retry', 'type': type(error).__name__, 'error': str(error)}
                    write_json(failed, receipt)
                    receipts.append(receipt)
                gc.collect()
            del x, y, held
            gc.collect()
    return receipts


def qualify(out: Path) -> None:
    """Exercise full eligible TC data and all three output shapes on real ROCm."""
    data, scaling = STATE['data'], STATE['scaling'][True]
    device = jax.devices()[0]
    assert device.platform in ('gpu', 'rocm')
    results = []
    with jax.default_device(device):
        x, y, held = upload(data, scaling, True, device)
        for family in ('hx', 'hc', 'joint'):
            task = dict(next(t for t in STATE['tasks'] if t['family'] == family))
            task['name'] = 'qualification_'+family
            task['fit_settings'] = {**task['fit_settings'], 'epochs': 1}
            receipt = fit_one(task, x, y, held, scaling, data, device, out)
            network = Network.load(out / task['name'] / 'network.npz')
            raw = data['base']['training']['features'][data['rows']['training'][:128]].astype(float)
            if family == 'joint':
                raw = np.pad(raw, ((0,0),(0,183)))
            parameters = {k: [jnp.asarray(a, dtype=jnp.float64) for a in getattr(network,k)] for k in ('weights','biases')}
            parameters['linear'] = jnp.asarray(network.linear, dtype=jnp.float64)
            gpu = np.asarray(fit.forward(parameters, jnp.asarray(network.standardised(raw)), 0., None))
            gpu = gpu * network.output_scale + network.output_mean
            coordinates, labels = network.predict(raw)
            cpu = np.column_stack([coordinates]+[labels[n] for n in network.label_names])
            difference = float(np.max(np.abs(cpu-gpu)))
            assert np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), difference
            settings = FitSettings(**task['fit_settings'])
            restored, key, epoch, best, criterion, best_epoch = restore_checkpoint(out / task['name'] / 'checkpoint.npz', settings)
            assert epoch == best_epoch == 1
            for block in ('weights','biases'):
                for saved, exported, current in zip(best[block], getattr(network,block), restored[0][block], strict=True):
                    assert np.array_equal(np.asarray(saved), exported) and np.array_equal(np.asarray(saved), np.asarray(current))
            assert np.array_equal(np.asarray(best['linear']), network.linear)
            assert np.array_equal(np.asarray(best['linear']), np.asarray(restored[0]['linear']))
            source_criterion = None
            if family != 'joint':
                predicted = np.asarray(fit.forward(restored[0], held, 0., None),dtype=float)[:,:64]
                predicted = predicted*network.output_scale[:64]+network.output_mean[:64]
                error = predicted-data['base']['calibration']['coordinates'][:,:64].astype(float)
                with np.load(TABLES/'table_calibration.npz', allow_pickle=False) as stored:
                    cells = float(stored['shot_cells'].sum())
                    mean = float(stored['shot_map_sum'].sum())/cells
                    sums = {'flux_squares_about_the_mean':float(stored['shot_squares_total'].sum())-cells*mean**2,
                            'flux_squares_left_out_by_the_basis':float(stored['shot_squares_left_out'].sum())+float(np.sum(stored['coordinates'][:,64:].astype(float)**2))}
                selected, _ = fit._held_scores(error,sums)
                source_criterion = selected
                assert np.isclose(selected,1.-(criterion+sums['flux_squares_left_out_by_the_basis'])/sums['flux_squares_about_the_mean'],rtol=1e-12,atol=1e-12)
            results.append({'family':family,'CPU_GPU_double_max_abs':difference,'checkpoint_exact':True,
                            'selection_source_parity':source_criterion,'fit':receipt})
        sample = data['rows']['training'][:16384]
        base = data['base']['training']
        actual = Table(base['features'][sample],base['coordinates'][sample,:64],base['labels'][sample],
                       data['metas']['training']['label_names'],{},'discharge')
        reference = _Scaling.of(actual)
        fm, ym = Moments(582), Moments(142)
        fm.add(actual.features.astype(float)); ym.add(np.column_stack([actual.coordinates,actual.labels]).astype(float))
        scale=ym.scale();scale[:64]=np.sqrt(np.sum(ym.m2[:64]/ym.count[:64]))
        for computed, expected in zip((fm.mean,fm.scale(),ym.mean,scale),reference.__dict__.values(),strict=True):
            assert np.allclose(computed,expected,rtol=1e-10,atol=1e-10)
        write_json(out/'qualification.json',{'status':'full_production69_scalar_ROCm_path_qualified',
                    'training_rows':len(x),'results':results,'scaling_source_parity':True,
                    'holdout_metrics_computed':False,'campaign_candidate':False})


def main() -> None:
    """Run qualification or finish all 77 declared fits without scoring holdout."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--qualify',action='store_true')
    args=parser.parse_args()
    STATE.update(data=initialize(),tasks=json.loads((HERE/'tasks.json').read_text())['tasks'])
    STATE['scaling']={tc:scaling_for(STATE['data'],tc) for tc in (False,True)}
    if args.qualify:
        qualify(args.out)
        return
    assert len(jax.devices())==3
    assignments=[(card, STATE['tasks'][card::3],args.out) for card in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts=[r for batch in pool.map(worker,assignments) for r in batch]
    assert len(receipts)==77
    write_json(args.out/'receipt.json',{'status':'all_77_neural_tasks_terminal','tasks':receipts,
        'successful':sum(r['status']=='fit_complete' for r in receipts),'held_out_metrics_computed':False})


if __name__=='__main__':
    main()
