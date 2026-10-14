# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed final-parameter training-gradient attribution
"""Measure actual float32 loss gradients without updating parameters."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import digest
from training_data import initialize, scaling_for
from candidate_tables import row_weights
from candidate_state import restore
from train_maps import geometry, settings
from map_objectives import interpolate, effective_decoder, objective
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.network import Network
import jax
import jax.numpy as jnp


def verify():
    for source, expected in json.loads((HERE / 'training_gradient_audit_sources_v2.json').read_text()).items():
        assert digest(source) == expected, source


def terms(p, x, y, teacher, points, rows, basis, mean, scale, mask, family):
    output = fit.forward(p, x, 0., None)
    error = jnp.where(jnp.isfinite(y), output[:, :142]-jnp.nan_to_num(y), 0.)
    coord = jnp.mean(jnp.sum(error[:, :64]**2, axis=1))
    labels = .05*jnp.mean(jnp.sum(error[:, 64:]**2, axis=1))
    zero = jnp.asarray(0., dtype=jnp.float32)
    if family == 'control':
        return coord, labels, zero, zero
    sampled = interpolate(basis, points)
    diff = sampled[:, 1:]-sampled[:, :1]
    truth = jnp.einsum('nc,npc->np', teacher/scale, diff)
    coefficients = output[:, :45]+mean[:45]/scale
    predicted = jnp.einsum('nc,npc->np', coefficients, diff[:, :, :45])
    decoder = None
    if family == 'spatial':
        decoder = effective_decoder(p, mask)
        sampled_decoder = interpolate(decoder, points)
        predicted += jnp.einsum('nc,npc->np', output[:, 142:], sampled_decoder[:, 1:]-sampled_decoder[:, :1])
    valid = jnp.isfinite(points[:, 1:]).all(axis=2) & jnp.isfinite(points[:, :1]).all(axis=2)
    ray_error = jnp.where(valid, predicted-truth, 0.)
    ray = .2*4225*jnp.mean(rows*jnp.sum(ray_error**2, axis=1)/jnp.maximum(valid.sum(axis=1), 1))
    field = zero
    if decoder is not None:
        field = .2*jnp.mean(jnp.sum((coefficients@basis[:45]+output[:, 142:]@decoder-(teacher/scale)@basis)**2, axis=1))
    return coord, labels, ray, field


def inner(a, b):
    return sum(jnp.sum(x*y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def main():
    started = time.time()
    try:
        verify()
        protocol = json.loads((HERE / 'training_gradient_audit_protocol.json').read_text())
        device = jax.devices()[0]
        assert device.platform in ('rocm', 'gpu'), device
        data = initialize()
        rows = data['rows']['training']
        assert len(rows) == 1321417
        scaling = scaling_for(data, True)
        frame_weights, shot_weights = row_weights(data)
        selected = np.asarray(protocol['eligible_training_row_positions'], dtype=np.int64).reshape(20, 4096)
        assert selected.min() >= 0 and selected.max() < len(rows) and np.unique(selected).size == selected.size
        base = data['base']['training']
        points = np.load(D38 / 'supervision/training_points.npy', mmap_mode='r', allow_pickle=False)
        reports = {}
        with jax.default_device(device):
            basis, mask = geometry(device)
            mean = jnp.asarray(scaling.output_mean[:64], dtype=jnp.float32)
            scale = jnp.asarray(scaling.output_scale[0], dtype=jnp.float32)
            cached = []
            for number, chosen in enumerate(selected):
                absolute = rows[chosen]
                raw = base['features'][absolute].astype(float)
                x = np.clip((raw-scaling.feature_mean)/scaling.feature_scale, -8., 8.).astype(np.float32)
                targets = np.column_stack([base['coordinates'][absolute, :64], base['labels'][absolute]]).astype(float)
                y = ((targets-scaling.output_mean)/scaling.output_scale).astype(np.float32)
                teacher = np.asarray(base['coordinates'][absolute, :128], dtype=np.float32)
                common_args = tuple(jax.device_put(v, device) for v in (x, y, teacher, np.asarray(points[chosen], dtype=np.float32)))
                cached.append((common_args, jax.device_put(frame_weights[chosen], device), jax.device_put(shot_weights[chosen], device)))
            print(json.dumps({'all20_original_batches_cached_on_same_ROCm_card': True}), flush=True)
            for family in protocol['families']:
                def measure(p, x, y, teacher, point, weight):
                    args = (x, y, teacher, point, weight, basis, mean, scale, mask, family)
                    value, total = jax.value_and_grad(objective)(p, *args)
                    evaluated = [jax.value_and_grad(lambda params: terms(params, *args)[i])(p) for i in range(4)]
                    losses = jnp.stack([a for a, _ in evaluated])
                    gradients = [g for _, g in evaluated]
                    summed = jax.tree.map(lambda *v: sum(v), *gradients)
                    delta = jax.tree.map(lambda a, b: a-b, summed, total)
                    norm = jnp.sqrt(inner(total, total))
                    norms = jnp.stack([jnp.sqrt(inner(g, g)) for g in gradients])
                    projections = jnp.stack([inner(g, total) for g in gradients])
                    shrink = jnp.minimum(1., 1./(norm+fit.EPSILON))
                    return value, losses, norm, norms, projections, shrink, jnp.sqrt(inner(delta, delta))
                compiled = jax.jit(measure)
                for seed in protocol['seeds']:
                    folder = D38 / 'map_fits' / f'{family}_{seed}'
                    network = Network.load(folder / 'network.npz')
                    params = {name: [jnp.asarray(v, dtype=jnp.float32) for v in getattr(network, name)] for name in ('biases', 'weights')}
                    params['linear'] = jnp.asarray(network.linear, dtype=jnp.float32)
                    if family == 'spatial':
                        params['decoder'] = jnp.zeros((16, 4225), dtype=jnp.float32)
                    template = (params, jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params), jnp.zeros((), dtype=jnp.int32))
                    recipe = {'family': family, 'settings': settings(seed).__dict__, 'protocol_sha256': digest(D38 / 'candidate_protocol_frozen.json')}
                    state, _, epoch = restore(folder / 'checkpoint.npz', template, recipe)
                    assert epoch == 60
                    p = state[0]
                    for name in ('biases', 'weights'):
                        assert all(np.array_equal(np.asarray(v, dtype=float), w) for v, w in zip(p[name], getattr(network, name), strict=True))
                    assert np.array_equal(np.asarray(p['linear'], dtype=float), network.linear)
                    if family == 'spatial':
                        assert np.array_equal(np.asarray(effective_decoder(p, mask), dtype=float), np.load(folder / 'decoder.npy', allow_pickle=False))
                    batch_reports = []
                    for number, chosen in enumerate(selected):
                        common_args, frame_weight, shot_weight = cached[number]
                        weighting = (shot_weights if family in ('boundary_shot', 'spatial') else frame_weights)[chosen]
                        args = (*common_args, shot_weight if family in ('boundary_shot', 'spatial') else frame_weight)
                        values = compiled(p, *args)
                        value, losses, norm, norms, projections, shrink, residual = [np.asarray(v) for v in values]
                        assert all(np.isfinite(v).all() for v in (value, losses, norm, norms, projections, shrink, residual))
                        assert np.isclose(float(value), float(losses.sum()), atol=protocol['loss_atol'], rtol=protocol['loss_rtol'])
                        assert float(residual) <= protocol['gradient_relative_l2_tolerance']*max(float(norm), 1e-8)
                        batch_reports.append({'batch': number, 'actual_original_loss': float(value), 'term_losses': losses.tolist(),
                                              'actual_original_gradient_L2': float(norm), 'term_gradient_L2': norms.tolist(),
                                              'term_dot_actual_gradient': projections.tolist(), 'native_clip_shrink': float(shrink),
                                              'decomposition_gradient_residual_L2': float(residual),
                                              'row_weight_min': float(weighting.min()), 'row_weight_max': float(weighting.max())})
                    reports[f'{family}_{seed}'] = {'checkpoint_sha256': digest(folder / 'checkpoint.npz'), 'final_epoch': epoch,
                          'batches': batch_reports, 'clip_count': sum(b['native_clip_shrink'] < 1 for b in batch_reports)}
                    print(json.dumps({'gradient_audit_completed': family, 'seed': seed}), flush=True)
                    del state, p, params, template
                    gc.collect()
                del compiled
        verify()
        with (HERE / 'training_gradient_audit_analysis.json').open('x') as stream:
            json.dump({'status': 'all32_final60_fixed20_training_batches_gradient_attribution_complete', 'reports': reports,
                       'term_order': ['coordinates', 'labels', 'ray', 'field'], 'float32_actual_checkpoint_parameters': True, 'original_batch_loading_expressions_cached_once': True,
                       'same_batches_for_all_models': True, 'no_optimizer_steps_or_model_updates': True,
                       'final_parameter_snapshot_not_historical_clip_frequency': True,
                       'raw_gradient_attribution_not_Adam_update_attribution': True, 'no358_or357_labels_read': True,
                       'protocol_sha256': digest(HERE / 'training_gradient_audit_protocol.json'),
                       'not_candidate_admission': True, 'publication': False, 'seconds': time.time()-started}, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (HERE / 'training_gradient_audit_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
