# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D48: X-position specialists with more epochs or width (generalised D47 trainer)
"""Fit small networks that predict only the seven geometry targets the finisher edits the map towards.

Same construction as the X-position specialists of D40/D43 (static arm): 582 input features, 512 x 4, 60 fixed
epochs, masked squared error on normalised targets, no map coordinates. Rows: ``--rows eligible`` = the 6326 shots
outside both evaluation cohorts (replica for evaluation); ``--rows all`` = every labelled shot (deployment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parent
D38 = E / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
P43 = E / 'D43_production_static_X_preparation_2026-10-12T152300Z'
sys.path.insert(0, str(D38))
for _path in ('$HOME/fusion-equilibrium-challenge/fec_adapter/vendor', '$HOME/fusion-equilibrium-challenge/fec_adapter'):
    sys.path.insert(0, _path)   # as D38 campaign_common does; needed before the first fec_adapter import
LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')
TABLE_LABELS = ()   # all four X targets are true-map teacher targets, as in D40
RECIPE = {'coordinates': 0, 'width': 512, 'depth': 4, 'epochs': 60, 'batch': 2048, 'learning_rate': 0.002,
          'weight_decay': 1e-05, 'label_weight': 1.0, 'dropout': 0.0, 'gradient_limit': 1.0}


def digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gather(rows_kind: str):
    """Return features and the seven targets for the chosen rows, in native table order."""
    from campaign_common import base_maps, metadata, mapped_array
    from training_data import initialize
    base = base_maps()
    if rows_kind == 'eligible':
        data = initialize()
        names = json.loads((D38 / 'supervision/receipt.json').read_text())['teacher_labels']
        parts = []
        for part in ('training', 'calibration'):
            ids = data['rows'][part]
            teacher = np.load(D38 / 'supervision' / (part + '_geometry.npy'), mmap_mode='r', allow_pickle=False)
            assert len(teacher) == len(ids)
            table_names = list(data['metas'][part]['label_names'])
            parts.append((base[part]['features'], ids, base[part]['labels'], table_names, teacher, names))
        return parts
    assert rows_kind == 'all'
    # deployment: every labelled row; teacher targets of all 7041 shots come from the per-shot teacher files that
    # the D43 production preparation verified (6326 cached + 715 derived), read through its row plan
    plan = json.loads((P43 / 'metadata_row_plan_v1.json').read_text())
    receipt = json.loads((P43 / 'all7041_data/receipt.json').read_text())
    parts = []
    for part in ('training', 'calibration'):
        meta = metadata(part)
        count = plan['totals'][part]['frames']
        teacher = np.full((count, 4), np.nan, dtype=np.float32)
        wanted = LABELS
        for row in plan['parts'][part]:
            old = row['cached_teacher']
            path = Path(old['path']) if old is not None else P43 / 'all7041_data/missing_teachers' / part / (row['stem'] + '.npz')
            assert str(path) in receipt['teacher_sources_sha256']
            with np.load(path, allow_pickle=False) as saved:
                names = saved['label_names'].astype(str).tolist()
                lo, hi = row['feature_span']
                teacher[lo:hi] = saved['labels'][:, [names.index(n) for n in wanted]]
        parts.append((base[part]['features'], np.arange(count), base[part]['labels'], list(meta['label_names']), teacher, list(wanted)))
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', choices=('eligible', 'all'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--name', default='x')
    args = parser.parse_args()
    RECIPE.update(epochs=args.epochs, width=args.width)
    import jax
    import jax.numpy as jnp
    from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
    from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling
    assert jax.devices()[0].platform == 'gpu'
    device = jax.devices()[0]
    args.out.mkdir(parents=True, exist_ok=True)
    parts = gather(args.rows)
    from training_data import Moments, put_block
    # two streamed passes keep the host copy bounded: moments first, then normalised blocks straight to the card

    def blocks():
        for features, ids, table, table_names, teacher, teacher_names in parts:
            for start in range(0, len(ids), 8192):
                take = ids[start:start + 8192]
                target = np.empty((len(take), 4), dtype=np.float64)
                for j, name in enumerate(LABELS):
                    if name in TABLE_LABELS:
                        target[:, j] = table[take, table_names.index(name)]
                    else:
                        target[:, j] = teacher[start:start + len(take), teacher_names.index(name)]
                yield np.asarray(features[take], dtype=np.float64), target

    xmom, ymom, n = Moments(582), Moments(4), 0
    for raw, target in blocks():
        assert np.isfinite(raw).all() and not np.isinf(target).any()
        xmom.add(raw); ymom.add(target); n += len(raw)
    feature_mean, feature_scale = xmom.mean.copy(), xmom.scale()
    output_mean = ymom.mean.copy()
    output_scale = np.full(4, float(np.sqrt(np.mean(ymom.m2 / ymom.count))))   # one metre scale for the four positions, as D40
    assert (ymom.count > 0).all() and (output_scale > 1e-6).all()
    scaling = _Scaling(feature_mean, feature_scale, output_mean, output_scale)
    np.savez(args.out / 'normalization.npz', feature_mean=feature_mean, feature_scale=feature_scale,
             output_mean=output_mean, output_scale=output_scale, labels=np.asarray(LABELS),
             finite_target_counts=ymom.count.astype(np.int64))
    with jax.default_device(device):
        x, y, at = jnp.zeros((n, 582), jnp.float32), jnp.zeros((n, 4), jnp.float32), 0
        for raw, target in blocks():
            x = put_block(x, jax.device_put(np.clip((raw - feature_mean) / feature_scale, -8., 8.).astype(np.float32), device), at)
            y = put_block(y, jax.device_put(((target - output_mean) / output_scale).astype(np.float32), device), at)
            x.block_until_ready(); y.block_until_ready()
            at += len(raw)
        assert at == n
        batches = n // RECIPE['batch']

        def loss(parameters, inputs, wanted):
            predicted = fit.forward(parameters, inputs, 0., None)
            valid = jnp.isfinite(wanted)
            error = jnp.where(valid, predicted - jnp.nan_to_num(wanted), 0.)
            return jnp.mean(jnp.sum(error ** 2, axis=1))

        receipts = []
        for seed in args.seeds:
            folder = args.out / f'{args.name}_{seed}'
            report = folder / 'fit_report.json'
            if report.exists():
                receipts.append(json.loads(report.read_text()))
                continue
            folder.mkdir(exist_ok=True)
            recipe = FitSettings(seed=seed, **RECIPE)
            key = jax.random.key(seed)
            key, init_key = jax.random.split(key)
            parameters = fit.initial_parameters(init_key, 582, 4, recipe)
            state = (parameters, jax.tree.map(jnp.zeros_like, parameters), jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))

            @jax.jit
            def one_epoch(carry, order_key, x, y):   # the data are arguments: captured, they are copied into the program as constants
                def update(inner, chosen):
                    value, gradient = jax.value_and_grad(loss)(inner[0], x[chosen], y[chosen])
                    return fit.decayed_adam_step(inner, gradient, recipe.epochs * batches, recipe), value
                order = jax.random.permutation(order_key, n)[:batches * recipe.batch]
                carry, losses = jax.lax.scan(update, carry, order.reshape(batches, recipe.batch))
                return carry, jnp.mean(losses)

            started, history = time.monotonic(), []
            for epoch in range(recipe.epochs):
                key, order = jax.random.split(key)
                state, value = one_epoch(state, order, x, y)
                value = float(value)
                assert np.isfinite(value), 'non-finite training loss; no retry'
                history.append(value)
            values = np.asarray(history)
            final = float(values[-10:].mean())
            best = float(np.lib.stride_tricks.sliding_window_view(values, 10).mean(axis=1).min())
            finite = all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state[0]))
            network = fit._export(state[0], scaling, recipe, SimpleNamespace(label_names=LABELS, feature_layout='discharge'))
            network.save(folder / 'network.npz')
            receipt = {'seed': seed, 'rows': args.rows, 'training_frames': n, 'epochs': recipe.epochs, 'loss_history': history,
                       'final10_mean_training_loss': final, 'lowest10_mean_training_loss': best,
                       'seed_quality_pass': bool(finite and final <= 1.25 * best), 'network_sha256': digest(folder / 'network.npz'),
                       'seconds': time.monotonic() - started, 'recipe': RECIPE, 'labels': list(LABELS),
                       'this_script_sha256': digest(__file__)}
            with report.open('x') as stream:
                json.dump(receipt, stream, indent=1)
            receipts.append(receipt)
            print(json.dumps({'x_fit': seed, 'final10': final, 'pass': receipt['seed_quality_pass'], 'seconds': receipt['seconds']}), flush=True)
    with (args.out / f'receipt_{time.time_ns()}.json').open('x') as stream:
        json.dump({'rows': args.rows, 'seeds': args.seeds, 'all_pass': all(r['seed_quality_pass'] for r in receipts),
                   'training_frames': n, 'normalization_sha256': digest(args.out / 'normalization.npz')}, stream, indent=1)


if __name__ == '__main__':
    main()
