# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D57: fit whole-shot head networks and their matched per-frame control
"""Fit temporal_model on crops of whole shots; ``--arm temporal`` mixes the frames, ``--arm control`` does not.

Rows: the 6326 shots outside both evaluation cohorts (D38 eligible rows). Targets: r_axis, z_axis (table labels),
kappa, tri_top, tri_bot, volume, li_scored (true-map teacher targets), q95, beta_n (table labels). Both arms take
the same crops in the same order for the same seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parent
D38 = E / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
sys.path.insert(0, str(HERE))
for _path in ('$HOME/fusion-equilibrium-challenge/fec_adapter/vendor', '$HOME/fusion-equilibrium-challenge/fec_adapter'):
    sys.path.insert(0, _path)
LABELS = ('r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored', 'q95', 'beta_n')
TEACHER = ('kappa', 'tri_top', 'tri_bot', 'volume', 'li_scored')
WEIGHT = np.array([5.5, 5.5, 4.9, 0.55, 0.55, 2.1, 10.6, 2.4, 6.8])   # microS per percent of rms error (D45; qb from the R2 term)
CROP, SHOTS_PER_STEP, EPOCHS, RATE, DECAY, LIMIT = 256, 8, 60, 0.002, 1e-5, 1.0
SINCE_COLUMN = 64                                                    # seconds since the discharge began (phase block)


def digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gather():
    """Features, targets, seconds and shot bounds of the eligible rows, shot by shot in table order."""
    from campaign_common import base_maps
    from training_data import initialize
    data = initialize()
    cohort = json.loads((D38 / 'cohort.json').read_text())
    wanted = {'training': set(cohort['training']), 'calibration': set(cohort['calibration'])}
    names = json.loads((D38 / 'supervision/receipt.json').read_text())['teacher_labels']
    base = base_maps()
    for part in ('training', 'calibration'):
        meta, ids = data['metas'][part], data['rows'][part]
        teacher = np.load(D38 / 'supervision' / (part + '_geometry.npy'), mmap_mode='r', allow_pickle=False)
        assert len(teacher) == len(ids)
        table_names = list(meta['label_names'])
        at = 0
        for i, stem in enumerate(meta['stems']):
            if stem not in wanted[part]:
                continue
            lo, hi = meta['bounds'][i], meta['bounds'][i + 1]
            assert np.array_equal(ids[at:at + hi - lo], np.arange(lo, hi))
            target = np.empty((hi - lo, len(LABELS)))
            for j, name in enumerate(LABELS):
                target[:, j] = teacher[at:at + hi - lo, names.index(name)] if name in TEACHER else base[part]['labels'][lo:hi, table_names.index(name)]
            yield np.asarray(base[part]['features'][lo:hi], dtype=np.float64), target
            at += hi - lo
        assert at == len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('temporal', 'control'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--smoke-steps', type=int, default=0, help='mechanics only: this many steps, nothing exported')
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import temporal_model as model
    from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
    from fec_adapter.diii_d_accuracy.fit_network import FitSettings
    from training_data import Moments, put_block
    assert jax.devices()[0].platform == 'gpu'
    device = jax.devices()[0]
    args.out.mkdir(parents=True, exist_ok=True)
    mix = args.arm == 'temporal'
    xmom, ymom, lengths = Moments(582), Moments(len(LABELS)), []
    for raw, target in gather():
        assert np.isfinite(raw).all() and not np.isinf(target).any()
        xmom.add(raw); ymom.add(target); lengths.append(len(raw))
    lengths = np.asarray(lengths)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    n = int(lengths.sum())
    feature_mean, feature_scale = xmom.mean.copy(), xmom.scale()
    output_mean, output_scale = ymom.mean.copy(), ymom.scale()
    assert (ymom.count > 0).all() and (output_scale > 1e-6).all()
    np.savez(args.out / 'normalization.npz', feature_mean=feature_mean, feature_scale=feature_scale, output_mean=output_mean,
             output_scale=output_scale, labels=np.asarray(LABELS), since_column=SINCE_COLUMN, shots=len(lengths), frames=n)
    weight = jnp.asarray(WEIGHT / WEIGHT.mean(), jnp.float32)
    with jax.default_device(device):
        x, y, seconds, at = jnp.zeros((n, 582), jnp.float32), jnp.zeros((n, len(LABELS)), jnp.float32), np.zeros(n, np.float32), 0
        for raw, target in gather():
            x = put_block(x, jax.device_put(np.clip((raw - feature_mean) / feature_scale, -8., 8.).astype(np.float32), device), at)
            y = put_block(y, jax.device_put(((target - output_mean) / output_scale).astype(np.float32), device), at)
            seconds[at:at + len(raw)] = raw[:, SINCE_COLUMN]
            x.block_until_ready(); y.block_until_ready()
            at += len(raw)
        assert at == n
        seconds = jax.device_put(seconds, device)
        starts_d, lengths_d = jax.device_put(starts.astype(np.int32), device), jax.device_put(lengths.astype(np.int32), device)
        log_share = jnp.log(lengths_d.astype(jnp.float32))
        steps_per_epoch = n // (CROP * SHOTS_PER_STEP)
        total = (args.smoke_steps or args.epochs * steps_per_epoch)
        recipe = FitSettings(seed=0, coordinates=0, width=model.WIDTH, depth=model.DEPTH, epochs=args.epochs, batch=CROP * SHOTS_PER_STEP,
                             learning_rate=RATE, weight_decay=DECAY, label_weight=1.0, dropout=0.0, gradient_limit=LIMIT)

        def crop(key):
            pick_key, offset_key = jax.random.split(key)
            chosen = jax.random.categorical(pick_key, log_share, shape=(SHOTS_PER_STEP,))
            length, start = lengths_d[chosen], starts_d[chosen]
            room = jnp.maximum(length - CROP, 0)
            offset = jnp.floor(jax.random.uniform(offset_key, (SHOTS_PER_STEP,)) * (room + 1)).astype(jnp.int32)
            position = jnp.arange(CROP, dtype=jnp.int32)[None, :]
            valid = position < jnp.minimum(length, CROP)[:, None]
            index = jnp.clip(start[:, None] + offset[:, None] + position, 0, n - 1)
            return index, valid

        def loss(parameters, key, x, y, seconds):
            index, valid = crop(key)
            predicted = model.forward(parameters, x[index], seconds[index], valid, mix)
            wanted = y[index]
            use = valid[:, :, None] & jnp.isfinite(wanted)
            error = jnp.where(use, predicted - jnp.nan_to_num(wanted), 0.)
            return jnp.sum(weight * jnp.sum(error ** 2, axis=(0, 1)) / jnp.maximum(jnp.sum(use, axis=(0, 1)), 1))

        for seed in args.seeds:
            folder = args.out / f'{args.arm}_{seed}'
            report = folder / 'fit_report.json'
            if report.exists():
                continue
            key = jax.random.key(seed)
            key, init_key = jax.random.split(key)
            parameters = model.initial(init_key, 582, len(LABELS))
            state = (parameters, jax.tree.map(jnp.zeros_like, parameters), jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))

            @jax.jit
            def many(carry, keys, x, y, seconds):
                def update(inner, step_key):
                    value, gradient = jax.value_and_grad(loss)(inner[0], step_key, x, y, seconds)
                    return fit.decayed_adam_step(inner, gradient, total, recipe), value
                carry, losses = jax.lax.scan(update, carry, keys)
                return carry, jnp.mean(losses)

            started, history = time.monotonic(), []
            if args.smoke_steps:
                key, sub = jax.random.split(key)
                state, value = many(state, jax.random.split(sub, args.smoke_steps), x, y, seconds)
                print(json.dumps({'smoke': args.arm, 'seed': seed, 'steps': args.smoke_steps, 'mean_loss': float(value),
                                  'seconds': time.monotonic() - started, 'steps_per_epoch': int(steps_per_epoch), 'frames': n,
                                  'shots': int(len(lengths)), 'median_length': float(np.median(lengths)),
                                  'since_step_median': float(np.median(np.diff(np.asarray(seconds[:int(lengths[0])]))))}), flush=True)
                continue
            for epoch in range(args.epochs):
                key, sub = jax.random.split(key)
                state, value = many(state, jax.random.split(sub, steps_per_epoch), x, y, seconds)
                value = float(value)
                assert np.isfinite(value), 'non-finite training loss; no retry'
                history.append(value)
            values = np.asarray(history)
            final = float(values[-10:].mean())
            best = float(np.lib.stride_tricks.sliding_window_view(values, 10).mean(axis=1).min())
            flat = {'/'.join(map(str, path)): np.asarray(leaf) for path, leaf in
                    ((tuple(getattr(k, 'key', getattr(k, 'idx', k)) for k in p), l) for p, l in jax.tree_util.tree_leaves_with_path(state[0]))}
            finite = all(np.isfinite(a).all() for a in flat.values())
            folder.mkdir(exist_ok=True)
            np.savez(folder / 'parameters.npz', **flat)
            receipt = {'arm': args.arm, 'seed': seed, 'training_frames': n, 'shots': int(len(lengths)), 'epochs': args.epochs,
                       'steps': int(args.epochs * steps_per_epoch), 'loss_history': history, 'final10_mean_training_loss': final,
                       'lowest10_mean_training_loss': best, 'seed_quality_pass': bool(finite and final <= 1.25 * best),
                       'parameters_sha256': digest(folder / 'parameters.npz'), 'seconds': time.monotonic() - started,
                       'labels': list(LABELS), 'this_script_sha256': digest(__file__), 'model_sha256': digest(HERE / 'temporal_model.py')}
            with report.open('x') as stream:
                json.dump(receipt, stream, indent=1)
            print(json.dumps({'fit': args.arm, 'seed': seed, 'final10': final, 'first': history[0], 'pass': receipt['seed_quality_pass'],
                              'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    main()
