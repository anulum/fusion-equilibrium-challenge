# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — X optimizer interruption and export qualification
"""Compare continuous and resumed GPU optimizer state using synthetic labels."""
import json
from pathlib import Path
import time

import numpy as np
import x_training_kernel as kernel
from fec_adapter.diii_d_accuracy.fit_network import _Scaling
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
OUT = HERE / 'optimizer_resume_qualification'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify():
    for path, expected in json.loads((OUT / 'sources.json').read_text()).items():
        assert kernel.initial.digest(path) == expected, path


def inputs(protocol):
    with np.load(protocol['static_input'], allow_pickle=False) as saved:
        static = saved['features'].astype(float)
    raw = np.load(protocol['temporal_input'], allow_pickle=False).astype(float)
    assert static.shape == (213, 582) and raw.shape == (213, 840)
    reference = kernel.model.XTemporalNetwork(HERE / 'prototype_qualification_models/static')
    x = np.clip((static-reference.network.feature_mean[:582])/reference.network.feature_scale[:582], -8., 8.)
    index = np.arange(4096)%213
    targets = np.random.default_rng(4005).normal(size=(4096, 4)).astype(np.float32)
    targets[::5, 2:] = np.nan
    targets[::7, :2] = np.nan
    data = (jnp.asarray(x[index], dtype=jnp.float32), jnp.asarray(targets), jnp.asarray(raw[index], dtype=jnp.float16))
    normalization = jnp.asarray(reference.mean, jnp.float32), jnp.asarray(reference.scale, jnp.float32)
    return static, raw, x, reference, data, normalization


def main():
    started = time.monotonic()
    try:
        verify()
        device = jax.devices()[0]
        assert device.platform == 'gpu'
        protocol = json.loads((OUT / 'protocol.json').read_text())
        static, raw, x, reference, data, normalization = inputs(protocol)
        model = kernel.model
        recipe = kernel.settings(500)
        reports = {}
        for mode in model.MODES:
            folder = OUT / mode
            folder.mkdir(exist_ok=False)
            template, key = kernel.initial_state(500, device)
            step = kernel.epoch_step(data, normalization, mode, recipe)
            continuous, continuous_key, continuous_history = kernel.advance(step, template, key, [], 3)
            prefix, prefix_key, prefix_history = kernel.advance(step, template, key, [], 1)
            frozen = {'scope': 'synthetic_optimizer_qualification_only', 'mode': mode,
                      'recipe_sha256': kernel.initial.digest(HERE / 'training_recipe_frozen_v1.json'),
                      'protocol_sha256': kernel.initial.digest(OUT / 'protocol.json')}
            path = kernel.checkpoint(folder, prefix, prefix_key, prefix_history, frozen)
            kernel.checkpoint(folder, continuous, continuous_key, continuous_history, frozen)
            restored, restored_key, restored_history = kernel.resume(path, template, frozen)
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(prefix), jax.tree.leaves(restored), strict=True))
            resumed, resumed_key, resumed_history = kernel.advance(step, restored, restored_key, restored_history, 2)
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True))
            assert np.array_equal(jax.random.key_data(continuous_key), jax.random.key_data(resumed_key))
            assert continuous_history == resumed_history and int(resumed[-1]) == 6
            second_step = kernel.epoch_step(data, normalization, mode, recipe)
            second_state, second_key, second_history = kernel.advance(second_step, restored, restored_key, restored_history, 2)
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(continuous), jax.tree.leaves(second_state), strict=True))
            assert continuous_history == second_history and np.array_equal(jax.random.key_data(continuous_key), jax.random.key_data(second_key))
            scaling = _Scaling(reference.network.feature_mean, reference.network.feature_scale, np.zeros(4), np.ones(4))
            model.save(folder / 'export', resumed[0], scaling, recipe, reference.mean, reference.scale, mode)
            portable = model.XTemporalNetwork(folder / 'export')
            values = portable.predict(static, raw)
            cpu = np.column_stack([values[n] for n in model.LABELS])
            double = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), resumed[0])
            gpu = np.asarray(model.forward(double, jnp.asarray(x), jnp.asarray(raw),
                                          jnp.asarray(reference.mean), jnp.asarray(reference.scale), mode))
            difference = float(np.max(np.abs(cpu-gpu)))
            assert np.allclose(cpu, gpu, atol=1e-10, rtol=1e-10), (mode, difference)
            reports[mode] = {'complete_optimizer_RNG_history_resume_bitexact': True,
                             'fresh_epoch_kernel_resume_bitexact': True, 'optimizer_updates_per_path': 6,
                             'GPU_updated_weights_portable_double_max_abs': difference,
                             'checkpoint_sha256': kernel.initial.digest(path)}
            print(json.dumps({'X_optimizer_resume_qualified': mode, **reports[mode]}), flush=True)
        verify()
        write(OUT / 'receipt.json', {'status': 'all3_ROCm_synthetic_optimizer_resume_and_updated_export_qualified',
              'reports': reports, 'true_labels_or_map_scores_read': False,
              'synthetic_parameter_updates_performed_not_D40_candidate_training': True,
              'all24_real_candidate_fits_still_not_started': True,
              'does_not_qualify_new_process_or_another_GPU_resume': True,
              'no_gate_tolerance_waiver': True, 'seconds': time.monotonic()-started})
    except Exception as error:
        write(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
