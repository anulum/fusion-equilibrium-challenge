# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — real ROCm X specialist arithmetic qualification
"""Check the GPU objective and portable inference without optimizer updates."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import x_temporal_model as model
from fec_adapter.diii_d_accuracy.fit_network import FitSettings
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    for path, expected in json.loads((HERE / 'ROCm_qualification_v2_sources.json').read_text()).items():
        assert digest(path) == expected, path


def main():
    started = time.monotonic()
    try:
        verify()
        devices = jax.devices()
        assert len(devices) == 1 and devices[0].platform == 'gpu', devices
        protocol = json.loads((HERE / 'ROCm_qualification_v2_protocol.json').read_text())
        with np.load(protocol['static_input'], allow_pickle=False) as saved:
            static = saved['features'].astype(float)
        raw = np.load(protocol['temporal_input'], allow_pickle=False).astype(float)
        assert static.shape == (213, 582) and raw.shape == (213, 840)
        exported = model.XTemporalNetwork(HERE / 'prototype_qualification_models/static')
        x = np.clip((static-exported.network.feature_mean[:582])/exported.network.feature_scale[:582], -8., 8.)
        settings = FitSettings(coordinates=0, seed=500, epochs=60, batch=2048, label_weight=1.)
        parameters = model.initial(jax.random.key(500), settings)
        mean, scale = exported.mean, exported.scale
        targets = np.random.default_rng(4001).normal(size=(213, 4)).astype(np.float32)
        targets[::5, 2:] = np.nan
        targets[::7, :2] = np.nan
        X, W, Y, MU, SIG = [jnp.asarray(a, dtype=jnp.float32) for a in (x, raw, targets, mean, scale)]
        assert np.array_equal(np.asarray(model.window(W)), model.unpack(raw.astype(np.float32)))
        reports = {}
        for mode in model.MODES:
            objective = jax.jit(jax.value_and_grad(lambda p: model.loss(p, X, W, Y, MU, SIG, mode)))
            value, gradient = objective(parameters)
            value.block_until_ready()

            def reference(p):
                inputs = jnp.concatenate([X, model.pool(p, W, MU, SIG, mode)], axis=1)
                return model.fit._loss(p, inputs, Y, jnp.ones(4, jnp.float32), 0., None, jnp.eye(4, dtype=jnp.float32))

            wanted, reference_gradient = jax.jit(jax.value_and_grad(reference))(parameters)
            wanted.block_until_ready()
            assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(gradient))
            assert np.allclose(value, wanted, atol=protocol['float32_atol'], rtol=protocol['float32_rtol'])
            leaves = list(zip(jax.tree.leaves(gradient), jax.tree.leaves(reference_gradient), strict=True))
            assert all(np.allclose(a, b, atol=protocol['float32_atol'], rtol=protocol['float32_rtol']) for a, b in leaves)
            gradient_difference = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a, b in leaves)
            portable = model.XTemporalNetwork(HERE / 'prototype_qualification_models' / mode)
            stored = {block: [jnp.asarray(a, dtype=jnp.float64) for a in getattr(portable.network, block)]
                      for block in ('weights', 'biases')}
            stored['linear'] = jnp.asarray(portable.network.linear, dtype=jnp.float64)
            stored['conv_weights'] = [jnp.asarray(a, dtype=jnp.float64) for a in portable.weights]
            stored['conv_biases'] = [jnp.asarray(a, dtype=jnp.float64) for a in portable.biases]
            stored['readout'] = jnp.asarray(portable.readout, dtype=jnp.float64)
            stored['readout_bias'] = jnp.asarray(portable.readout_bias, dtype=jnp.float64)
            initialization_difference = max(float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b))))
                for a, b in zip(jax.tree.leaves(parameters), jax.tree.leaves(stored), strict=True))
            double = stored
            XD, WD, MD, SD = [jnp.asarray(a, dtype=jnp.float64) for a in (x, raw, mean, scale)]
            expected = np.asarray(jax.jit(lambda p: model.forward(p, XD, WD, MD, SD, mode))(double))
            portable = model.XTemporalNetwork(HERE / 'prototype_qualification_models' / mode)
            labels = portable.predict(static, raw)
            actual = np.column_stack([labels[n] for n in model.LABELS])
            difference = float(np.max(np.abs(actual-expected)))
            assert np.allclose(actual, expected, atol=protocol['double_atol'], rtol=protocol['double_rtol']), (mode, difference)
            reports[mode] = {'float32_masked_loss': float(value), 'masked_gradient_reference_max_abs': gradient_difference,
                             'portable_numpy_GPU_double_max_abs': difference, 'CPU_vs_GPU_separate_initialization_max_abs': initialization_difference, 'gradients_finite': True}
            print(json.dumps({'ROCm_prototype_qualified': mode, **reports[mode]}), flush=True)
        verify()
        with (HERE / 'ROCm_qualification_v2.json').open('x') as stream:
            json.dump({'status': 'all3_real_ROCm_objectives_and_portable_double_qualified',
                       'device': str(devices[0]), 'device_kind': devices[0].device_kind, 'reports': reports,
                       'no_optimizer_updates_no_training_no_true_labels_or_map_scores': True,
                       'protocol_sha256': digest(HERE / 'ROCm_qualification_v2_protocol.json'),
                       'not_trainer_checkpoint_resume_qualification': True,
                       'not_candidate_admission': True, 'seconds': time.monotonic()-started}, stream, indent=2)
    except Exception as error:
        with (HERE / 'ROCm_qualification_v2_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
