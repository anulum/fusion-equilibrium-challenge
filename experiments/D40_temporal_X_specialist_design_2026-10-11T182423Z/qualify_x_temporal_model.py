# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — real-input temporal X prototype qualification
"""Verify the portable model and masked objective without fitting or scoring maps."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import x_temporal_model as model
from fec_adapter.diii_d_accuracy.fit_network import FitSettings, _Scaling
from temporal_model import pool as old_pool
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    for path, expected in json.loads((HERE / 'prototype_qualification_sources.json').read_text()).items():
        assert digest(Path(path)) == expected, path


def main():
    started = time.time()
    try:
        verify()
        protocol = json.loads((HERE / 'prototype_qualification_protocol.json').read_text())
        assert jax.devices()[0].platform == 'cpu'
        with np.load(Path(protocol['static_input']), allow_pickle=False) as saved:
            static = saved['features'].astype(float)
        raw = np.load(protocol['temporal_input'], allow_pickle=False).astype(float)
        assert static.shape == (213, 582) and raw.shape == (213, 840)
        assert np.isfinite(static).all() and np.isfinite(raw).all()
        with np.load(model.D38 / 'scaling_tc.npz', allow_pickle=False) as saved:
            scaling = _Scaling(np.r_[saved['feature_mean'], np.zeros(32)],
                               np.r_[saved['feature_scale'], np.ones(32)], np.zeros(4), np.ones(4))
        with np.load(model.D38 / 'temporal/normalization.npz', allow_pickle=False) as saved:
            mean, scale = saved['mean'].copy(), saved['scale'].copy()
        settings = FitSettings(coordinates=0, seed=500, epochs=60, batch=2048, label_weight=1.)
        params = model.initial(jax.random.key(500), settings)
        # Qualification preserves trained-parameter precision via a double export; no updates occur.
        double = jax.tree.map(lambda p: jnp.asarray(p, dtype=jnp.float64), params)
        x = np.clip((static-scaling.feature_mean[:582])/scaling.feature_scale[:582], -8., 8.)
        X, W, MU, SIG = map(jnp.asarray, (x, raw, mean, scale))
        assert np.array_equal(np.asarray(model.window(W)), model.unpack(raw))
        assert np.all(model.unpack(raw)[:, 20] == 0.)
        targets = np.random.default_rng(4001).normal(size=(213, 4))
        targets[::5, 2:] = np.nan
        targets[::7, :2] = np.nan
        y = jnp.asarray(targets)
        reports = {}
        for mode in model.MODES:
            folder = HERE / 'prototype_qualification_models' / mode
            folder.parent.mkdir(exist_ok=True)
            model.save(folder, params, scaling, settings, mean, scale, mode)
            exported = model.XTemporalNetwork(folder)
            labels = exported.predict(static, raw)
            actual = np.column_stack([labels[n] for n in model.LABELS])
            expected = np.asarray(model.forward(double, X, W, MU, SIG, mode))
            delta = float(np.max(np.abs(actual-expected)))
            assert np.isfinite(actual).all()
            assert np.allclose(actual, expected, atol=protocol['atol'], rtol=protocol['rtol']), (mode, delta)
            embedding = np.asarray(model.pool(double, W, MU, SIG, mode))
            assert np.allclose(exported.embedding(raw), embedding, atol=protocol['atol'], rtol=protocol['rtol'])
            if mode != 'ordered':
                assert np.array_equal(embedding, np.asarray(old_pool(double, W, MU, SIG, mode == 'mean')))
            ours, gradient = jax.value_and_grad(model.loss)(double, X, W, y, MU, SIG, mode)

            def reference(parameters):
                inp = jnp.concatenate([X, model.pool(parameters, W, MU, SIG, mode)], axis=1)
                return model.fit._loss(parameters, inp, y, jnp.ones(4), 0., None, jnp.eye(4))

            wanted, reference_gradient = jax.value_and_grad(reference)(double)
            assert np.allclose(ours, wanted, atol=protocol['atol'], rtol=protocol['rtol'])
            residual = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                           for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(reference_gradient), strict=True))
            assert all(np.allclose(a, b, atol=protocol['atol'], rtol=protocol['rtol'])
                       for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(reference_gradient), strict=True))
            reports[mode] = {'portable_double_max_abs': delta, 'masked_gradient_max_abs': residual,
                             'masked_loss': float(ours), 'all_four_outputs_finite': True,
                             'network_sha256': digest(folder / 'network.npz'),
                             'encoder_sha256': digest(folder / 'encoder.npz')}
            print(json.dumps({'prototype_qualified': mode, **reports[mode]}), flush=True)
        verify()
        with (HERE / 'prototype_qualification.json').open('x') as stream:
            json.dump({'status': 'all_three_X_temporal_prototypes_real213_inputs_portable_and_masked_objective_qualified',
                       'reports': reports, 'same_paired_initial_trunk': True,
                       'raw41_time_grid_with_true_zero_relative_center_exact': True,
                       'no_optimizer_updates_no_training': True, 'no_reference_labels_or_map_scores_read': True,
                       'native_full69_zero_contribution_integration_not_yet_qualified': True,
                       'not_candidate_admission': True, 'publication': False,
                       'protocol_sha256': digest(HERE / 'prototype_qualification_protocol.json'),
                       'seconds': time.time()-started}, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (HERE / 'prototype_qualification_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
