# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — shared CPU initialization for paired X experiments
"""Persist one source-bound initialization per seed before any GPU update."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import x_temporal_model as model
from fec_adapter.diii_d_accuracy.fit_network import FitSettings
import jax

HERE = Path(__file__).resolve().parent
OUT = HERE / 'shared_initial_parameters'
LISTS = ('weights', 'biases', 'conv_weights', 'conv_biases')
SINGLES = ('linear', 'readout', 'readout_bias')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        params = {name: saved[name].copy() for name in SINGLES}
        for name in LISTS:
            params[name] = [saved[name+'_'+str(i)].copy() for i in range(int(saved[name+'_count']))]
    assert all(np.isfinite(a).all() and a.dtype == np.float32 for a in jax.tree.leaves(params))
    return params


def main():
    started = time.monotonic()
    assert all(d.platform == 'cpu' for d in jax.devices())
    recipe_path = HERE / 'training_recipe_frozen_v1.json'
    recipe = json.loads(recipe_path.read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert digest(recipe_path) == protocol['recipe_sha256']
    assert digest(HERE / 'x_temporal_model.py') == protocol['model_sha256']
    assert digest(Path(__file__)) == protocol['initializer_source_sha256']
    reports = []
    for seed in recipe['paired_seeds']:
        settings = FitSettings(**{**recipe['fit_settings_seed500'], 'seed': seed})
        params = model.initial(jax.random.key(seed), settings)
        arrays = {name: np.asarray(params[name]) for name in SINGLES}
        for name in LISTS:
            arrays[name+'_count'] = np.asarray(len(params[name]), dtype=np.int64)
            for i, value in enumerate(params[name]):
                arrays[name+'_'+str(i)] = np.asarray(value)
        path = OUT / ('seed'+str(seed)+'.npz')
        assert not path.exists()
        np.savez(path, **arrays)
        restored = load(path)
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored), strict=True))
        reports.append({'seed': seed, 'file': str(path), 'sha256': digest(path), 'CPU_roundtrip_bitexact': True})
    assert len(reports) == 8
    with (OUT / 'receipt.json').open('x') as stream:
        json.dump({'status': 'all8_paired_CPU_initializations_saved_and_roundtrip_bitexact',
                   'models': reports, 'recipe_sha256': protocol['recipe_sha256'],
                   'no_optimizer_updates_or_data_or_labels_read': True,
                   'same_arrays_required_for_all3_families_and_GPUs': True,
                   'GPU_training_initial_state_identity_not_yet_qualified': True,
                   'seconds': time.monotonic()-started}, stream, indent=2)
    print(json.dumps({'shared_initializations': 8, 'no_updates': True}), flush=True)


if __name__ == '__main__':
    main()
