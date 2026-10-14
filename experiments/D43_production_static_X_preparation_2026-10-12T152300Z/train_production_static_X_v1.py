# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — confirmed all-data static X-point specialist fits
"""Refit exactly eight static specialists with the unchanged qualified kernel."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

from prepare_production_X_data_v1 import HERE, digest, verify, write


def main() -> None:
    """Verify activation, upload every labelled frame, and run all eight fits."""
    activation = verify()
    import numpy as np
    draft = json.loads((HERE / 'preparation_draft_v1.json').read_text())
    d40, d38 = Path(draft['D40_root']), Path(draft['D38_root'])
    sys.path.insert(0, str(d40))
    import train_X_specialists_v3 as trainer
    import x_training_kernel_v2 as kernel
    from campaign_common import mapped_array
    from training_data import put_block
    import jax
    import jax.numpy as jnp
    assert activation['runtime_versions'] == {'jax': jax.__version__, 'jaxlib': trainer.jaxlib.__version__, 'numpy': np.__version__}
    assert len(jax.devices()) == 3 and all(device.platform == 'gpu' for device in jax.devices())
    plan = json.loads((HERE / 'metadata_row_plan_v1.json').read_text())
    data_root = HERE / 'all7041_data'
    receipt_path = data_root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    activation_path = HERE / 'production_activation_v1.json'
    assert receipt['status'] == 'all7041_native_X_targets_and_active_normalizers_complete'
    assert receipt['activation_sha256'] == digest(activation_path)
    assert receipt['totals'] == plan['totals'] and receipt['label_names'] == draft['label_names']
    assert digest(data_root / 'normalization.npz') == receipt['normalization_sha256']
    for part, path in receipt['target_paths'].items():
        assert digest(Path(path)) == receipt['target_sha256'][part]
    with np.load(data_root / 'normalization.npz', allow_pickle=False) as saved:
        normalization = {name: saved[name].copy() for name in
                         ('feature_mean', 'feature_scale', 'output_mean', 'output_scale', 'temporal_mean', 'temporal_scale')}
    assert all(np.isfinite(a).all() for a in normalization.values())
    assert all(normalization[k].shape == (582,) for k in ('feature_mean', 'feature_scale'))
    assert all(normalization[k].shape == (4,) for k in ('output_mean', 'output_scale'))
    assert all(normalization[k].shape == (21,) for k in ('temporal_mean', 'temporal_scale'))
    assert all((normalization[k] > 0).all() for k in ('feature_scale', 'output_scale', 'temporal_scale'))
    n = sum(row['frames'] for row in receipt['totals'].values())
    assert n == 1559340
    out = HERE / 'all7041_fits'
    out.mkdir(exist_ok=True)
    device = jax.devices()[0]
    with jax.default_device(device):
        x, y = jnp.zeros((n, 582), jnp.float32), jnp.zeros((n, 4), jnp.float32)
        raw = jnp.zeros((n, 840), jnp.float16)
        at = 0
        for part in ('training', 'calibration'):
            features = mapped_array(Path(plan['table_paths'][part]), 'features')
            targets = np.load(receipt['target_paths'][part], mmap_mode='r', allow_pickle=False)
            assert targets.shape == (len(features), 4)
            for start in range(0, len(features), 4096):
                inputs = np.asarray(features[start:start + 4096], dtype=float)
                inputs = np.clip((inputs - normalization['feature_mean']) / normalization['feature_scale'], -8., 8.).astype(np.float32)
                wanted = np.asarray(targets[start:start + 4096], dtype=float)
                wanted = ((wanted - normalization['output_mean']) / normalization['output_scale']).astype(np.float32)
                assert np.isfinite(inputs).all() and not np.isinf(wanted).any()
                x = put_block(x, jax.device_put(inputs, device), at)
                y = put_block(y, jax.device_put(wanted, device), at)
                x.block_until_ready(); y.block_until_ready()
                at += len(inputs)
            print(json.dumps({'production_uploaded': part, 'rows': at}), flush=True)
        assert at == n
        raw.block_until_ready()
        state, _ = kernel.initial_state(500, device)
        original = mapped_array(d38 / 'temporal/windows/rawwindow_training.npz', 'features')
        real_raw = jax.device_put(np.asarray(original[:16]), device)
        mean, scale = [jnp.asarray(normalization[key], dtype=jnp.float32) for key in ('temporal_mean', 'temporal_scale')]
        value_grad = jax.value_and_grad(kernel.model.loss)
        real_loss, real_grad = value_grad(state[0], x[:16], real_raw, y[:16], mean, scale, 'static')
        zero_loss, zero_grad = value_grad(state[0], x[:16], raw[:16], y[:16], mean, scale, 'static')
        assert np.isfinite(np.asarray(y[:16])).any()
        assert np.isfinite(float(real_loss)) and np.array_equal(np.asarray(real_loss), np.asarray(zero_loss))
        assert any(np.any(np.asarray(a) != 0) for a in jax.tree.leaves(real_grad))
        assert all(np.isfinite(np.asarray(a)).all() and np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(jax.tree.leaves(real_grad), jax.tree.leaves(zero_grad), strict=True))
        qualification = out / 'static_zero_window_loss_gradient_qualification.json'
        if not qualification.exists():
            write(qualification, {'status': 'actual_GPU_static_real_vs_zero_raw_loss_gradient_bitexact',
                  'activation_sha256': digest(activation_path), 'loss': float(real_loss),
                  'optimizer_updates': 0, 'scope': 'Raw dependency only; original windows need not align with the sixteen production rows.'})
        records = []
        for seed in range(500, 508):
            destination = out / f'static_{seed}'
            destination.mkdir(exist_ok=True)
            prior = [destination / name for name in ('numerical_failure.json', 'integrity_failure.json')]
            existing = next((path for path in prior if path.exists()), None)
            if existing is not None:
                record = json.loads(existing.read_text())
                assert record['activation_sha256'] == digest(activation_path) and record['seed'] == seed
                records.append(record)
                continue
            try:
                records.append(trainer.fit_one('static', seed, (x, y, raw), normalization,
                                               device, out, activation_path))
            except Exception as error:
                if isinstance(error, kernel.NumericalTrainingError):
                    status, path = 'numerical_failure_no_retry', prior[0]
                elif isinstance(error, AssertionError):
                    status, path = 'integrity_failure_no_retry', prior[1]
                else:
                    status = 'operational_fault_resume_allowed'
                    folder = destination / 'operational_faults'
                    folder.mkdir(exist_ok=True)
                    path = folder / f'production_attempt_{time.time_ns()}.json'
                record = {'status': status, 'seed': seed, 'mode': 'static',
                          'activation_sha256': digest(activation_path),
                          'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()}
                write(path, record)
                records.append(record)
    verify()
    assert len(records) == 8 and {r['seed'] for r in records} == set(range(500, 508))
    unresolved = any(r['status'] in ('operational_fault_resume_allowed', 'integrity_failure_no_retry') for r in records)
    passed = all(r['status'] == 'fit_complete' and r['quality']['seed_quality_pass'] for r in records)
    write(out / f'attempt_{time.time_ns()}.json',
          {'status': 'non_numerical_resolution_required' if unresolved else 'all8_production_static_fits_terminal',
           'fits': records, 'family_training_quality_pass': None if unresolved else passed,
           'activation_sha256': digest(activation_path), 'no_reduced_member_subset': True,
           'no_deployment_list_before_checkpoint_portable_native_FORK_qualification': True})


if __name__ == '__main__':
    main()
