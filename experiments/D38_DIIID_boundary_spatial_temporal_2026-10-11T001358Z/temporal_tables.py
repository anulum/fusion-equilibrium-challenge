# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bounded temporal training transfers
"""Keep raw input caches in half precision and training targets in single precision."""
import json
import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import HERE, PARTS
from window_cache import mapped_array
from training_data import TRANSFER, put_block


def upload(data: dict, scaling, device: jax.Device) -> tuple:
    """Transfer only eligible aligned input windows and two scalar targets."""
    assert json.loads((HERE / 'temporal/preparation_receipt.json').read_text())['status'] == 'eligible_channel_statistics_and_all_development_windows_complete'
    total = sum(len(data['rows'][p]) for p in PARTS)
    names = data['metas']['training']['label_names']
    ids = [64 + names.index(n) for n in ('q95', 'beta_n')]
    with TRANSFER, jax.default_device(device):
        static = jnp.zeros((total, 582), dtype=jnp.float32)
        targets = jnp.zeros((total, 2), dtype=jnp.float32)
        windows = jnp.zeros((total, 840), dtype=jnp.float16)
        at = 0
        for part in PARTS:
            selected = data['rows'][part]
            base = data['base'][part]
            raw = mapped_array(HERE / 'temporal/windows' / f'rawwindow_{part}.npz', 'features')
            assert len(raw) == len(selected)
            for start in range(0, len(raw), 4096):
                chosen = selected[start:start + 4096]
                x = base['features'][chosen].astype(float)
                x = np.clip((x - scaling.feature_mean) / scaling.feature_scale, -8., 8.).astype(np.float32)
                y = base['labels'][chosen][:, [names.index(n) for n in ('q95', 'beta_n')]].astype(float)
                y = ((y - scaling.output_mean[ids]) / scaling.output_scale[ids]).astype(np.float32)
                static = put_block(static, jax.device_put(x, device), at)
                targets = put_block(targets, jax.device_put(y, device), at)
                windows = put_block(windows, jax.device_put(np.asarray(raw[start:start + len(chosen)]), device), at)
                static.block_until_ready(); targets.block_until_ready(); windows.block_until_ready()
                at += len(chosen)
        assert at == total
    return static, targets, windows
