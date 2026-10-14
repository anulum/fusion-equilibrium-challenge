# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bounded eligible map-candidate transfers
"""Upload one corpus per card and normalize only the boundary loss by shots."""
from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp

from campaign_common import HERE, PARTS
from training_data import TRANSFER, put_block


def row_weights(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return comparable frame- and shot-normalized valid-ray weights."""
    validity, identities = [], []
    offset = 0
    for part in PARTS:
        folder = HERE / 'supervision'
        valid = np.load(folder / (part + '_valid_ray_rows.npy'), allow_pickle=False)
        shot = np.load(folder / (part + '_shot_ids.npy'), allow_pickle=False)
        assert len(valid) == len(data['rows'][part])
        validity.append(valid)
        identities.append(shot + offset)
        offset += int(shot.max()) + 1
    valid = np.concatenate(validity)
    shot = np.concatenate(identities)
    counts = np.bincount(shot[valid], minlength=offset)
    frame = valid.astype(float) * len(valid) / valid.sum()
    weighted = np.zeros(len(valid), dtype=float)
    weighted[valid] = len(valid) / (np.count_nonzero(counts) * counts[shot[valid]])
    assert abs(frame.mean() - 1) < 1e-12 and abs(weighted.mean() - 1) < 1e-12
    return frame.astype(np.float32), weighted.astype(np.float32)


def upload(data: dict, scaling, device):
    """Transfer only the frozen eligible rows using bounded host chunks."""
    assert json.loads((HERE / 'supervision/receipt.json').read_text())['status'] == 'all6326_eligible_native_geometry_and_ray_targets_complete'
    total = sum(len(data['rows'][part]) for part in PARTS)
    frame, shot = row_weights(data)
    with TRANSFER, jax.default_device(device):
        arrays = [jnp.zeros((total, width), dtype=jnp.float32) for width in (582, 142, 128, 86)]
        at = 0
        for part in PARTS:
            ids, base = data['rows'][part], data['base'][part]
            points = np.load(HERE / 'supervision' / (part + '_points.npy'), mmap_mode='r', allow_pickle=False)
            assert len(points) == len(ids)
            for start in range(0, len(ids), 8192):
                chosen = ids[start:start + 8192]
                raw = base['features'][chosen].astype(float)
                inputs = np.clip((raw - scaling.feature_mean) / scaling.feature_scale, -8., 8.).astype(np.float32)
                wanted = np.column_stack([base['coordinates'][chosen, :64], base['labels'][chosen]]).astype(float)
                wanted = ((wanted - scaling.output_mean) / scaling.output_scale).astype(np.float32)
                blocks = [inputs, wanted, np.asarray(base['coordinates'][chosen, :128], dtype=np.float32),
                          np.asarray(points[start:start + len(chosen)], dtype=np.float32).reshape(len(chosen), 86)]
                for index, block in enumerate(blocks):
                    arrays[index] = put_block(arrays[index], jax.device_put(block, device), at)
                    arrays[index].block_until_ready()
                at += len(chosen)
        assert at == total
        return (*arrays[:3], arrays[3].reshape(total, 43, 2), jax.device_put(frame, device), jax.device_put(shot, device))
