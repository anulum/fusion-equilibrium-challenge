# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible production-recipe training tables
"""Stream only eligible shots and normalize without the confirmation cohort."""
import json
import threading
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import HERE, PARTS, base_maps, metadata, rows_for, write_json
from fec_adapter.diii_d_accuracy.fit_network import _Scaling

TRANSFER = threading.Lock()
COMPILE = threading.Lock()
CHUNK = 16384


class Moments:
    """Accumulate finite per-column population moments in bounded host memory."""
    def __init__(self, size: int):
        self.count = np.zeros(size)
        self.mean = np.zeros(size)
        self.m2 = np.zeros(size)

    def add(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        count = finite.sum(axis=0).astype(float)
        mean = np.where(finite, values, 0.).sum(axis=0) / np.maximum(count, 1.)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += np.where(finite, values - mean, 0.).__pow__(2).sum(axis=0) + delta**2 * self.count * count / np.maximum(total, 1.)
        self.mean += delta * count / np.maximum(total, 1.)
        self.count = total

    def scale(self) -> np.ndarray:
        spread = np.sqrt(np.maximum(0., self.m2 / np.maximum(self.count, 1.)))
        return np.where(spread > 1e-6, spread, 1.)


def initialize() -> dict:
    """Verify frame counts and exact split disjointness before reading targets."""
    cohort = json.loads((HERE / 'cohort.json').read_text())
    allowed = set(cohort['training']) | set(cohort['validation'])
    held, calibration = set(cohort['holdout']), set(cohort['calibration'])
    assert len(allowed) == 6311 and not (allowed & held or calibration & held or calibration & allowed)
    metas = {p: metadata(p) for p in PARTS}
    rows = {'training': rows_for(metas['training'], allowed), 'calibration': rows_for(metas['calibration'], calibration)}
    assert len(rows['training']) == 1397668 and len(rows['calibration']) == 82797
    return {'base': base_maps(), 'metas': metas, 'rows': rows}


def scaling_for(data: dict, tc: bool) -> _Scaling:
    """Use source normalization including one shared scale for 64 coordinates."""
    cache = HERE / ('scaling_tc.npz' if tc else 'scaling_plain.npz')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            return _Scaling(*(stored[k].copy() for k in ('feature_mean', 'feature_scale', 'output_mean', 'output_scale')))
    xmom, ymom = Moments(582), Moments(142)
    for part in (PARTS if tc else ('training',)):
        ids, base = data['rows'][part], data['base'][part]
        for start in range(0, len(ids), CHUNK):
            take = ids[start:start+CHUNK]
            xmom.add(base['features'][take].astype(float))
            ymom.add(np.column_stack([base['coordinates'][take, :64], base['labels'][take]]).astype(float))
    assert (xmom.count > 0).all() and (ymom.count > 0).all()
    scale = ymom.scale()
    scale[:64] = np.sqrt(np.sum(ymom.m2[:64] / ymom.count[:64]))
    result = _Scaling(xmom.mean, xmom.scale(), ymom.mean, scale)
    np.savez(cache, **result.__dict__)
    write_json(cache.with_suffix('.json'), {'training_rows': 1480465 if tc else 1397668,
                                         'holdout_used': False, 'method': 'finite_population_moments_shared_coordinate_scale'})
    return result


@partial(jax.jit, donate_argnums=(0,))
def put_block(array, block, start):
    return jax.lax.dynamic_update_slice(array, block, (start, 0))


def upload(data: dict, scaling: _Scaling, tc: bool, device):
    """Upload F582 inputs and 142 targets once per card and permitted row group."""
    n = 1480465 if tc else 1397668
    with TRANSFER, jax.default_device(device):
        x, y = jnp.zeros((n, 582), jnp.float32), jnp.zeros((n, 142), jnp.float32)
        at = 0
        for part in (PARTS if tc else ('training',)):
            ids, base = data['rows'][part], data['base'][part]
            for start in range(0, len(ids), CHUNK):
                take = ids[start:start+CHUNK]
                raw = base['features'][take].astype(float)
                features = np.clip((raw-scaling.feature_mean)/scaling.feature_scale, -8., 8.).astype(np.float32)
                labels = np.column_stack([base['coordinates'][take, :64], base['labels'][take]]).astype(float)
                labels = ((labels-scaling.output_mean)/scaling.output_scale).astype(np.float32)
                x = put_block(x, jax.device_put(features, device), at)
                y = put_block(y, jax.device_put(labels, device), at)
                x.block_until_ready(); y.block_until_ready()
                at += len(take)
        assert at == n
        raw = data['base']['calibration']['features'].astype(float)
        held = jax.device_put(np.clip((raw-scaling.feature_mean)/scaling.feature_scale, -8., 8.).astype(np.float32), device)
        held.block_until_ready()
    return x, y, held
