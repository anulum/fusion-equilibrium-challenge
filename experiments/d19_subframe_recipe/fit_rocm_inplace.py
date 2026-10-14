# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D19: the accelerator fit with a device copy that does not need the table twice on the card
"""fit_on_accelerators unchanged except on_device: the table is written into one preallocated device array chunk by
chunk (the buffer is donated to a compiled update), instead of joining chunks on the card. The 582-column all-data
table is 3.2 GB; chunks plus their join need 6.4 GB at once and do not fit an 8 GB card beside the targets.
The values on the card are the same; nothing in the fit changes.

    python fit_rocm_inplace.py --tables DIR --out-root DIR --reserved FILE --variants FILE [--train-on-calibration]
"""
import jax
import jax.numpy as jnp
import numpy as np

from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit


@jax.jit
def _written(target, piece, start):
    return jax.lax.dynamic_update_slice(target, piece, (start,) + (0,) * (target.ndim - 1))


_filled = jax.jit(_written.__wrapped__, donate_argnums=(0,))


def on_device(array: np.ndarray, device):
    with fit._TRANSFER:
        with jax.default_device(device):
            first = jax.device_put(array[: fit.CHUNK_ROWS], device)
            if array.shape[0] <= fit.CHUNK_ROWS:
                first.block_until_ready()
                return first
            joined = jax.device_put(jnp.zeros(array.shape, dtype=first.dtype), device)
            joined = _filled(joined, first, 0)
            del first
            for start in range(fit.CHUNK_ROWS, array.shape[0], fit.CHUNK_ROWS):
                joined = _filled(joined, jax.device_put(array[start : start + fit.CHUNK_ROWS], device), start)
            joined.block_until_ready()
    return joined


fit.on_device = on_device
raise SystemExit(fit.main())
