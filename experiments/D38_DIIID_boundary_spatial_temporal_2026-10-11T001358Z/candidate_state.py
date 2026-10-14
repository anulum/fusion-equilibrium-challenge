# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — candidate optimizer checkpoint storage
"""Preserve the complete optimizer tree and random state without pickle."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np


def save(path: Path, state: tuple, key: jax.Array, epoch: int, recipe: dict) -> None:
    """Atomically retain all parameter and optimizer leaves for the same recipe."""
    leaves = jax.tree.leaves(state)
    arrays = {f'leaf_{i}': np.asarray(v) for i, v in enumerate(leaves)}
    arrays.update(epoch=np.asarray(epoch), key=np.asarray(jax.random.key_data(key)),
                  recipe=np.asarray(json.dumps(recipe, sort_keys=True)), count=np.asarray(len(leaves)))
    temporary = path.with_name('checkpoint.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def restore(path: Path, template: tuple, recipe: dict) -> tuple:
    """Restore a checkpoint only when its recipe and tree shape match."""
    leaves, structure = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored['recipe']) == json.dumps(recipe, sort_keys=True)
        assert int(stored['count']) == len(leaves)
        restored = [jnp.asarray(stored[f'leaf_{i}']) for i in range(len(leaves))]
        assert all(x.shape == y.shape and x.dtype == y.dtype for x, y in zip(leaves, restored, strict=True))
        return (jax.tree.unflatten(structure, restored),
                jax.random.wrap_key_data(jnp.asarray(stored['key'])), int(stored['epoch']))
