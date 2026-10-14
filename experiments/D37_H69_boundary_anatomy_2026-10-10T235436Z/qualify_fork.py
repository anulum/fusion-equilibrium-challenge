# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — real native parity with shared read-only model weights
"""Check bounded input conversion and forked native predictions before the pool."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import json
import numpy as np
import production_path as production


def public_probe(path: str) -> dict:
    """Exercise the native prediction pipeline in a forked worker."""
    maps, scalars, _shot = production.maps_of(Path(path))
    expected = np.load(production.HERE / 'profiles/uninstrumented_public_psirz.npy', allow_pickle=False)
    assert np.array_equal(expected, maps['final'])
    archive = production.E / 'D36_production_scalar_submission_2026-10-10T223719Z/submission/diii_d_public_test.npz'
    with np.load(archive, allow_pickle=False) as stored:
        for name in ('q95', 'betaN'):
            assert np.array_equal(scalars[name], stored['shot_0000_' + name])
    return {'status': 'real_forked_current_native_map_and_archived_scalars_bitexact'}


if __name__ == '__main__':
    production.initialize()
    directory = Path('$FEC_DATA/data/diii_d_public_test')
    path = sorted(directory.glob('*.parquet'))[0]
    original = production.STATE.legacy_read_row(path)
    bounded = production.STATE.read_row(path)
    assert set(original) == set(bounded)
    checked = []
    for name in original:
        if isinstance(bounded[name], np.ndarray):
            assert np.array_equal(np.asarray(original[name], dtype=float), bounded[name], equal_nan=True)
            checked.append(name)
    del original, bounded
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        receipt = pool.submit(public_probe, str(path)).result(timeout=600)
    receipt['bounded_numeric_input_columns_verified'] = checked
    receipt['read_only_models_loaded_once_shared_by_fork'] = True
    with (production.HERE / 'fork_parity.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)
