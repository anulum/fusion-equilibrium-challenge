# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D6: out-of-fold maps with V0 flux coordinates and specialist scalar heads
"""Finish one fold's held-out shots as kfold_build does, with one change: the scalar outputs the
finisher and the archive read (SCALAR_LABELS) are the mean of a specialist network set, while the
flux coordinates, X-point and boundary-radius outputs stay the V0 set's mean.

The shots, their order, the v25 finisher and the V0 networks are those of kfold/oof_fold{K}
(receipt checked); the specialist networks are those of the receipt given with --specialist.

    python d6_build.py --fold K --specialist DIR/oof_foldK --out DIR [--workers 5]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAT = HERE.parent
sys.path.insert(0, str(SEAT / 'd4c'))
import kfold_build as kb  # noqa: E402  (v25 finisher, data and basis paths)

vbuild = kb.vbuild

SCALAR_LABELS = ('q95', 'beta_n', 'li', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume')
"""Every scalar head of the S16 families; the X-point and boundary-radius outputs are geometry, not scalars."""

_ORIGINAL = vbuild.predict_together
_SPECIALISTS = []


def _swapped(networks, features):
    coordinates, labels = _ORIGINAL(networks, features)
    _, special = _ORIGINAL(_SPECIALISTS, features)
    for name in SCALAR_LABELS:
        if name in labels:
            if name not in special:
                raise ValueError(f'the specialist set has no output {name}')
            labels[name] = special[name]
    return coordinates, labels


def _predict_many(task):
    files, v0_files, special_files, basis_file, finish = task
    networks = [vbuild.Network.load(f) for f in v0_files]
    _SPECIALISTS[:] = [vbuild.Network.load(f) for f in special_files]
    if {n.feature_layout for n in _SPECIALISTS} - {n.feature_layout for n in networks}:
        raise ValueError('a specialist reads a feature layout the V0 set does not')
    vbuild.predict_together = _swapped
    basis = vbuild.FluxBasis.load(basis_file).truncated(networks[0].coordinates)
    return [vbuild.predict_shot(path, networks, basis, finish) for path in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--specialist', type=Path, required=True, help='an oof_fold directory whose receipt lists the specialist networks')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    base = json.loads((SEAT / 'kfold' / f'oof_fold{args.fold}' / 'receipt.json').read_text())
    special = json.loads((args.specialist / 'receipt.json').read_text())
    assert base['fold'] == special['fold'] == args.fold and base['stems'] == special['stems']
    finish, values = kb.v25_finish()
    assert {k: str(v) for k, v in values.items()} == base['finish'], 'finisher differs from the V0 build'
    v0 = [Path(p) for p in base['networks']]
    spec = [Path(p) for p in special['networks']]
    assert [kb.digest(p) for p in v0] == base['network_sha256']
    assert [kb.digest(p) for p in spec] == special['network_sha256']
    paths = [kb.DATA / f'{s}.parquet' for s in base['stems']]
    args.out.mkdir(parents=True, exist_ok=False)
    import itertools
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    bounds = np.linspace(0, len(paths), min(args.workers, len(paths)) + 1).astype(int)
    runs = [list(paths[a:b]) for a, b in itertools.pairwise(bounds)]
    archive = vbuild.PredictionArchive()
    with ProcessPoolExecutor(max_workers=len(runs), mp_context=multiprocessing.get_context('spawn')) as pool:
        for predictions in pool.map(_predict_many, [(r, v0, spec, kb.BASIS, finish) for r in runs]):
            for maps, q95, beta_n in predictions:
                archive.add(maps, q95, beta_n)
    archive.write(args.out / 'diii_d_train.npz')
    receipt = {'status': 'd6_headswap_oof_maps_built', 'fold': args.fold, 'shots': archive.shots,
               'frames': archive.frames, 'replaced': archive.replaced, 'stems': base['stems'],
               'networks': base['networks'], 'network_sha256': base['network_sha256'],
               'specialist_networks': special['networks'], 'specialist_sha256': special['network_sha256'],
               'scalar_labels_from_specialist': list(SCALAR_LABELS), 'finish': base['finish'],
               'builder': str(Path(vbuild.__file__)), 'builder_sha256': kb.digest(vbuild.__file__),
               'this_script_sha256': kb.digest(__file__),
               'archive_sha256': kb.digest(args.out / 'diii_d_train.npz'), 'seconds': time.time() - started,
               'validation_targets_used': False, 'MAST_targets_used': False}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('stems', 'networks', 'network_sha256',
                                                                  'specialist_networks', 'specialist_sha256')}), flush=True)


if __name__ == '__main__':
    main()
