# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D6 deployment: public DIII-D archive, 195-net flux + specialist scalar heads
"""Run the #949223 package's own DIII-D entrypoint functions with the exact flags of build_public_v25v27v28.py and its
195 networks (v25 package, v27, v28; checksums verified), with one change: the ten scalar outputs the finisher and the
archive read come from the mean of the specialist networks listed in --specialists (a sha256-per-line file).
--no-swap runs the same code without the change (a reproduction check against the deployed archive).

    python deploy_build.py --specialists FILE --out DIR [--workers 3] [--limit N] [--no-swap]
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAT = HERE.parents[1]
PKG = Path('$LAB/agentic_group_workspace/'
           'second_agent-recovery-20260926/results/joint_fixed_axis_volume_reproduction_56cb')
for p in (PKG / 'entrypoints', PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter'):
    sys.path.insert(0, str(p))
import predict_diii_d as pd  # noqa: E402

sys.path.insert(0, str(SEAT / 'alldata'))
import build_public_v25v27v28 as public  # noqa: E402  (FLAGS, DATA, network lists)

SCALAR_LABELS = ('q95', 'beta_n', 'li', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume')
_ORIGINAL = pd.predict_together
_SPECIALISTS = []


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _swapped(networks, features):
    coordinates, labels = _ORIGINAL(networks, features)
    _, special = _ORIGINAL(_SPECIALISTS, features)
    for name in SCALAR_LABELS:
        if name in labels:
            labels[name] = special[name]
    return coordinates, labels


def _predict_many(task):
    files, v0_files, special_files, basis_file, finish, swap = task
    networks = [pd.Network.load(f) for f in v0_files]
    if swap:
        _SPECIALISTS[:] = [pd.Network.load(f) for f in special_files]
        pd.predict_together = _swapped
    basis = pd.FluxBasis.load(basis_file).truncated(networks[0].coordinates)
    return [pd.predict_shot(path, networks, basis, finish, None) for path in files]


def networks_195():
    manifest = json.loads((public.PKG / 'package_manifest.json').read_text())
    v25 = []
    for item in manifest['recipe']:
        p = public.PKG / item['path']
        assert digest(p) == item['sha256'], p
        v25.append(str(p))
    rest = []
    for tag in ('v27', 'v28'):
        for line in (SEAT / 'alldata' / f'networks_{tag}.sha').read_text().splitlines():
            sha, p = line.split(maxsplit=1)
            assert digest(p) == sha, p
            rest.append(p)
    assert len(v25) == 65 and len(rest) == 130
    basis = public.PKG / manifest.get('basis_path', 'models/flux_basis.npz')
    assert digest(basis) == manifest['basis_sha256']
    return v25 + rest, basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specialists', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-swap', action='store_true')
    a = ap.parse_args()
    started = time.time()
    v0, basis = networks_195()
    special = []
    for line in a.specialists.read_text().splitlines():
        sha, p = line.split(maxsplit=1)
        assert digest(p) == sha, p
        special.append(p)
    assert len(special) == 16
    args = pd._parser().parse_args(['--data-root', str(public.DATA), '--network', *v0, '--basis', str(basis),
                                    '--out', str(a.out), '--split', 'diii_d_public_test', *public.FLAGS])
    finish = pd._finish_from(args)
    files = pd.split_files(public.DATA, 'diii_d_public_test', None)
    if a.limit:
        files = files[:a.limit]
    a.out.mkdir(parents=True, exist_ok=False)
    bounds = __import__('numpy').linspace(0, len(files), min(a.workers, len(files)) + 1).astype(int)
    runs = [list(files[x:y]) for x, y in itertools.pairwise(bounds)]
    archive = pd.PredictionArchive()
    with ProcessPoolExecutor(max_workers=len(runs), mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = [(r, v0, special, basis, finish, not a.no_swap) for r in runs]
        for predictions in pool.map(_predict_many, tasks):
            for maps, q95, beta_n in predictions:
                archive.add(maps, q95, beta_n)
    if archive.replaced:
        raise RuntimeError('Refuse silent nonfinite-to-zero replacement')
    archive.write(a.out / 'diii_d_public_test.npz')
    receipt = {'status': 'd6_deploy_build', 'swap': not a.no_swap, 'scalar_labels_from_specialist': list(SCALAR_LABELS),
               'shots': archive.shots, 'frames': archive.frames, 'limit': a.limit, 'flags': public.FLAGS,
               'networks_195': v0, 'specialists': special, 'specialists_file_sha256': digest(a.specialists),
               'entrypoint_sha256': digest(pd.__file__), 'this_script_sha256': digest(__file__),
               'archive_sha256': digest(a.out / 'diii_d_public_test.npz'), 'seconds': time.time() - started,
               'MAST_targets_used': False}
    (a.out / 'deploy_receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('networks_195', 'specialists', 'flags')}), flush=True)


if __name__ == '__main__':
    main()
