# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — H69 public-test maps with the static X-point specialists blended into the X targets
"""Build the H69 map archive as D25 did, with the four X-point targets moved a share of the way to the specialists' mean.

The blend is applied to the networks' X-point outputs after the tree blend and before the finisher, which is where
the finisher's X-point move takes its targets from (its target smoothing is off in the deployed flags). A share of
zero, or no specialists, runs the unchanged D25 path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path('$HOME/fec_first_agent_d16')
sys.path.insert(0, str(ROOT))
import deploy_build_subframe as b

RUNS = Path('$FEC_RUNS/ml350/runs')
PKG = RUNS / 'legacy_tree_fec_first_agent_20261005/deploy/pkg'
DATA = Path('$FEC_DATA/data')
WEAK = str(ROOT / 'v29_readout_s2/discharge_hx_seed2_all/network.npz')
TREE_WEIGHTS = [('li_scored', .26), ('r_axis', .14), ('kappa', .09), ('x_lower_r', .16)]
LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')
X = {'files': [], 'share': 0.0, 'networks': None, 'shots': 0}
NATIVE_BLEND = b._blended


def blended_with_x(networks, features):
    """The tree-blended answer of D25, then the X targets moved ``share`` of the way to the specialists' mean."""
    coordinates, labels = NATIVE_BLEND(networks, features)
    if X['share'] > 0.0:
        if X['networks'] is None:
            pd = b._load_entrypoint(PKG)
            X['networks'] = [pd.Network.load(path) for path in X['files']]
            assert len(X['networks']) == 8 and all(n.coordinates == 0 and n.label_names == LABELS for n in X['networks'])
        matrix = features['discharge']
        assert matrix.shape[1] == 582
        padded = np.concatenate([matrix, np.zeros((len(matrix), 32))], axis=1)
        answers = [network.predict(padded)[1] for network in X['networks']]
        for name in LABELS:
            expert = np.mean([answer[name] for answer in answers], axis=0)
            assert expert.shape == labels[name].shape and np.isfinite(expert).all()
            labels[name] = (1. - X['share']) * np.asarray(labels[name], dtype=float) + X['share'] * expert
        X['shots'] += 1
    return coordinates, labels


def run(task):
    core, files, share = task
    X.update(files=files, share=share, networks=None, shots=0)
    b._blended = blended_with_x
    try:
        out = b._predict_many(core)
    finally:
        b._blended = NATIVE_BLEND
    assert share == 0.0 or X['shots'] == len(core[1])
    return out


def plan(limit=None):
    pd = b._load_entrypoint(PKG)
    nets = b.verified(ROOT / 'networks_v29_h3.sha', [])
    assert nets.count(WEAK) == 1
    nets.remove(WEAK)
    assert len(nets) == 69
    args = pd._parser().parse_args(['--data-root', str(DATA), '--network', *nets, '--basis', str(PKG / 'models/flux_basis.npz'),
                                    '--out', '/dev/null', '--split', 'diii_d_public_test', *b.FLAGS])
    files = [str(p) for p in pd.split_files(DATA, 'diii_d_public_test', None)]
    assert len(files) == 874
    return pd, nets, pd._finish_from(args), files[:limit] if limit else files


def core_task(files, nets, finish):
    return (str(PKG), files, nets, [], str(PKG / 'models/flux_basis.npz'), finish, None,
            str(ROOT / 'tree_heads_public_test.npz'), TREE_WEIGHTS, str(ROOT / 'out_public_test.npz'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-list', type=Path, required=True, help='sha256-per-line list of the eight static specialists')
    parser.add_argument('--x-share', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    assert 0.0 < args.x_share <= 1.0
    started = time.time()
    specialists = b.verified(args.x_list, [])
    assert len(specialists) == 8
    pd, nets, finish, files = plan(args.limit)
    args.out.mkdir(parents=True, exist_ok=False)
    bounds = np.linspace(0, len(files), min(args.workers, len(files)) + 1).astype(int)
    tasks = [(core_task(files[x:y], nets, finish), specialists, args.x_share) for x, y in zip(bounds[:-1], bounds[1:])]
    archive = pd.PredictionArchive()
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('fork')) as pool:
        for predictions in pool.map(run, tasks):
            for maps, q95, beta in predictions:
                assert np.isfinite(maps).all() and np.isfinite(q95).all() and np.isfinite(beta).all()
                archive.add(maps, q95, beta)
    assert archive.replaced == 0 and archive.shots == len(files)
    archive.write(args.out / 'diii_d_public_test.npz')
    receipt = {'status': 'h69_maps_with_static_x_specialists', 'x_share': args.x_share, 'x_specialists': specialists,
               'x_list_sha256': b.digest(args.x_list), 'network_count': len(nets), 'excluded': WEAK,
               'tree_weights': dict(TREE_WEIGHTS), 'shots': archive.shots, 'frames': archive.frames, 'limit': args.limit,
               'builder_sha256': b.digest(b.__file__), 'this_script_sha256': b.digest(__file__),
               'entrypoint_sha256': b.digest(pd.__file__), 'archive_sha256': b.digest(args.out / 'diii_d_public_test.npz'),
               'scalars_note': 'q95 and betaN of this archive are the map networks alone; the submitted scalars are taken from the D36 archive',
               'seconds': time.time() - started}
    (args.out / 'deploy_receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'x_specialists'}), flush=True)


if __name__ == '__main__':
    main()
