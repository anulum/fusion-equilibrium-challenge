# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — public DIII-D archive builder for networks that read the sub-frame input features
"""DIII-D build with an arbitrary verified network list and the sub-frame features appended to the inputs (D19).

deploy_build_blend.py (sha e6cd21eb) plus two things: --list FILE (repeatable) names the networks instead of the
fixed v25 + v27 + v28 set (--with-v25 adds the package's 65), and --subframe FILE appends each shot's precomputed
sub-frame feature block to the predictor's feature matrix, so networks fitted on the 582-column table read their
inputs; older networks read their own leading 367 columns of the same matrix. The description of the parent follows.

The deployed 195-network DIII-D build; named heads are blended with precomputed boosted-tree heads (D9).

deploy_build_portable.py (sha 90338b0a) plus one step: after the networks answer together, every head named in
--tree-weight NAME=W becomes network + W * (tree - network), the tree head read from --tree-heads for the shot at
hand. Everything downstream (axis move, shape edit, X-points, contour, the two submitted scalars) then uses the
blended head exactly as it used the network head. Without --tree-heads the build is the portable builder's.

The original description follows.

The deployed 195-network DIII-D build with optional scalar-head swap and optional finisher overrides.

Same functions and flags as deploy_build.py (the #949223 package entrypoint and the build_public_v25v27v28 flags),
but every path is an argument, so it runs wherever the package, the networks and the data are mounted. Network lists
are sha256-per-line files; --map OLD=NEW rewrites path prefixes written on the workstation.

    python deploy_build_portable.py --pkg DIR --data-root DIR --v27 FILE --v28 FILE --out DIR
        [--specialists FILE] [--map OLD=NEW ...] [--override FLAG=VALUE ...] [--workers N] [--limit N]

Without --specialists the scalar heads are the ensemble's own (the deployed archive, up to platform rounding).
--override replaces the value of one finisher flag, e.g. --override contour-share=0.65.
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FLAGS = ['--kept-coordinates', '45', '--smooth-from', '30', '--smooth-axis', '0', '--shape', 'kappa', 'tri_top', 'tri_bot',
         'volume', 'li', '--x-points', '--reshape-after-x-points', '--geometric-share', '0.5', '--li-share', '0.6',
         '--edit-margin', '1', '--contour', '--contour-first', '--contour-share', '0.5', '--contour-rounds', '1',
         '--scalar-share', 'kappa=0.5', 'volume=0.5', 'tri_top=0.25', 'tri_bot=0.25']
"""Exactly build_public_v25v27v28.FLAGS (the deployed finisher)."""

SCALAR_LABELS = ('q95', 'beta_n', 'li', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume')
_STATE = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mapped(path, maps):
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def verified(listing, maps):
    out = []
    for line in Path(listing).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        sha, path = line.split(maxsplit=1)
        path = mapped(path, maps)
        assert digest(path) == sha, path
        out.append(path)
    return out


def _load_entrypoint(pkg):
    for p in (Path(pkg) / 'entrypoints', Path(pkg) / 'source/fec_adapter/vendor', Path(pkg) / 'source/fec_adapter'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import predict_diii_d as pd
    return pd


def _swapped(networks, features):
    coordinates, labels = _STATE['original'](networks, features)
    _, special = _STATE['original'](_STATE['specialists'], features)
    names = SCALAR_LABELS if _STATE['swap'] == 'scalars' else tuple(labels)
    for name in names:
        if name in labels and name in special:
            labels[name] = special[name]
    return coordinates, labels


def _blended(networks, features):
    """The networks' answer with the named heads moved a weight of the way to the tree heads of this shot."""
    coordinates, labels = _STATE['together'](networks, features)
    tree = _STATE['tree']
    stem = _STATE['stem']
    start, stop = tree['span'][stem]
    for name, weight in _STATE['weights'].items():
        head = tree['pred'][start:stop, tree['names'].index(name)].astype(np.float64)
        assert head.shape == labels[name].shape, (stem, name, head.shape, labels[name].shape)
        labels[name] = labels[name] + weight * (head - labels[name])
    _STATE['blended_shots'] = _STATE.get('blended_shots', 0) + 1
    return coordinates, labels


def _predict_many(task):
    pkg, files, v0_files, special_files, basis_file, finish, swap, tree_file, weights, subframe_file = task
    pd = _load_entrypoint(pkg)
    networks = [pd.Network.load(f) for f in v0_files]
    if special_files:
        _STATE['original'] = pd.predict_together
        _STATE['swap'] = swap
        _STATE['specialists'] = [pd.Network.load(f) for f in special_files]
        pd.predict_together = _swapped
    if tree_file or subframe_file:
        read_row = pd.read_row

        def _read_row(path):
            _STATE['stem'] = Path(path).stem
            return read_row(path)
        pd.read_row = _read_row
    if subframe_file:
        with np.load(subframe_file, allow_pickle=False) as z:
            stems, offsets = z['stems'].astype(str).tolist(), z['offsets']
            table = z['features']  # read ONCE: indexing the archive member inside a loop re-reads the whole array each time
            blocks = {s: table[int(offsets[i]):int(offsets[i + 1])] for i, s in enumerate(stems)}
        layout_features = pd.layout_features

        def _layout_features(row, shot, layouts):
            out = layout_features(row, shot, layouts)
            block = blocks[_STATE['stem']]
            for layout, matrix in out.items():
                assert matrix.shape[0] == block.shape[0], (_STATE['stem'], matrix.shape, block.shape)
                out[layout] = np.concatenate([matrix, block.astype(matrix.dtype)], axis=1)
            _STATE['extended_shots'] = _STATE.get('extended_shots', 0) + 1
            return out
        pd.layout_features = _layout_features
    if tree_file:
        with np.load(tree_file, allow_pickle=False) as z:
            stems, offsets = z['stems'].astype(str).tolist(), z['offsets']
            _STATE['tree'] = {'pred': z['pred'], 'names': z['targets'].astype(str).tolist(),
                              'span': {s: (int(offsets[i]), int(offsets[i + 1])) for i, s in enumerate(stems)}}
        _STATE['weights'] = dict(weights)
        _STATE['together'] = pd.predict_together
        pd.predict_together = _blended
    basis = pd.FluxBasis.load(basis_file).truncated(networks[0].coordinates)
    out = [pd.predict_shot(Path(path), networks, basis, finish, None) for path in files]
    if tree_file:
        assert _STATE.get('blended_shots', 0) == len(files), (_STATE.get('blended_shots'), len(files))
    if subframe_file:
        assert _STATE.get('extended_shots', 0) == len(files), (_STATE.get('extended_shots'), len(files))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--list', type=Path, action='append', required=True, help='sha256-per-line network list')
    ap.add_argument('--with-v25', action='store_true', help='also the 65 networks of the package manifest')
    ap.add_argument('--subframe', type=Path, default=None, help='npz of sub-frame features per frame: features, offsets, stems')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--specialists', type=Path, default=None)
    ap.add_argument('--swap', choices=('scalars', 'all'), default='scalars',
                    help="which outputs come from the specialists: the ten scalar heads, or every output they have")
    ap.add_argument('--map', action='append', default=[])
    ap.add_argument('--override', action='append', default=[], metavar='FLAG=VALUE',
                    help='replace one finisher flag value, e.g. contour-share=0.65')
    ap.add_argument('--scalar-shares', nargs='+', default=None, metavar='NAME=SHARE',
                    help='replace the whole --scalar-share list of the finisher')
    ap.add_argument('--tree-heads', type=Path, default=None,
                    help='npz of boosted-tree heads per frame: pred, targets, stems, offsets')
    ap.add_argument('--tree-weight', nargs='+', default=[], metavar='NAME=WEIGHT',
                    help='heads blended with the tree heads and the weight of the tree head')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    started = time.time()
    maps = [tuple(m.split('=', 1)) for m in a.map]
    pd = _load_entrypoint(a.pkg)
    manifest = json.loads((a.pkg / 'package_manifest.json').read_text())
    v25 = []
    for item in manifest['recipe']:
        p = a.pkg / item['path']
        assert digest(p) == item['sha256'], p
        v25.append(str(p))
    rest = [p for listing in a.list for p in verified(listing, maps)]
    assert len(v25) == 65 and len(rest) >= 1
    if not a.with_v25:
        v25 = []
    basis = a.pkg / manifest.get('basis_path', 'models/flux_basis.npz')
    assert digest(basis) == manifest['basis_sha256']
    special = verified(a.specialists, maps) if a.specialists else []
    assert len(special) in (0, 16)
    flags = list(FLAGS)
    for item in a.override:
        flag, value = item.split('=', 1)
        i = flags.index('--' + flag)
        flags[i + 1] = value
    if a.scalar_shares:
        i = flags.index('--scalar-share')
        assert i + 5 == len(flags)
        flags = flags[:i + 1] + list(a.scalar_shares)
    v0 = v25 + rest
    args = pd._parser().parse_args(['--data-root', str(a.data_root), '--network', *v0, '--basis', str(basis),
                                    '--out', str(a.out), '--split', 'diii_d_public_test', *flags])
    finish = pd._finish_from(args)
    files = [str(f) for f in pd.split_files(a.data_root, 'diii_d_public_test', None)]
    if a.limit:
        files = files[:a.limit]
    weights = [(n, float(w)) for n, w in (item.split('=', 1) for item in a.tree_weight)]
    assert bool(a.tree_heads) == bool(weights) and not (a.tree_heads and special)
    assert all(0.0 < w <= 1.0 for _, w in weights)
    if a.tree_heads:
        with np.load(a.tree_heads, allow_pickle=False) as z:
            assert set(n for n, _ in weights) <= set(z['targets'].astype(str).tolist())
            assert {Path(f).stem for f in files} <= set(z['stems'].astype(str).tolist())
            assert np.isfinite(z['pred']).all()
    a.out.mkdir(parents=True, exist_ok=False)
    bounds = np.linspace(0, len(files), min(a.workers, len(files)) + 1).astype(int)
    runs = [files[x:y] for x, y in itertools.pairwise(bounds)]
    archive = pd.PredictionArchive()
    with ProcessPoolExecutor(max_workers=len(runs), mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = [(str(a.pkg), r, v0, special, str(basis), finish, a.swap,
                  str(a.tree_heads) if a.tree_heads else None, weights,
                  str(a.subframe) if a.subframe else None) for r in runs]
        for predictions in pool.map(_predict_many, tasks):
            for maps_, q95, beta_n in predictions:
                archive.add(maps_, q95, beta_n)
    if archive.replaced:
        raise RuntimeError('Refuse silent nonfinite-to-zero replacement')
    archive.write(a.out / 'diii_d_public_test.npz')
    receipt = {'status': 'portable_deploy_build_tree_blend' if a.tree_heads else 'portable_deploy_build',
               'subframe': str(a.subframe) if a.subframe else None,
               'subframe_sha256': digest(a.subframe) if a.subframe else None, 'with_v25': a.with_v25,
               'network_count': len(v0), 'tree_heads': str(a.tree_heads) if a.tree_heads else None,
               'tree_heads_sha256': digest(a.tree_heads) if a.tree_heads else None, 'tree_weights': dict(weights), 'swap': bool(special), 'overrides': a.override, 'flags': flags,
               'swap_mode': a.swap if special else None,
               'scalar_labels_from_specialist': list(SCALAR_LABELS) if special else [], 'shots': archive.shots,
               'frames': archive.frames, 'limit': a.limit, 'workers': a.workers, 'networks': v0,
               'specialists': special, 'entrypoint_sha256': digest(pd.__file__), 'this_script_sha256': digest(__file__),
               'archive_sha256': digest(a.out / 'diii_d_public_test.npz'), 'seconds': time.time() - started,
               'MAST_targets_used': False}
    (a.out / 'deploy_receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('networks', 'specialists')}), flush=True)


if __name__ == '__main__':
    main()
