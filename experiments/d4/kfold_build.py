# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — out-of-fold finished maps with the exact v25 finisher (K-fold 20260929)
"""Finish one fold's held-out training shots with that fold's S16 networks.

The finisher is the one v25 was built with: the fixed-axis-volume builder module
(second_agent-recovery scratch/build_fixed_axis_volume_DIII_D_56cb.py) and the Finish values of the
v25 full357 build_report, gains none. Shots: the fold's held-out training shots ordered by
sha256(stem), the first SHOTS_PER_FOLD of them. Networks: the fold's direct members as fitted,
bank members after kfold_readout.py; members dropped under amendment 1 are left out.

    python kfold_build.py --fold K --fits DIR --readouts DIR --out DIR [--workers 6]
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('$LAB/'
         'work/second_agent-recovery-20260926')
sys.path.insert(0, str(W / 'scratch'))
import build_fixed_axis_volume_DIII_D_56cb as vbuild  # noqa: E402

DATA = Path('$FEC_DATA/data/diii_d_train')
BASIS = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/flux_basis.npz')
V25_REPORT = W / 'results/fixed_axis_volume_full357_56cb/build_report.json'
SHOTS_PER_FOLD = 80


def fold_of(stem, folds):
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16) % folds


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def v25_finish():
    report = json.loads(V25_REPORT.read_text())
    values = {}
    for field in dataclasses.fields(vbuild.Finish):
        value = report[field.name]
        if field.name == 'scalar_shares':
            value = tuple((s.split('=')[0], float(s.split('=')[1])) for s in value)
        elif isinstance(value, list):
            value = tuple(value)
        values[field.name] = value
    return vbuild.Finish(**values), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--readouts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    started = time.time()
    protocol = json.loads((HERE / 'protocol_kfold_oof_maps_20260929.json').read_text())
    ledger = json.loads((HERE / 'kfold_fit_ledger.json').read_text())
    k = args.fold
    splits = json.loads((HERE / 'splits.json').read_text())
    reserved = set(json.loads((HERE / 'reserved_for_evaluation.json').read_text())['stems'])
    held = sorted((s for s in splits['training'] if s not in reserved and fold_of(s, protocol['folds']) == k),
                  key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:SHOTS_PER_FOLD]
    networks = []
    for m in protocol['members']:
        entry = ledger[str(k)][m['name']]
        if entry['status'] == 'dropped':
            continue
        root = args.readouts if m['readout_bank_refit'] else args.fits
        networks.append(root / entry['fitted_name'] / 'network.npz')
    finish, values = v25_finish()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [DATA / f'{s}.parquet' for s in held]
    archive = vbuild.predicted_archive(paths, networks, BASIS, args.workers, finish)
    archive.write(args.out / 'diii_d_train.npz')
    receipt = {'status': 'oof_maps_built', 'fold': k, 'shots': archive.shots, 'frames': archive.frames,
               'replaced': archive.replaced, 'stems': held, 'networks': [str(p) for p in networks],
               'network_sha256': [digest(p) for p in networks], 'finish': {k2: str(v) for k2, v in values.items()},
               'builder': str(Path(vbuild.__file__)), 'builder_sha256': digest(vbuild.__file__),
               'archive_sha256': digest(args.out / 'diii_d_train.npz'), 'seconds': time.time() - started,
               'reserved_overlap': 0, 'validation_targets_used': False, 'MAST_targets_used': False}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k2: v for k2, v in receipt.items() if k2 not in ('stems', 'networks', 'network_sha256')}), flush=True)


if __name__ == '__main__':
    main()
