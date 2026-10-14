# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — K-fold refit of a v25 sub-ensemble for out-of-fold maps
"""Fit one fold's S16 sub-ensemble exactly as protocol_kfold_oof_maps_20260929 fixes it.

The same file runs on the workstation (ROCm) and on Kaggle (CUDA). For fold k it
1. derives each family's table from the complete lcfs table by selecting that family's
   label columns (verified equal to the family tables the v25 members were fitted on),
2. writes a reserved-stem file = reserved_for_evaluation stems + the fold's training stems,
   so the fold's shots take no part in fitting (training) nor in the calibration pass
   selection beyond what the original member saw,
3. runs fec_adapter.diii_d_accuracy.fit_on_accelerators once per (family, calibration) group
   with the members' original settings and seeds.

    python kfold_run.py --fold K --lcfs DIR --work DIR --out DIR [--members i,j,...]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol_kfold_oof_maps_20260929.json'


def fold_of(stem, folds):
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16) % folds


def family_tables(lcfs, work, family, labels):
    """Write the family's table directory (label subset of the lcfs table) once."""
    out = work / f'tables_{family}'
    done = out / 'DONE'
    if done.is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for part in ('training', 'calibration', 'validation'):
        with np.load(lcfs / f'table_{part}.npz', allow_pickle=False) as table:
            names = table['label_names'].astype(str).tolist()
            columns = [names.index(n) for n in labels]
            arrays = {k: table[k] for k in table.files if k not in ('labels', 'label_names')}
            arrays['labels'] = table['labels'][:, columns]
            arrays['label_names'] = np.asarray(labels)
        np.savez(out / f'table_{part}.npz', **arrays)
        del arrays
    (out / 'flux_basis.npz').write_bytes((lcfs / 'flux_basis.npz').read_bytes())
    done.write_text('ok')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--lcfs', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--members', default='')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['status'] == 'frozen_before_any_kfold_fit'
    k, folds = args.fold, protocol['folds']
    assert 0 <= k < folds
    splits = json.loads((HERE / 'splits.json').read_text())
    reserved = set(json.loads((HERE / 'reserved_for_evaluation.json').read_text())['stems'])
    held = sorted(s for s in splits['training'] if s not in reserved and fold_of(s, folds) == k)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    reserved_file = args.work / f'reserved_fold{k}.json'
    reserved_file.write_text(json.dumps({'stems': sorted(reserved | set(held)),
                                         'note': f'reserved_for_evaluation + training fold {k} of {folds}'}))
    members = protocol['members']
    if args.members:
        keep = {int(x) for x in args.members.split(',')}
        members = [m for m in members if m['recipe_index'] in keep]
    groups = {}
    for m in members:
        groups.setdefault((m['family'], m['train_on_calibration']), []).append(m)
    log = {'fold': k, 'held_training_shots': len(held), 'groups': []}
    for (family, tc), group in sorted(groups.items()):
        tables = family_tables(args.lcfs, args.work, family, protocol['family_labels'][family])
        variants = {f"{m['name']}_k{k}": {s: v for s, v in m['settings'].items() if s != 'threads'}
                    for m in group}
        vfile = args.work / f'variants_{family}_{"tc" if tc else "plain"}_fold{k}.json'
        vfile.write_text(json.dumps(variants, indent=1))
        cmd = [args.python, '-m', 'fec_adapter.diii_d_accuracy.fit_on_accelerators',
               '--tables', str(tables), '--out-root', str(args.out), '--reserved', str(reserved_file),
               '--variants', str(vfile)] + (['--train-on-calibration'] if tc else [])
        started = time.time()
        print(json.dumps({'run': cmd}), flush=True)
        code = subprocess.call(cmd, env=os.environ.copy())
        log['groups'].append({'family': family, 'train_on_calibration': tc, 'members': list(variants),
                              'exit': code, 'seconds': round(time.time() - started)})
        (args.out / f'kfold_log_fold{k}.json').write_text(json.dumps(log, indent=1))
        if code != 0:
            raise SystemExit(f'fit group {family} tc={tc} failed with {code}')
    print(json.dumps({'FOLD_DONE': k, 'log': log}), flush=True)


if __name__ == '__main__':
    main()
