# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fold-excluded original-geometry readout refit (K-fold, protocol 20260929)
"""Refit the geometry readout rows of one fold's bank members, exactly as the v25 bank did
(original-map geometry targets, ridge 0.001, exact per-column validity groups), but with the
fold's training shots and the reserved stems left out of the moments. CPU, streamed in blocks.

    python kfold_readout.py --fold K --nets DIR_OF_FOLD_FITS --out DIR
"""
import argparse
import os
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path('$LAB/'
         'work/second_agent-recovery-20260926')
sys.path.insert(0, os.environ.get('FEC_SCRATCH', str(W / 'scratch')))
from fec_adapter.diii_d_accuracy.network import Network  # noqa: E402
from geometry_readout_streaming import validity_groups, cpu_moments, corrected_from_moments  # noqa: E402

TABLES = Path(os.environ.get('FEC_TABLES', '$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge'))
TARGETS_ROOT = Path(os.environ.get('FEC_TARGETS_ROOT', str(W / 'results/original_map_geometry_targets_20260927')))
TARGETS = TARGETS_ROOT / 'complete_receipt.json'
ORIGINAL_TARGETS_ROOT = str(W / 'results/original_map_geometry_targets_20260927')
RIDGE, BLOCK = 0.001, 16384


def fold_of(stem, folds):
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16) % folds


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--nets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol_kfold_oof_maps_20260929.json').read_text())
    assert protocol['status'] == 'frozen_before_any_kfold_fit'
    k = args.fold
    reserved = set(json.loads((HERE / 'reserved_for_evaluation.json').read_text())['stems'])
    complete = json.loads(TARGETS.read_text())
    records = {r['stem']: r for r in complete['records']}
    all_names = complete['label_names']
    tables = {}
    for part in ('training', 'calibration'):
        with np.load(TABLES / f'table_{part}.npz', allow_pickle=False) as table:
            stems = table['stems'].astype(str)
            shot_of_row = table['shot_of_row']
            features = table['features']
        counts = np.bincount(shot_of_row, minlength=len(stems))
        bounds = np.concatenate(([0], np.cumsum(counts)))
        truth = np.full((len(features), len(all_names)), np.nan, dtype=np.float32)
        rows = []
        for i, stem in enumerate(stems):
            if stem in reserved or (part == 'training' and fold_of(stem, protocol['folds']) == k):
                continue
            b, e = int(bounds[i]), int(bounds[i + 1])
            r = records[stem]
            assert r['part'] == part and r['frames'] == e - b
            with np.load(r['output_path'].replace(ORIGINAL_TARGETS_ROOT, str(TARGETS_ROOT)), allow_pickle=False) as target:
                assert target['stem'].item() == stem
                truth[b:e] = target['labels']
            rows.append(np.arange(b, e))
        tables[part] = (features, truth, np.concatenate(rows))
        print(json.dumps({'part': part, 'rows': int(len(tables[part][2]))}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    done = []
    for m in protocol['members']:
        if not m['readout_bank_refit']:
            continue
        started = time.time()
        path = args.nets / f"{m['name']}_k{k}" / 'network.npz'
        old = Network.load(path)
        # As the v25 bank: every original-map geometry target the network has an output for.
        names = [n for n in all_names if n in old.label_names]
        assert len(names) == {'h': 5, 'hx': 9, 'hc': 73}[m['family']], (m['name'], len(names))
        selected = [all_names.index(n) for n in names]
        parts = ('training', 'calibration') if m['train_on_calibration'] else ('training',)
        stacked = np.concatenate([tables[p][1][tables[p][2]][:, selected] for p in parts])
        groups = validity_groups(stacked)
        del stacked
        total = None
        for p in parts:
            features, truth, rows = tables[p]
            for off in range(0, len(rows), BLOCK):
                block = rows[off:off + BLOCK]
                mom = cpu_moments(old, features[block, :old.feature_mean.size].astype(np.float64),
                                  truth[block][:, selected], names, groups)
                total = mom if total is None else tuple(tuple(a + b for a, b in zip(x, y)) for x, y in zip(total, mom))
        updated, reports = corrected_from_moments(old, total, names, groups, RIDGE)
        d = args.out / f"{m['name']}_k{k}"
        d.mkdir(parents=True, exist_ok=True)
        updated.save(d / 'network.npz')
        report = {'member': m['name'], 'fold': k, 'geometry_labels': names, 'fit_parts': list(parts),
                  'fold_training_shots_excluded': True, 'reserved_excluded': True, 'ridge': RIDGE,
                  'group_reports': reports, 'initial_sha256': digest(path),
                  'network_sha256': digest(d / 'network.npz'), 'seconds': time.time() - started}
        (d / 'fit_report.json').write_text(json.dumps(report, indent=1))
        done.append(report['member'])
        print(json.dumps({'refit': m['name'], 'labels': len(names), 'seconds': round(report['seconds'])}), flush=True)
    (args.out / f'readout_receipt_fold{k}.json').write_text(json.dumps({'fold': k, 'members': done,
                                                                         'target_receipt_sha256': digest(TARGETS)}, indent=1))


if __name__ == '__main__':
    main()
