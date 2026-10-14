# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — paired geometry readouts after all physics fits terminate
"""Refit each successful model on fold-permitted geometry targets only."""

import gc
import json
import sys
from pathlib import Path

import jax
import numpy as np

from campaign_common import (HERE, PARTS, ROOT, SAS, base_maps, digest, extra_maps,
                             feature_block, metadata, row_ids, write_json)
from fec_adapter.diii_d_accuracy.network import Network

WORKSPACE = ROOT / 'work/second_agent-recovery-20260926'
sys.path.insert(0, str(WORKSPACE / 'scratch'))
from geometry_readout_streaming import (corrected_from_moments, cpu_moments,
                                       gpu_moment_function, validity_groups)


def main():
    neural = json.loads((HERE / 'fits/receipt.json').read_text())
    factors = json.loads((HERE / 'pressure_factors/receipt.json').read_text())
    assert neural['status'] == 'all_50_neural_tasks_terminal' and len(neural['tasks']) == 50
    assert factors['status'] == 'all_five_factor_fits_complete'
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform in ('gpu', 'rocm')
    target_root = SAS / 'workstation/second_agent-recovery-20260926_workspace_20261008/results/original_map_geometry_targets_20260927'
    original_root = WORKSPACE / 'results/original_map_geometry_targets_20260927'
    complete = json.loads((target_root / 'complete_receipt.json').read_text())
    names = complete['label_names']
    records = {r['stem']: r for r in complete['records']}
    base, extra = base_maps(), extra_maps()
    metas = {part: metadata(part) for part in PARTS}
    start_cal = metas['training']['rows']
    total_rows = start_cal + metas['calibration']['rows']
    truth = np.full((total_rows, len(names)), np.nan, dtype=np.float32)
    for part in ('training', 'calibration'):
        offset = 0 if part == 'training' else start_cal
        for i, stem in enumerate(metas[part]['stems']):
            if stem not in records:
                continue
            record = records[stem]
            assert record['part'] == part and record['frames'] == metas[part]['counts'][i]
            path = target_root / Path(record['output_path']).relative_to(original_root)
            assert digest(path) == record['output_sha256']
            a, b = metas[part]['bounds'][i:i + 2]
            with np.load(path, allow_pickle=False) as stored:
                truth[offset + a:offset + b] = stored['labels']
    out = HERE / 'readouts'
    out.mkdir(exist_ok=False)
    receipts = []
    groups = validity_groups(truth)
    with jax.default_device(devices[0]):
        for task in neural['tasks']:
            if task['status'] != 'fixed_final_epoch_fit_complete':
                receipts.append({'name': task['name'], 'status': 'upstream_fit_failed'})
                continue
            name = task['name']
            arm, fold = name.split('_seed')[0], int(name.rsplit('_k', 1)[1])
            path = HERE / 'fits' / name / 'network.npz'
            assert digest(path) == task['network_sha256']
            old = Network.load(path)
            assert set(names) <= set(old.label_names)
            rows = {p: row_ids(metas[p], fold) for p in ('training', 'calibration')}
            assert sum(len(x) for x in rows.values()) == task['training_frames']
            function = gpu_moment_function(old, names, groups)
            sample = rows['training'][:128]
            x = feature_block(base['training'], extra['training'], sample, arm)
            y = truth[sample]
            gpu, bad, mask_bad = function(x, y)
            assert int(bad) == 0 and int(mask_bad) == 0
            gpu = jax.tree.map(np.asarray, gpu)
            cpu = cpu_moments(old, x, y, names, groups)
            parity = 0.
            for a, b in zip(cpu, gpu, strict=True):
                assert int(a[0]) == int(b[0])
                for c, d in zip(a[1:], b[1:], strict=True):
                    error = float(np.max(np.abs(c - d) / (1 + np.abs(c))))
                    parity = max(parity, error)
                    assert error < 1e-9
            total = None
            for part, ids in rows.items():
                offset = 0 if part == 'training' else start_cal
                for i in range(0, len(ids), 8192):
                    take = ids[i:i + 8192]
                    x = feature_block(base[part], extra[part], take, arm)
                    mm, bad, mask_bad = function(x, truth[offset + take])
                    assert int(bad) == 0 and int(mask_bad) == 0
                    mm = jax.tree.map(np.asarray, mm)
                    total = mm if total is None else tuple(tuple(a + b for a, b in zip(c, d, strict=True))
                                                           for c, d in zip(total, mm, strict=True))
            new, report = corrected_from_moments(old, total, names, groups, ridge=.001)
            changed = {old.coordinates + old.label_names.index(label) for label in names}
            kept = [i for i in range(len(old.output_mean)) if i not in changed]
            assert np.array_equal(old.weights[-1][kept], new.weights[-1][kept])
            assert np.array_equal(old.biases[-1][kept], new.biases[-1][kept])
            assert np.array_equal(old.linear, new.linear)
            directory = out / name
            directory.mkdir()
            new.save(directory / 'network.npz')
            receipt = {'name': name, 'fold': fold, 'arm': arm, 'status': 'geometry_readout_complete',
                       'network_sha256': digest(directory / 'network.npz'), 'CPU_GPU_relative_max': parity,
                       'training_rows': sum(len(r) for r in rows.values()),
                       'heldfold_reserved_excluded': True, 'groups': report}
            write_json(directory / 'receipt.json', receipt)
            receipts.append(receipt)
            print(json.dumps({'readout_complete': name, 'device': str(devices[0])}), flush=True)
            del old, new, total, function
            gc.collect()
    assert len(receipts) == 50
    write_json(out / 'receipt.json', {'status': 'all_readout_tasks_terminal', 'tasks': receipts,
                                    'comparative_metrics_computed': False})


if __name__ == '__main__':
    main()
