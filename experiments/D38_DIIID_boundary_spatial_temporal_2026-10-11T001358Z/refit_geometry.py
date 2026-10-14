# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — two-cohort-excluded geometry residual readouts
"""Refit the original true-map heads without altering trunks, flux or scalar rows."""
from __future__ import annotations

import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np

from campaign_common import HERE, ROOT, PARTS, digest, write_json
from training_data import COMPILE, initialize
from fec_adapter.diii_d_accuracy.network import Network

WORK = ROOT / 'work/second_agent-recovery-20260926/scratch'
sys.path.insert(0, str(WORK))
from geometry_readout_streaming import validity_groups, cpu_moments, gpu_moment_function, corrected_from_moments

STATE = {}


def prepare() -> list:
    """Require all fit receipts and select all 101 map readouts prospectively."""
    assert json.loads((HERE / 'candidate_terminal.json').read_text())['all125_successful']
    data = initialize()
    names = json.loads((HERE / 'supervision/receipt.json').read_text())['teacher_labels']
    assert len(names) == 73
    truths = {part: np.load(HERE / 'supervision' / (part + '_geometry.npy'), mmap_mode='r', allow_pickle=False) for part in PARTS}
    truth = np.concatenate([truths[part] for part in PARTS])
    assert len(truth) == sum(len(data['rows'][part]) for part in PARTS)
    all_models = []
    for root in ('fits', 'map_fits'):
        for report in sorted((HERE / root).glob('*/fit_report.json')):
            receipt = json.loads(report.read_text())
            assert receipt['status'] == 'fit_complete'
            if receipt.get('family') == 'joint':
                continue
            path = report.parent / 'network.npz'
            assert digest(path) == receipt['network_sha256']
            all_models.append({'path': str(path), 'sha256': receipt['network_sha256'], 'name': report.parent.name,
                               'parts': list(PARTS) if receipt.get('train_on_calibration', True) else ['training'],
                               'training_frames': receipt['training_frames'], 'family': receipt['family']})
    assert len(all_models) == 101
    STATE.update(data=data, names=names, truth=truth, truths=truths, groups={}, sources=all_models)
    for names_tuple in {tuple(name for name in names if name in Network.load(item['path']).label_names) for item in all_models}:
        selected = [names.index(n) for n in names_tuple]
        STATE['groups'][names_tuple] = validity_groups(truth[:, selected])
    return all_models


def fit_one(item: dict, device: jax.Device) -> dict:
    """Match original ridge residual mechanics using only eligible source rows."""
    out = HERE / 'readouts' / item['name']
    out.mkdir(exist_ok=True)
    if (out / 'fit_report.json').exists():
        receipt = json.loads((out / 'fit_report.json').read_text())
        assert digest(out / 'network.npz') == receipt['network_sha256']
        return receipt
    started = time.monotonic()
    path = Path(item['path'])
    assert digest(path) == item['sha256']
    old = Network.load(path)
    names = tuple(name for name in STATE['names'] if name in old.label_names)
    selected = [STATE['names'].index(n) for n in names]
    groups = STATE['groups'][names]
    function = gpu_moment_function(old, names, groups)
    data = STATE['data']
    ids = data['rows']['training'][:128]
    features = np.asarray(data['base']['training']['features'][ids], dtype=float)
    wanted = np.asarray(STATE['truths']['training'][:128, selected], dtype=float)
    with COMPILE:
        gpu, bad, masks_bad = function(features, wanted)
        gpu = jax.tree.map(np.asarray, gpu)
    assert int(bad) == 0 and int(masks_bad) == 0
    cpu = cpu_moments(old, features, wanted, names, groups)
    parity = 0.
    for original, accelerated in zip(cpu, gpu, strict=True):
        assert int(original[0]) == int(accelerated[0])
        for a, b in zip(original[1:], accelerated[1:], strict=True):
            difference = float(np.max(np.abs(a - b) / (1 + np.abs(a))))
            parity = max(parity, difference)
            assert difference < 1e-9, (item['name'], difference)
    total = None
    used = 0
    for part in item['parts']:
        rows = data['rows'][part]
        assert len(rows) == len(STATE['truths'][part])
        for start in range(0, len(rows), 8192):
            chosen = rows[start:start + 8192]
            features = np.asarray(data['base'][part]['features'][chosen], dtype=float)
            wanted = np.asarray(STATE['truths'][part][start:start + len(chosen), selected], dtype=float)
            moments, bad, masks_bad = function(features, wanted)
            assert int(bad) == 0 and int(masks_bad) == 0
            moments = jax.tree.map(np.asarray, moments)
            total = moments if total is None else tuple(tuple(a + b for a, b in zip(x, y, strict=True)) for x, y in zip(total, moments, strict=True))
            used += len(chosen)
    assert used == item['training_frames']
    new, groups_report = corrected_from_moments(old, total, names, groups, ridge=.001)
    changed = {old.coordinates + old.label_names.index(n) for n in names}
    keep = [i for i in range(len(old.output_mean)) if i not in changed]
    for a, b in zip(old.weights[:-1] + old.biases[:-1], new.weights[:-1] + new.biases[:-1], strict=True):
        assert np.array_equal(a, b)
    for a, b in [(old.weights[-1][keep], new.weights[-1][keep]), (old.biases[-1][keep], new.biases[-1][keep]),
                 (old.linear, new.linear), (old.feature_mean, new.feature_mean), (old.feature_scale, new.feature_scale),
                 (old.output_mean, new.output_mean), (old.output_scale, new.output_scale)]:
        assert np.array_equal(a, b)
    new.save(out / 'network.npz')
    if (path.parent / 'decoder.npy').exists():
        assert digest(path.parent / 'decoder.npy') == json.loads((path.parent / 'fit_report.json').read_text())['decoder_sha256']
    receipt = {'name': item['name'], 'family': item['family'], 'status': 'readout_complete',
               'input': str(path), 'input_sha256': item['sha256'], 'network_sha256': digest(out / 'network.npz'),
               'fit_parts': item['parts'], 'rows': used, 'development_and_confirmation_excluded': True,
               'only_named_geometry_final_rows_changed': True, 'spatial_latent_rows_unchanged': True,
               'CPU_GPU_moment_relative_max': parity, 'groups': groups_report,
               'seconds': time.monotonic() - started, 'device': str(device)}
    write_json(out / 'fit_report.json', receipt)
    print(json.dumps({'readout_complete': item['name'], 'seconds': receipt['seconds'], 'device': str(device)}), flush=True)
    return receipt


def worker(assignment: tuple) -> list:
    """Read bounded source blocks and retain every readout's terminal outcome."""
    card, models = assignment
    device = jax.devices()[card]
    receipts = []
    with jax.default_device(device):
        for item in models:
            failure = HERE / 'readouts' / (item['name'] + '_failure.json')
            try:
                receipts.append(fit_one(item, device))
            except Exception as error:
                receipt = {'name': item['name'], 'status': 'readout_failed_not_admitted',
                           'type': type(error).__name__, 'error': str(error)}
                write_json(failure, receipt)
                receipts.append(receipt)
            gc.collect()
    return receipts


def main() -> None:
    """Finish all current-campaign geometry readouts without held-out scoring."""
    models = prepare()
    assert len(jax.devices()) == 3
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = [r for batch in pool.map(worker, [(i, models[i::3]) for i in range(3)]) for r in batch]
    write_json(HERE / 'readouts/receipt.json', {'status': 'all101_geometry_readouts_terminal', 'tasks': receipts,
               'successful': sum(r['status'] == 'readout_complete' for r in receipts), 'development_metrics_computed': False})


if __name__ == '__main__':
    main()
