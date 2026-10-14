# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete scalar-task comparisons
"""Compare every declared scalar arm on the same full shot-disjoint population."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from campaign_common import DATA, D26, D27, HERE, base_maps, digest, feature_block, fold_of, metadata, reserved, write_json
from fec_adapter.diii_d_accuracy.network import Network

NAMES = ('q95', 'beta_n')
ARMS = ('stronger_p45', 'matched_control', 'joint', 'separate')
BASE = None


def baseline_models(fold: int) -> dict:
    """Load only fold-excluded original models and complete new scalar pairs."""
    receipt = json.loads((D26 / 'geometry_readout_receipt.json').read_text())
    prior = {}
    for role in ('hx_shared', 'hc_p'):
        rows = [r for r in receipt['results'] if r['fold'] == fold and r['role'] == role]
        assert len(rows) == 2
        paths = [D26 / f'readouts_f{fold}' / Path(r['input']).parent.name / 'network.npz' for r in rows]
        for path, row in zip(paths, rows, strict=True):
            assert digest(path) == row['network_sha256']
        prior[role] = [Network.load(path) for path in paths]
    control = [D27 / f'readouts/control_seed{seed}_k{fold}/network.npz' for seed in (200, 201)]
    controls = json.loads((D27 / 'readouts/receipt.json').read_text())['tasks']
    for path in control:
        record = next(r for r in controls if r['name'] == path.parent.name)
        assert digest(path) == record['network_sha256']
    prior['control'] = [Network.load(path) for path in control]
    return prior


def models(fold: int) -> dict:
    """Load the qualified control and both complete scalar-only variants."""
    prior = baseline_models(fold)
    for arm in ('joint', 'q95', 'beta_n'):
        paths = [HERE / f'fits/{arm}_seed{seed}_k{fold}/network.npz' for seed in (200, 201)]
        for path in paths:
            record = json.loads(path.with_name('fit_report.json').read_text())
            assert record['status'] == 'fixed_final_epoch_fit_complete' and record['epochs_completed'] == 60
            assert digest(path) == record['network_sha256']
        prior[arm] = [Network.load(path) for path in paths]
    return prior


def prediction(network: Network, features: np.ndarray, names: tuple) -> np.ndarray:
    """Use the public NumPy predictor in exported double precision."""
    _, labels = network.predict(features[:, :network.feature_mean.size])
    return np.column_stack([labels[n] for n in names])


def fold_cases(task: tuple) -> list:
    """Preserve all scalar predictions and sufficient statistics before comparison."""
    fold, cases = task
    networks = models(fold)
    records = []
    for case in cases:
        path = HERE / 'evaluation/shots' / (case['stem'] + '.npz')
        receipt_path = path.with_suffix('.json')
        if receipt_path.exists():
            record = json.loads(receipt_path.read_text())
            assert digest(path) == record['sha256']
            records.append(record)
            continue
        a, b = case['span']
        features = feature_block(BASE['training'], {}, np.arange(a, b), 'control')
        hx = sum(prediction(n, features, NAMES) for n in networks['hx_shared'])
        values = {
            'stronger_p45': (hx + sum(prediction(n, features, NAMES) for n in networks['hc_p'])) / 4,
            'matched_control': (hx + sum(prediction(n, features, NAMES) for n in networks['control'])) / 4,
            'joint': (hx + sum(prediction(n, features, NAMES) for n in networks['joint'])) / 4,
        }
        separate = np.column_stack([
            sum(prediction(n, features, (name,))[:, 0] for n in networks[name]) for name in NAMES])
        values['separate'] = (hx + separate) / 4
        table = pq.read_table(DATA / (case['stem'] + '.parquet'),
                              columns=['efit_q95', 'efit_beta_n'], use_threads=False)
        truth = np.column_stack([np.asarray(table[n][0].as_py(), dtype=np.float64)
                                 for n in ('efit_q95', 'efit_beta_n')])
        assert truth.shape == (b - a, 2)
        valid = np.isfinite(truth)
        count = valid.sum(axis=0)
        clean = np.where(valid, truth, 0.)
        residual = []
        payload = {'truth': truth}
        for arm in ARMS:
            pred = values[arm].astype(np.float32)
            assert pred.shape == truth.shape and np.isfinite(pred).all()
            payload[arm] = pred
            residual.append(np.where(valid, truth - pred.astype(np.float64), 0.)**2)
        squared = np.asarray(residual)
        temporary = path.with_name(case['stem'] + '.partial.npz')
        np.savez(temporary, **payload)
        temporary.replace(path)
        record = {**case, 'frames': b - a, 'count': count.tolist(),
                  'sum': clean.sum(axis=0).tolist(), 'sumsq': (clean**2).sum(axis=0).tolist(),
                  'sse': squared.sum(axis=1).tolist(),
                  'last3_sse': squared[:, -3:, :].sum(axis=1).tolist(),
                  'sha256': digest(path)}
        write_json(receipt_path, record)
        records.append(record)
        print(json.dumps({'scalar_prediction_case_complete': case['stem'], 'fold': fold}), flush=True)
    return records


def pooled(records: list, weights: np.ndarray) -> dict:
    """Use the competition's finite-truth masks and pooled two-scalar R2 formula."""
    sums = np.asarray([r['sum'] for r in records])
    sumsquares = np.asarray([r['sumsq'] for r in records])
    counts = np.asarray([r['count'] for r in records])
    sse = np.asarray([r['sse'] for r in records])
    denominator = weights @ sumsquares - (weights @ sums)**2 / (weights @ counts)
    assert (denominator > 0).all()
    r2 = 1. - np.einsum('n,nat->at', weights, sse) / denominator
    return {arm: {'r2_q95': float(r2[i, 0]), 'r2_betaN': float(r2[i, 1]),
                  'r2_qb': float(np.mean(r2[i])), 'weighted_scalar_term': .15 * max(0., float(np.mean(r2[i])))}
            for i, arm in enumerate(ARMS)}


def main() -> None:
    """Finish all predictions before reporting full-population comparisons."""
    global BASE
    receipt = json.loads((HERE / 'fits/receipt.json').read_text())
    assert receipt['status'] == 'all_30_neural_tasks_terminal' and len(receipt['tasks']) == 30
    assert receipt['successful'] == 30, 'incomplete arm; no seed omission or partial cohort comparison'
    BASE = base_maps()
    meta = metadata('training')
    skipped = reserved()
    cases = [{'stem': stem, 'fold': fold_of(stem), 'span': [meta['bounds'][i], meta['bounds'][i+1]]}
             for i, stem in enumerate(meta['stems']) if stem not in skipped]
    assert len(cases) == 5936
    (HERE / 'evaluation/shots').mkdir(exist_ok=True)
    grouped = [(fold, [c for c in cases if c['fold'] == fold]) for fold in range(5)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [r for batch in pool.map(fold_cases, grouped) for r in batch]
    records.sort(key=lambda r: r['stem'])
    assert len(records) == 5936 and sum(r['frames'] for r in records) == 1317704
    write_json(HERE / 'evaluation/statistics.json', records)
    results = pooled(records, np.ones(len(records)))
    rng = np.random.default_rng(995)
    boot = {arm: {control: [] for control in ARMS[:2]} for arm in ARMS[2:]}
    for _ in range(2000):
        weights = np.bincount(rng.integers(0, len(records), len(records)), minlength=len(records))
        sampled = pooled(records, weights)
        for arm in ARMS[2:]:
            for control in ARMS[:2]:
                boot[arm][control].append(sampled[arm]['weighted_scalar_term'] - sampled[control]['weighted_scalar_term'])
    comparisons = {}
    for arm in ARMS[2:]:
        controls = {}
        for control in ARMS[:2]:
            delta = results[arm]['weighted_scalar_term'] - results[control]['weighted_scalar_term']
            samples = np.asarray(boot[arm][control])
            controls[control] = {'delta_S_if_map_identical': delta,
                                'ci95': np.percentile(samples, [2.5, 97.5]).tolist(),
                                'P_positive': float(np.mean(samples > 0)),
                                'passes': bool(delta >= 3e-5 and np.mean(samples > 0) >= .99)}
        comparisons[arm] = {'controls': controls, 'hypothesis_screen_passes': all(c['passes'] for c in controls.values()),
                            'production_admission': False}
    known = json.loads((D27 / 'evaluation/manifest.json').read_text())['cases']
    subset_stems = {c['stem'] for c in known}
    subset = [r for r in records if r['stem'] in subset_stems]
    assert len(subset) == 400
    subset_results = pooled(subset, np.ones(400))
    write_json(HERE / 'evaluation/summary.json', {
        'status': 'all_scalar_comparisons_complete', 'shots': len(records), 'frames': 1317704,
        'results': results, 'comparisons': comparisons, 'diagnostic_reused400_only': subset_results,
        'map_outputs_changed': False, 'score_evidence': 'Exact scalar contribution using raw EFIT labels; full-map production equivalence not established.',
        'full_H69_out_of_sample_validation_required': True,
        'separate_cohort_confirmation_required': True, 'public_upload_or_submission': False})


if __name__ == '__main__':
    main()
