# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — pressure calibration on supported inputs
"""Fit the declared five models using fold-permitted reliable teacher rows."""

from __future__ import annotations

import json

import numpy as np

from reliability import D27, HERE, PARTS, digest, metadata, write_json
from campaign_common import base_maps, extra_maps, fold_of, reserved


def main() -> None:
    """Reuse frozen input normalization and exclude unsupported pressure rows."""
    assert json.loads((HERE / 'inputs/receipt.json').read_text())['status'] == 'all_input_support_prepared'
    out = HERE / 'factors'
    out.mkdir(exist_ok=True)
    receipts = []
    base, extra = base_maps(), extra_maps()
    metas = {part: metadata(part) for part in PARTS}
    supported = {part: np.load(HERE / 'inputs' / (part + '.npy'), mmap_mode='r') for part in PARTS}
    teacher = {part: np.load(D27 / 'repair_axis_flux_v1/beta_targets' / (part + '.npy'), mmap_mode='r') for part in PARTS}
    held = reserved()
    for fold in range(5):
        output, receipt = out / f'factor_fold{fold}.npz', out / f'factor_fold{fold}.json'
        if receipt.exists():
            record = json.loads(receipt.read_text())
            assert digest(output) == record['sha256']
            receipts.append(record)
            continue
        with np.load(D27 / f'pressure_factors/factor_fold{fold}.npz', allow_pickle=False) as old:
            mean, scale = old['feature_mean'].copy(), old['feature_scale'].copy()
        xx, xy, shots, rows = np.zeros((9, 9)), np.zeros(9), 0, 0
        for part in ('training', 'calibration'):
            meta = metas[part]
            label = meta['label_names'].index('beta_n')
            for i, stem in enumerate(meta['stems']):
                if stem in held or (part == 'training' and fold_of(stem) == fold):
                    continue
                a, b = meta['bounds'][i:i+2]
                electron, truth = teacher[part][a:b].astype(float), base[part]['labels'][a:b, label].astype(float)
                good = supported[part][a:b] & np.isfinite(electron) & np.isfinite(truth) & (electron > 0) & (truth > 0)
                if not good.any():
                    continue
                values = extra[part]['factor_features'][a:b][good].astype(float)
                z = np.column_stack([np.ones(good.sum()), np.clip((values - mean) / scale, -8, 8)])
                y = np.log(truth[good] / electron[good])
                xx += z.T @ z / len(z)
                xy += z.T @ y / len(z)
                shots += 1
                rows += len(z)
        assert shots > 0 and rows > 0
        penalty = np.eye(9) * .01
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(xx / shots + penalty, xy / shots)
        assert np.isfinite(coefficients).all()
        temporary = out / f'factor_fold{fold}.partial.npz'
        np.savez(temporary, coefficients=coefficients, feature_mean=mean, feature_scale=scale,
                 factor_bounds=np.array([.5, 8.]), fold=np.asarray(fold))
        if output.exists():
            assert digest(output) == digest(temporary)
            temporary.unlink()
        else:
            temporary.replace(output)
        record = {'fold': fold, 'status': 'factor_fit_complete', 'shots': shots, 'rows': rows,
                  'sha256': digest(output), 'reserved_and_heldfold_excluded': True,
                  'same_D27_input_normalization': True, 'comparative_metrics_computed': False}
        write_json(receipt, record)
        receipts.append(record)
        print(json.dumps({'supported_factor_fit_complete': fold}), flush=True)
    if not (out / 'receipt.json').exists():
        write_json(out / 'receipt.json', {'status': 'all_five_supported_factors_complete', 'models': receipts})


if __name__ == '__main__':
    main()
