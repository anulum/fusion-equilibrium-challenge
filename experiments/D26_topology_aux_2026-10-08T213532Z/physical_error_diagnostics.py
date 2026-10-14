# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — descriptive pressure and current-rate error attribution
"""Summarize frozen held-out predictions without fitting or selecting a model."""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    out = Path(sys.argv[1])
    manifest = json.loads((out / 'manifest.json').read_text())
    complete = json.loads((out / 'comparison.json').read_text())
    assert not manifest['smoke_only'] and complete['status'] == 'finished_oof_gate_complete'
    assert len(manifest['cases']) == 400
    blocks, stages = [], {}
    for case in manifest['cases']:
        base = out / 'diagnostics' / case['stem']
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as inputs:
            with np.load(base.with_name(base.name + '_errors.npz'), allow_pickle=False) as errors:
                names = inputs['stage_names'].astype(str).tolist()
                distances = inputs['stage_distance']
                failures = inputs['stage_failures']
                for i, name in enumerate(names):
                    valid = np.isfinite(distances[i])
                    stages.setdefault(name, []).append({
                        'stem': case['stem'], 'valid_frames': int(valid.sum()),
                        'fails': int(failures[i].sum()),
                        'mean_distance': float(distances[i, valid].mean()) if valid.any() else None})
                blocks.append(np.column_stack([
                    np.abs(inputs['ip_rate_A_s']), inputs['electron_pressure_mean_Pa'],
                    inputs['pressure_channel_coverage'], errors['flux_mse'],
                    distances[names.index('final')], errors['q95_squared_error'],
                    errors['betaN_squared_error']]))
    array = np.concatenate(blocks)
    metrics = ('flux_mse', 'lcfs_distance', 'q95_squared_error', 'betaN_squared_error')
    strata = {}
    for column, name in enumerate(('absolute_ip_rate_A_s', 'electron_pressure_channel_mean_Pa', 'pressure_coverage')):
        valid = np.isfinite(array[:, column])
        bounds = np.quantile(array[valid, column], [0, .25, .5, .75, 1])
        bins = np.searchsorted(bounds[1:-1], array[:, column], side='right')
        rows = []
        for bin_id in range(4):
            member = valid & (bins == bin_id)
            metric_values = {}
            for j, metric in enumerate(metrics, start=3):
                finite = member & np.isfinite(array[:, j])
                metric_values[metric] = float(array[finite, j].mean()) if finite.any() else None
            rows.append({'quartile': bin_id + 1, 'frames': int(member.sum()), **metric_values})
        strata[name] = {'input_quantile_bounds': bounds.tolist(), 'bins': rows, 'missing_frames': int((~valid).sum())}
    summary = {}
    for name, rows in stages.items():
        distances = [r['mean_distance'] for r in rows if r['mean_distance'] is not None]
        summary[name] = {'mean_of_shot_distances': float(np.mean(distances)) if distances else None,
                         'valid_frames': sum(r['valid_frames'] for r in rows),
                         'failures': sum(r['fails'] for r in rows)}
    value = {'status': 'descriptive_diagnostics_complete', 'shots': len(blocks), 'frames': len(array),
        'stages': summary, 'strata': strata,
        'limits': ['Correlations and input quartiles are descriptive; no causal instability diagnosis or model admission.',
                   'Electron-pressure mean is over available diagnostic channels, not a volume average or vessel pressure.',
                   'Current rate here is on actual EFIT timestamps and can underresolve fast input transients.',
                   'Baseline intermediates are cast float32 before the official LCFS extractor; canonical global sign verified.']}
    with (out / 'physical_diagnostics.json').open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'physical_diagnostics': str(out / 'physical_diagnostics.json'), 'shots': len(blocks), 'frames': len(array)}), flush=True)


if __name__ == '__main__':
    main()
