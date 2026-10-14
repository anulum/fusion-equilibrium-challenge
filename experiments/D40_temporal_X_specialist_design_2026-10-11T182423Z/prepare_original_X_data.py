# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — unused357 original6326 X data contract
"""Bind the original eligible normalizers and exact row alignment without fitting."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import metadata, digest


def main():
    out = HERE / 'original6326_data'
    out.mkdir(exist_ok=False)
    cohort = json.loads((D38 / 'cohort.json').read_text())
    plans, totals, sources = {}, {}, {}
    for part in ('training', 'calibration'):
        alignment_path = D38 / 'temporal' / (part+'_alignment.json')
        alignment = json.loads(alignment_path.read_text())
        assert alignment['stems'] == sorted(cohort[part])
        meta = metadata(part)
        bounds = {s: meta['bounds'][i:i+2] for i, s in enumerate(meta['stems'])}
        plans[part] = [{'stem': s, 'teacher_window_span': alignment['offsets'][i:i+2],
                        'feature_span': bounds[s]} for i, s in enumerate(alignment['stems'])]
        assert all(s['teacher_window_span'][1]-s['teacher_window_span'][0] ==
                   s['feature_span'][1]-s['feature_span'][0] for s in plans[part])
        totals[part] = {'shots': len(plans[part]), 'frames': alignment['offsets'][-1]}
        sources[str(alignment_path)] = digest(alignment_path)
    static_path, temporal_path, target_path = D38 / 'scaling_tc.npz', D38 / 'temporal/normalization.npz', HERE / 'eligible_X_target_scaling_preview.npz'
    preview = json.loads((HERE / 'eligible_X_target_contract_preview.json').read_text())
    assert digest(target_path) == preview['scaling_sha256']
    with np.load(static_path, allow_pickle=False) as static, np.load(temporal_path, allow_pickle=False) as temporal, np.load(target_path, allow_pickle=False) as target:
        assert static['feature_mean'].shape == (582,) and temporal['mean'].shape == (21,)
        assert target['labels'].astype(str).tolist() == preview['label_order']
        np.savez(out / 'normalization.npz', feature_mean=static['feature_mean'], feature_scale=static['feature_scale'],
                 output_mean=target['mean'], output_scale=target['scale'],
                 temporal_mean=temporal['mean'], temporal_scale=temporal['scale'])
    for path in (static_path, temporal_path, target_path, D38 / 'cohort.json', Path(__file__)):
        sources[str(path)] = digest(path)
    with (out / 'fit_row_plan.json').open('x') as stream:
        json.dump({'parts': plans, 'totals': totals, 'excluded_stems': []}, stream, indent=2)
    assert totals['training'] == {'shots': 5954, 'frames': 1321417}
    assert totals['calibration'] == {'shots': 372, 'frames': 82797}
    with (out / 'receipt.json').open('x') as stream:
        json.dump({'status': 'original6326_X_data_contract_ready_only_if_D39_never_opens357',
                   'totals': totals, 'fit_frames': 1404214, 'sources_sha256': sources,
                   'normalization_sha256': digest(out / 'normalization.npz'),
                   'row_plan_sha256': digest(out / 'fit_row_plan.json'),
                   'no_true_development_or_confirmation_targets_read': True,
                   'no_optimizer_updates_no_activation': True}, stream, indent=2)
    print(json.dumps({'original6326_data_contract_ready': True}), flush=True)


if __name__ == '__main__':
    main()
