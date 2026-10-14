# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — whole-shot X training corpus GPU transfer qualification
"""Check real eligible feature, temporal and teacher row identities on the GPU."""
import json
from pathlib import Path
import time

import numpy as np
import train_X_specialists as trainer
from campaign_common import base_maps
from window_cache import mapped_array
import jax

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_transfer_qualification'


def main():
    started = time.monotonic()
    sources = json.loads((OUT / 'sources.json').read_text())
    for path, expected in sources.items():
        assert trainer.kernel.initial.digest(path) == expected, path
    protocol = json.loads((OUT / 'protocol.json').read_text())
    plan = json.loads(Path(protocol['full_row_plan']).read_text())
    subset = {'parts': {}}
    for part in ('training', 'calibration'):
        rows = plan['parts'][part]
        subset['parts'][part] = [rows[i] for i in sorted({0, len(rows)//3, 2*len(rows)//3, len(rows)-1})]
    with np.load(protocol['normalization'], allow_pickle=False) as saved:
        normalization = {n: saved[n].copy() for n in ('feature_mean', 'feature_scale', 'output_mean', 'output_scale', 'temporal_mean', 'temporal_scale')}
    device = jax.devices()[0]
    assert device.platform == 'gpu'
    uploaded = trainer.upload(subset, normalization, device)
    x, y, raw = map(np.asarray, uploaded)
    base = base_maps()
    names = json.loads((trainer.kernel.model.D38 / 'supervision/receipt.json').read_text())['teacher_labels']
    columns = [names.index(n) for n in trainer.kernel.model.LABELS]
    at, reports = 0, []
    for part in ('training', 'calibration'):
        windows = mapped_array(trainer.kernel.model.D38 / 'temporal/windows' / ('rawwindow_'+part+'.npz'), 'features')
        teachers = np.load(trainer.kernel.model.D38 / 'supervision' / (part+'_geometry.npy'), mmap_mode='r', allow_pickle=False)
        for row in subset['parts'][part]:
            a, b = row['teacher_window_span']; c, d = row['feature_span']
            size = b-a
            assert np.array_equal(raw[at:at+size], windows[a:b])
            feature = base[part]['features'][c:d].astype(float)
            expected = np.clip((feature-normalization['feature_mean'])/normalization['feature_scale'], -8., 8.).astype(np.float32)
            assert np.array_equal(x[at:at+size], expected)
            wanted = teachers[a:b, columns].astype(float)
            assert np.array_equal(np.isfinite(y[at:at+size]), np.isfinite(wanted))
            recovered = y[at:at+size].astype(float)*normalization['output_scale']+normalization['output_mean']
            valid = np.isfinite(wanted)
            maximum = float(np.max(np.abs(recovered[valid]-wanted[valid]))) if valid.any() else 0.
            assert np.allclose(recovered, wanted, atol=1e-7, rtol=0., equal_nan=True), maximum
            reports.append({'part': part, 'stem': row['stem'], 'frames': size,
                            'whole_shot_features_and_windows_bitexact': True,
                            'teacher_mask_exact': True, 'physical_teacher_reconstruction_max_m': maximum})
            at += size
    assert at == len(x) == len(y) == len(raw) and len(reports) == 8
    for path, expected in sources.items():
        assert trainer.kernel.initial.digest(path) == expected, path
    with (OUT / 'receipt.json').open('x') as stream:
        json.dump({'status': 'all8_real_eligible_whole_shot_X_data_transfers_qualified', 'reports': reports,
                   'frames': at, 'no_optimizer_updates_or_heldout_GT_read': True,
                   'partial_corpus_qualification_full_upload_still_runs_under_activation': True,
                   'seconds': time.monotonic()-started}, stream, indent=2)
    print(json.dumps({'real_X_data_transfer_qualified': 8, 'frames': at}), flush=True)


if __name__ == '__main__':
    main()
