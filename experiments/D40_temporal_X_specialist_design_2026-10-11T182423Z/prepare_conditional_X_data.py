# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — conditional excluded400 X fitting data preparation
"""Recompute every specialist normalizer without the prospective400 shots."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import base_maps, metadata
from training_data import Moments
from prepare_temporal import unpack
from window_cache import mapped_array

LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')
OUT = HERE / 'conditional400_data'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    for path, expected in json.loads((OUT / 'sources.json').read_text()).items():
        assert digest(path) == expected, path


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    started = time.monotonic()
    try:
        verify()
        cohort = json.loads((D38 / 'cohort.json').read_text())
        conditional = json.loads((HERE / 'conditional400_manifest_frozen.json').read_text())
        excluded = set(conditional['stems'])
        assert len(excluded) == 400 and excluded <= set(cohort['training'])
        assert not excluded & (set(cohort['development']) | set(cohort['reserved_confirmation']) | set(cohort['calibration']))
        teacher = json.loads((D38 / 'supervision/receipt.json').read_text())
        columns = [teacher['teacher_labels'].index(n) for n in LABELS]
        base = base_maps()
        xmom, ymom, wmom = Moments(582), Moments(4), Moments(21)
        plans, totals = {}, {}
        for part in ('training', 'calibration'):
            meta = metadata(part)
            native = {s: (meta['bounds'][i], meta['bounds'][i+1]) for i, s in enumerate(meta['stems'])}
            alignment = json.loads((D38 / 'temporal' / (part+'_alignment.json')).read_text())
            assert alignment['stems'] == sorted(cohort[part])
            raw = mapped_array(D38 / 'temporal/windows' / ('rawwindow_'+part+'.npz'), 'features')
            targets = np.load(D38 / 'supervision' / (part+'_geometry.npy'), mmap_mode='r', allow_pickle=False)
            assert len(raw) == len(targets) == alignment['offsets'][-1]
            shots = []
            for i, stem in enumerate(alignment['stems']):
                if stem in excluded:
                    continue
                a, b = alignment['offsets'][i:i+2]
                c, d = native[stem]
                assert b-a == d-c
                shots.append({'stem': stem, 'teacher_window_span': [a, b], 'feature_span': [c, d]})
            expected = 5554 if part == 'training' else 372
            assert len(shots) == expected
            frames = 0
            for shot in shots:
                a, b = shot['teacher_window_span']
                c, _ = shot['feature_span']
                for offset in range(0, b-a, 4096):
                    size = min(4096, b-a-offset)
                    features = base[part]['features'][c+offset:c+offset+size].astype(float)
                    wanted = targets[a+offset:a+offset+size, columns].astype(float)
                    windows = unpack(raw[a+offset:a+offset+size]).astype(float).reshape(-1, 21)
                    assert np.isfinite(features).all() and np.isfinite(windows).all()
                    assert not np.isinf(wanted).any()
                    xmom.add(features)
                    ymom.add(wanted)
                    wmom.add(windows)
                    frames += size
            plans[part] = shots
            totals[part] = {'shots': len(shots), 'frames': frames}
            print(json.dumps({'conditional400_normalized': part, **totals[part]}), flush=True)
        assert (xmom.count > 0).all() and (ymom.count > 0).all() and (wmom.count > 0).all()
        common = float(np.sqrt(np.mean(ymom.m2/ymom.count)))
        assert common > 1e-6
        scale = OUT / 'normalization.npz'
        np.savez(scale, feature_mean=xmom.mean, feature_scale=xmom.scale(),
                 output_mean=ymom.mean, output_scale=np.full(4, common),
                 temporal_mean=wmom.mean, temporal_scale=wmom.scale(),
                 finite_target_counts=ymom.count.astype(np.int64), labels=np.asarray(LABELS))
        write(OUT / 'fit_row_plan.json', {'parts': plans, 'excluded_stems': sorted(excluded), 'totals': totals,
                                        'conditional400_manifest_sha256': digest(HERE / 'conditional400_manifest_frozen.json')})
        verify()
        write(OUT / 'receipt.json', {'status': 'conditional400_all_fit_only_X_normalizers_and_row_plan_complete',
              'totals': totals, 'fit_shots': 5926, 'fit_frames': sum(p['frames'] for p in totals.values()),
              'finite_target_counts': ymom.count.astype(int).tolist(), 'shared_output_scale_m': common,
              'normalization_sha256': digest(scale), 'row_plan_sha256': digest(OUT / 'fit_row_plan.json'),
              'activation_only_if_D39_opens357': True, 'matched69plus8_refit_still_required': True,
              'excluded400_not_read_for_normalization': True, 'no358_or357_targets_read': True,
              'not_full_training_activation': True, 'no_optimizer_updates': True,
              'seconds': time.monotonic()-started})
    except Exception as error:
        write(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
