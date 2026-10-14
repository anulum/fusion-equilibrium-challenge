# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible four-position target contract preparation
"""Stream original eligible teachers and bind temporal row alignment without fitting."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import metadata

LABELS = ('x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    cohort = json.loads((D38 / 'cohort.json').read_text())
    supervision = json.loads((D38 / 'supervision/receipt.json').read_text())
    assert supervision['status'] == 'all6326_eligible_native_geometry_and_ray_targets_complete'
    columns = [supervision['teacher_labels'].index(name) for name in LABELS]
    forbidden = set(cohort['development']) | set(cohort['reserved_confirmation'])
    count, total, squares = np.zeros(4), np.zeros(4), np.zeros(4)
    parts = {}
    sources = {}
    for part, expected in [('training', 5954), ('calibration', 372)]:
        allowed = set(cohort[part])
        assert len(allowed) == expected and not allowed & forbidden
        meta = metadata(part)
        chosen = [(s, meta['bounds'][i+1]-meta['bounds'][i]) for i, s in enumerate(meta['stems']) if s in allowed]
        assert [s for s, _ in chosen] == sorted(allowed)
        offsets = np.r_[0, np.cumsum([n for _, n in chosen])]
        alignment_path = D38 / 'temporal' / (part+'_alignment.json')
        alignment = json.loads(alignment_path.read_text())
        assert alignment['stems'] == [s for s, _ in chosen]
        assert alignment['offsets'] == offsets.tolist()
        target_path = D38 / 'supervision' / (part+'_geometry.npy')
        targets = np.load(target_path, mmap_mode='r', allow_pickle=False)
        assert targets.shape == (int(offsets[-1]), 73)
        finite_count = np.zeros(4, dtype=np.int64)
        for start in range(0, len(targets), 4096):
            values = np.asarray(targets[start:start+4096, columns], dtype=float)
            finite = np.isfinite(values)
            assert not np.isinf(values).any()
            n = finite.sum(axis=0)
            count += n
            finite_count += n
            total += np.where(finite, values, 0.).sum(axis=0)
            squares += np.where(finite, values*values, 0.).sum(axis=0)
        parts[part] = {'shots': len(chosen), 'frames': len(targets), 'finite_targets_by_column': finite_count.tolist(),
                       'ordered_stems_sha256': hashlib.sha256(json.dumps([s for s, _ in chosen]).encode()).hexdigest(),
                       'teacher_array_path': str(target_path), 'columns': columns,
                       'temporal_cache_sha256_from_original_qualified_alignment': alignment['cache_sha256']}
        for path in (target_path, alignment_path):
            sources[str(path)] = digest(path)
    assert np.all(count > 0)
    mean = total/count
    variance = np.maximum(0., squares/count-mean**2)
    common_scale = float(np.sqrt(variance.mean()))
    assert common_scale > 1e-6
    scaling_path = HERE / 'eligible_X_target_scaling_preview.npz'
    assert not scaling_path.exists()
    np.savez(scaling_path, labels=np.asarray(LABELS), mean=mean, scale=np.full(4, common_scale))
    with (HERE / 'eligible_X_target_contract_preview.json').open('x') as stream:
        json.dump({'status': 'original6326_fit_cohort_X_targets_and_temporal_alignment_prepared',
                   'parts': parts, 'label_order': LABELS, 'sources_sha256': sources,
                   'output_mean': mean.tolist(), 'shared_output_scale_m': common_scale,
                   'equal_physical_position_weights_via_shared_scale': True,
                   'undefined_extractor_targets_remain_masked_not_physical_absence_labels': True,
                   'scaling_sha256': digest(scaling_path), 'no358_or357_targets_read': True,
                   'preview_only_recompute_if_final_D40_fit_split_changes': True,
                   'separate_confirmation_contract_not_yet_frozen': True,
                   'no_optimizer_updates_no_training': True, 'publication': False}, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
