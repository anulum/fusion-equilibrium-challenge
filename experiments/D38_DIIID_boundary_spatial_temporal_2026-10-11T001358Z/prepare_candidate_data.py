# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible native geometry and boundary/X supervision
"""Reuse verified true-map targets and generate every missing eligible teacher."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from campaign_common import HERE, ROOT, DATA, digest, metadata, base_maps

WORK = ROOT / 'work/second_agent-recovery-20260926'
TEACHER = Path('$FEC_RUNS/workstation/second_agent-recovery-20260926_workspace_20261008/results/original_map_geometry_targets_20260927')


def main() -> None:
    """Create complete training-only caches without touching confirmation targets."""
    cohort = json.loads((HERE / 'cohort.json').read_text())
    allowed = {'training': set(cohort['training']), 'calibration': set(cohort['calibration'])}
    forbidden = set(cohort['development']) | set(cohort['reserved_confirmation'])
    assert not (allowed['training'] & forbidden or allowed['calibration'] & forbidden)
    old = json.loads((TEACHER / 'complete_receipt.json').read_text())
    records = {row['stem']: row for row in old['records']}
    names = old['label_names']
    assert len(names) == 73
    spec = importlib.util.spec_from_file_location('original_true_map_generator', WORK / 'scratch/build_original_map_geometry_targets.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    missing_root = HERE / 'teacher_missing'
    missing_root.mkdir(exist_ok=True)
    for part in allowed:
        (missing_root / part).mkdir(exist_ok=True)
    generator.OUTPUT = missing_root
    generator.initialise()
    assert list(generator.NAMES) == names
    tables = base_maps()
    out = HERE / 'supervision'
    out.mkdir(exist_ok=True)
    generated, reused, receipts = [], 0, {}
    with np.load(Path('$HOME/fusion-equilibrium-challenge/fec_harness/fusion_scoring/masks/d3d_envelope.npz'), allow_pickle=False) as z:
        grid_r, grid_z = z['grid_R'], z['grid_Z']
    dr, dz = float(grid_r[1] - grid_r[0]), float(grid_z[1] - grid_z[0])
    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)[::2]
    lcfs_columns = [names.index('lcfs_r_' + f'{index:02d}') for index in range(0, 64, 2)]
    teacher_mapping = {}
    for part in allowed:
        meta = metadata(part)
        selected = [index for index, stem in enumerate(meta['stems']) if stem in allowed[part]]
        count = sum(meta['bounds'][index + 1] - meta['bounds'][index] for index in selected)
        target = np.lib.format.open_memmap(out / (part + '_geometry.npy'), mode='w+', dtype=np.float32, shape=(count, 73))
        points = np.lib.format.open_memmap(out / (part + '_points.npy'), mode='w+', dtype=np.float32, shape=(count, 43, 2))
        shot_ids = np.lib.format.open_memmap(out / (part + '_shot_ids.npy'), mode='w+', dtype=np.int32, shape=(count,))
        valid_rows = np.zeros(count, dtype=bool)
        at = 0
        for slot, index in enumerate(selected):
            stem = meta['stems'][index]
            lo, hi = meta['bounds'][index:index + 2]
            assert stem not in forbidden
            if stem in records:
                item = records[stem]
                path = TEACHER / Path(item['output_path']).relative_to(WORK / 'results/original_map_geometry_targets_20260927')
                assert item['part'] == part and item['frames'] == hi - lo
                assert digest(path) == item['output_sha256']
                reused += 1
            else:
                path = missing_root / part / (stem + '.npz')
                if path.exists():
                    item = json.loads(path.with_suffix('.json').read_text())
                    assert digest(path) == item['output_sha256']
                else:
                    item = generator.derive_shot({'stem': stem, 'part': part,
                        'frames': int(hi - lo), 'input_path': str(DATA / (stem + '.parquet'))})
                    path.with_suffix('.json').write_text(json.dumps(item, indent=2) + '\n')
                generated.append(item)
            with np.load(path, allow_pickle=False) as stored:
                values = stored['labels']
                assert stored['label_names'].astype(str).tolist() == names
                assert values.shape == (hi - lo, 73)
                target[at:at + len(values)] = values
            teacher_mapping[stem] = {'path': str(path), 'sha256': item['output_sha256'], 'part': part, 'frames': int(hi - lo)}
            axis = tables[part]['labels'][lo:hi, 3:5].astype(np.float32)
            radius = values[:, lcfs_columns]
            positions = np.full((len(values), 43, 2), np.nan, dtype=np.float32)
            positions[:, 0] = axis
            positions[:, 1:33, 0] = axis[:, :1] + radius * np.cos(angles)
            positions[:, 1:33, 1] = axis[:, 1:2] + radius * np.sin(angles)
            offsets = np.asarray([[0, 0], [dr, 0], [-dr, 0], [0, dz], [0, -dz]], dtype=np.float32)
            for start, end in ((33, 'lower'), (38, 'upper')):
                centre = values[:, [names.index('x_' + end + '_r'), names.index('x_' + end + '_z')]]
                positions[:, start:start + 5] = centre[:, None, :] + offsets[None, :, :]
            positions[:, :, 0] = (positions[:, :, 0] - grid_r[0]) / dr
            positions[:, :, 1] = (positions[:, :, 1] - grid_z[0]) / dz
            inside = np.isfinite(positions).all(axis=2) & (positions >= 0).all(axis=2) & (positions <= 64).all(axis=2)
            positions[~inside] = np.nan
            valid_rows[at:at + len(values)] = inside[:, 0] & inside[:, 1:].any(axis=1)
            points[at:at + len(values)] = positions
            shot_ids[at:at + len(values)] = slot
            at += len(values)
            if (slot + 1) % 200 == 0:
                print(json.dumps({'supervision_part': part, 'shots': slot + 1, 'rows': at}), flush=True)
        assert at == count
        target.flush(); points.flush(); shot_ids.flush()
        np.save(out / (part + '_valid_ray_rows.npy'), valid_rows)
        receipts[part] = {'shots': len(selected), 'frames': count, 'valid_ray_frames': int(valid_rows.sum())}
    (out / 'teacher_mapping.json').write_text(json.dumps(teacher_mapping, indent=2) + '\n')
    (out / 'receipt.json').write_text(json.dumps({'status': 'all6326_eligible_native_geometry_and_ray_targets_complete',
        'parts': receipts, 'cached_teachers_verified': reused, 'missing_teachers_generated': generated,
        'teacher_labels': names, 'development_or_confirmation_targets_read': False,
        'surrogate_axis': 'EFIT label axis; contour radii about native map axis; near-boundary approximation disclosed',
        'ray_term': 'axis-relative flux error at32rays and5pointsperX, PCA128teacher; independent of official scorer'}, indent=2) + '\n')
    print(json.dumps({'status': 'eligible_supervision_complete', 'cached': reused, 'generated': len(generated)}), flush=True)


if __name__ == '__main__':
    main()
