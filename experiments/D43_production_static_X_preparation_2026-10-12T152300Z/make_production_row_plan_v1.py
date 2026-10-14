# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — production static specialist row planning
"""Plan all labelled rows from metadata without opening target arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLES = Path('$SAS/'
              'DATA_fusion-equilibrium-challenge/ml350/runs/tables28_subframe_20261006')


def digest(path: Path) -> str:
    """Hash a source without loading its complete contents into RAM."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    """Bind every shot to its feature rows and existing teacher receipt."""
    draft_path = HERE / 'preparation_draft_v1.json'
    draft = json.loads(draft_path.read_text())
    d38 = Path(draft['D38_root'])
    cohort_path = d38 / 'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    mapping_path = d38 / 'supervision/teacher_mapping.json'
    cached = json.loads(mapping_path.read_text())
    parts: dict[str, list[dict]] = {}
    totals: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    missing: list[str] = []
    sources = {str(path): digest(path) for path in
               (Path(__file__), draft_path, cohort_path, mapping_path)}
    for part in ('training', 'calibration'):
        path = TABLES / f'table_{part}.npz'
        with np.load(path, allow_pickle=False) as archive:
            stems = archive['stems'].astype(str).tolist()
            shot_ids = archive['shot_of_row']
            counts = np.bincount(shot_ids, minlength=len(stems))
            assert str(archive['feature_layout']) == 'discharge'
        assert len(stems) == len(set(stems)) == len(counts)
        assert not seen.intersection(stems)
        seen.update(stems)
        bounds = np.r_[0, np.cumsum(counts)].astype(int)
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            assert np.all(shot_ids[start:end] == index)
        rows = []
        for index, stem in enumerate(stems):
            teacher = cached.get(stem)
            if teacher is None:
                missing.append(stem)
            else:
                assert teacher['part'] == part
                assert teacher['frames'] == int(counts[index])
            rows.append({'stem': stem, 'part': part,
                         'feature_span': bounds[index:index + 2].tolist(),
                         'frames': int(counts[index]),
                         'cached_teacher': teacher})
        parts[part] = rows
        totals[part] = {'shots': len(stems), 'frames': int(bounds[-1])}
    eligible = set(cohort['training']) | set(cohort['calibration'])
    held = set(cohort['development']) | set(cohort['reserved_confirmation'])
    assert len(eligible) == 6326 and len(held) == 715
    assert seen == eligible | held and len(seen) == draft['fit_shots']
    assert set(cached) == eligible and set(missing) == held
    assert sum(row['frames'] for row in totals.values()) == draft['fit_frames']
    plan = {'status': 'all7041_metadata_row_plan_no_target_arrays_opened',
            'parts': parts, 'totals': totals,
            'cached_teacher_shots': len(cached),
            'missing_native_teacher_stems': sorted(missing),
            'missing_native_teacher_shots': len(missing),
            'sources_sha256': sources,
            'table_paths': {part: str(TABLES / f'table_{part}.npz') for part in parts},
            'full_table_sha256_required_at_activation': True,
            'targets_or_maps_read': False, 'optimizer_updates': 0,
            'must_wait_for_reviewed_D43_357_PASS': True}
    with (HERE / 'metadata_row_plan_v1.json').open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    print(json.dumps({'totals': totals, 'cached_teachers': len(cached),
                      'missing_teachers': len(missing), 'targets_read': False}))


if __name__ == '__main__':
    main()
