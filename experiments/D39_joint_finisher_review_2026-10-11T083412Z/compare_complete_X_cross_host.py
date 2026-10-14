# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — complete X oracle cross-host integrity review
"""Compare every metric without accepting a tolerance or selecting a candidate."""
import hashlib
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')
LOCAL = HERE / 'X_target_causal_split'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def differences(left, right, path=''):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        return [v for key in sorted(left) for v in differences(left[key], right[key], path + '/' + key)]
    if isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        return [v for i, (a, b) in enumerate(zip(left, right, strict=True))
                for v in differences(a, b, path + '/' + str(i))]
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and not isinstance(right, bool), path
        assert math.isfinite(left) and math.isfinite(right), path
    if left == right:
        return []
    result = {'metric_path': path, 'local': left, 'peer': right}
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        result['absolute_difference'] = abs(left - right)
    return [result]


def main():
    started = time.time()
    try:
        frozen = json.loads((HERE / 'X_cross_host_sources.json').read_text())

        def verify():
            for source, expected in frozen.items():
                assert digest(Path(source)) == expected, source

        verify()
        cases = json.loads((PEER / 'd32/manifest.json').read_text())['cases']
        cohort = json.loads((HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z/cohort.json').read_text())
        assert len(cases) == 358 and [c['stem'] for c in cases] == sorted(cohort['development'])
        selected = [c for c in cases if c['index'] % 3 == 0]
        assert len(selected) == 120
        paths = [p for c in selected for p in (LOCAL / 'cases' / (c['stem'] + '.json'),
                 PEER / 'variants/base' / (c['stem'] + '.json'), PEER / 'variants/orx' / (c['stem'] + '.json'))]
        deadline = time.monotonic() + 30 * 3600
        while not all(p.is_file() for p in paths) or not (LOCAL / 'terminal.json').is_file():
            (HERE / 'X_cross_host_state.json').write_text(json.dumps({
                'phase': 'waiting_for_complete120_local_and_remote', 'available_records': sum(p.is_file() for p in paths),
                'total_records': 360, 'UTC_seconds': time.time()}))
            assert not (LOCAL / 'failure.json').exists(), 'local factorial failed'
            assert time.monotonic() < deadline, 'complete120 wait deadline expired'
            time.sleep(60)
        verify()
        analysis = json.loads((LOCAL / 'analysis.json').read_text())
        assert analysis['status'] == 'all4x120_post_smoothing_X_causal_diagnostic_complete'
        assert analysis['shots'] == 120 and analysis['nondeployable_nonselectable']
        correction = json.loads((HERE / 'X_target_causal_split_protocol_v2_base_smoothing_correction.json').read_text())
        assert correction['baseline_target_half_window'] == 0
        bound = {str(p): digest(p) for p in paths}
        detailed, frame_count = [], 0
        protocol = digest(HERE / 'X_target_causal_split_protocol_v2.json')
        for case in selected:
            stem = case['stem']
            local_path = LOCAL / 'cases' / (stem + '.json')
            local = json.loads(local_path.read_text())
            assert local['protocol_sha256'] == protocol
            assert digest(local_path.with_suffix('.npz')) == local['arrays_sha256']
            bound[str(local_path.with_suffix('.npz'))] = local['arrays_sha256']
            expected = json.loads((HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z/evaluation/development' / (stem + '.json')).read_text())
            assert local['parts']['baseline'] == expected['parts']['matched69_scalar8']
            frame_count += local['frames']
            for variant, arm in (('base', 'baseline'), ('orx', 'positions_and_validity')):
                remote = json.loads((PEER / 'variants' / variant / (stem + '.json')).read_text())
                assert remote['stem'] == local['stem'] == case['stem']
                assert remote['index'] == local['index'] == case['index']
                assert remote['frames'] == local['frames'] == case['span'][1] - case['span'][0]
                diff = differences(local['parts'][arm], remote['part'])
                detailed.append({'stem': stem, 'variant': variant, 'exact': not diff, 'differences': diff})
        assert frame_count == analysis['frames']
        bound[str(LOCAL / 'analysis.json')] = digest(LOCAL / 'analysis.json')
        bound[str(LOCAL / 'terminal.json')] = digest(LOCAL / 'terminal.json')
        verify()
        for source, expected in bound.items():
            assert digest(Path(source)) == expected, source
        totals = {v: sum(not r['exact'] for r in detailed if r['variant'] == v) for v in ('base', 'orx')}
        write('X_cross_host_review.json', {
            'status': 'all120_base_and_joint_oracle_cross_host_metrics_compared',
            'shots': 120, 'frames': frame_count, 'mismatching_shots': totals, 'details': detailed,
            'all_metric_parts_exact': not any(totals.values()), 'source_sha256': frozen, 'record_sha256': bound,
            'no_tolerance_accepted': True, 'differences_require_human_integrity_review': bool(any(totals.values())),
            'base_target_smoothing_identity': True,
            'original_local_analysis_non_equivalence_label_superseded_by_preserved_correction': True,
            'diagnostic_only_not_candidate_admission': True, 'no357_targets_read': True, 'publication': False,
            'seconds': time.time() - started})
    except Exception as error:
        write('X_cross_host_failure.json', {'type': type(error).__name__, 'error': str(error),
              'seconds': time.time() - started, 'candidate_admitted': False})
        raise


if __name__ == '__main__':
    main()
