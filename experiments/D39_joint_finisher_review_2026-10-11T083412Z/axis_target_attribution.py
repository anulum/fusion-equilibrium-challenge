# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — axis target and finished-map attribution
"""Compare actual native axis targets with already scored maps at fixed times."""
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

import joint_prediction as jp
from evaluate_development import CASES

ROOT = jp.HERE / 'axis_target_attribution'
ARMS = {jp.BASE: None, 'map_control25': 'control', 'map_boundary_frame25': 'boundary_frame',
        'map_boundary_shot25': 'boundary_shot', 'map_spatial25': 'spatial'}


def digest(path):
    return jp.cp.digest(path)


def verify():
    for path, expected in json.loads((jp.HERE / 'axis_target_attribution_sources.json').read_text()).items():
        assert digest(path) == expected, path


def main():
    ROOT.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        verify()
        cp = jp.cp; jp.initialize(); pd = cp.STATE.pd
        assert cp.STATE.finish.smooth_axis == 0, 'target attribution assumes deployed unsmoothed axis labels'
        names = json.loads((jp.D38 / 'supervision/receipt.json').read_text())['teacher_labels']
        assert len(names) == 73 and not ({'r_axis', 'z_axis'} & set(names))
        features, gold, known, strata, metadata = [], [], [], [], []
        finished = {arm: [] for arm in ARMS}
        for case in CASES:
            path = Path(case['path']); row = cp.read_inputs(path)
            shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
            n = shot.n_frames
            choices = [('first', 0), ('quarter', n // 4), ('middle', n // 2), ('three_quarter', 3 * n // 4), ('last', n - 1)]
            chosen, labels = [], []
            for label, frame in choices:
                if frame not in chosen:
                    chosen.append(frame); labels.append(label)
            raw = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'],
                                  cp.STATE.subframe.features_of(path).astype(float)], axis=1)
            assert raw.shape == (n, 582) and np.isfinite(raw).all()
            features.append(raw[chosen]); strata.extend(labels)
            reference_path = cp.D32 / 'references' / (case['stem'] + '.pkl')
            reference = pickle.loads(reference_path.read_bytes())
            assert np.shape(reference[1]) == (n, 7) and np.shape(reference[2]) == (n, 7)
            gold.append(np.asarray(reference[1])[chosen, :2]); known.append(np.asarray(reference[2])[chosen, :2])
            original_path = jp.D38 / 'evaluation/development' / (case['stem'] + '.json')
            original = json.loads(original_path.read_text())
            archive = original_path.with_suffix('.npz')
            assert digest(archive) == original['arrays_sha256']
            with np.load(archive, allow_pickle=False) as saved:
                for arm in ARMS:
                    assert saved[arm + '_derived'].shape == (n, 7)
                    finished[arm].append(saved[arm + '_derived'][chosen, :2])
            metadata.extend({'stem': case['stem'], 'frame': i, 'stratum': s} for i, s in zip(chosen, labels, strict=True))
            if len(features) % 50 == 0:
                print(json.dumps({'axis_attribution_inputs_complete_shots': len(features)}), flush=True)
        x, truth, valid, strata = np.concatenate(features), np.concatenate(gold), np.concatenate(known).astype(bool), np.asarray(strata)
        assert len(features) == 358
        _, base = cp.STATE.together(cp.STATE.maps, {'discharge': x})
        trees = cp.trees_of(x)
        targets, unchanged = {}, {}
        for arm, family in ARMS.items():
            mixed = base
            if family is not None:
                _, candidate = cp.STATE.together(cp.STATE.candidate[family], {'discharge': x})
                mixed = {n: .75 * base[n] + .25 * candidate[n] for n in ('r_axis', 'z_axis')}
                for seed, after in zip(range(400, 408), cp.STATE.candidate[family], strict=True):
                    before_path = jp.D38 / 'map_fits' / f'{family}_{seed}' / 'network.npz'
                    receipt = json.loads(before_path.with_name('fit_report.json').read_text())
                    assert digest(before_path) == receipt['network_sha256']
                    before = pd.Network.load(before_path)
                    _, old_heads = before.predict(x)
                    _, new_heads = after.predict(x)
                    assert all(np.array_equal(old_heads[n], new_heads[n]) for n in ('r_axis', 'z_axis'))
                    unchanged[f'{family}_{seed}'] = True
            values = np.column_stack([mixed['r_axis'], mixed['z_axis']])
            values[:, 0] += cp.TREE_WEIGHTS['r_axis'] * (trees['r_axis'].astype(float) - values[:, 0])
            assert np.isfinite(values).all()
            targets[arm] = values
        derived = {arm: np.concatenate(value) for arm, value in finished.items()}
        report = {}
        for arm in ARMS:
            report[arm] = {}
            for stratum in ('all', 'first', 'quarter', 'middle', 'three_quarter', 'last'):
                selected = np.ones(len(x), dtype=bool) if stratum == 'all' else strata == stratum
                result = {}
                for column, name in enumerate(('R_axis', 'Z_axis')):
                    mask = selected & valid[:, column] & np.isfinite(truth[:, column]) & np.isfinite(derived[arm][:, column])
                    assert mask.any()
                    result[name] = {'samples': int(mask.sum()),
                        'target_vs_native_truth_RMSE_m': float(np.sqrt(np.mean((targets[arm][mask, column] - truth[mask, column])**2))),
                        'finished_axis_vs_native_truth_RMSE_m': float(np.sqrt(np.mean((derived[arm][mask, column] - truth[mask, column])**2))),
                        'finished_axis_vs_target_RMSE_m': float(np.sqrt(np.mean((derived[arm][mask, column] - targets[arm][mask, column])**2)))}
                report[arm][stratum] = result
        verify()
        output = {'status': 'all358_fixed_five_time_samples_axis_attribution_complete',
                  'shots': 358, 'samples': len(x), 'sampling': 'first/quarter/middle/three_quarter/last; duplicate frames removed',
                  'native_input_features_used': True, 'targets_include_original_R_axis_tree_blend': True,
                  'all32_candidate_R_Z_heads_unchanged_by_geometry73_readout': unchanged,
                  'reports': report, 'not_full_frame_performance_or_candidate_admission': True,
                  'no357_targets_read': True, 'no_models_modified': True, 'publication': False,
                  'seconds': time.monotonic() - started,
                  'sample_manifest_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}
        with (ROOT / 'analysis.json').open('x') as stream:
            json.dump(output, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (ROOT / 'failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic() - started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
