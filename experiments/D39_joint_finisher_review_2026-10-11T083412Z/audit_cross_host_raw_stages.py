# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — cross-host raw prediction stage attribution
"""Compare pre-finisher data on the four largest complete-report discrepancies."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import joint_prediction as jp

HERE = Path(__file__).resolve().parent
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    for source, expected in json.loads((HERE / 'cross_host_raw_stage_sources.json').read_text()).items():
        assert digest(source) == expected, source


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a.astype(float)-b.astype(float)
    return {'shape': list(a.shape), 'exact': bool(np.array_equal(a, b)),
            'differing_elements': int(np.count_nonzero(delta)), 'max_absolute_difference': float(np.max(np.abs(delta))),
            'RMS_difference': float(np.sqrt(np.mean(delta**2)))}


def main():
    started = time.time()
    try:
        verify()
        chosen = json.loads((HERE / 'cross_host_raw_stage_protocol.json').read_text())['cases']
        cases = {c['stem']: c for c in json.loads((jp.cp.D32 / 'manifest.json').read_text())['cases']}
        jp.initialize()
        cp, pd = jp.cp, jp.cp.STATE.pd
        reports = {}
        for stem in chosen:
            path = Path(cases[stem]['path'])
            row = cp.read_inputs(path)
            shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=float))
            features = np.concatenate([cp.STATE.layout_features(row, shot, {'discharge'})['discharge'],
                                       cp.STATE.subframe.features_of(path).astype(float)], axis=1)
            coordinates, labels = cp.STATE.together(cp.STATE.maps, {'discharge': features})
            _, joint = cp.STATE.together(cp.STATE.joint, {'discharge': np.pad(features, ((0, 0), (0, 183)))})
            trees = cp.trees_of(features)
            with np.load(PEER / 'cache' / (stem+'.npz'), allow_pickle=False) as cache:
                report = {'features_after_original_cache_float32_cast': compare(features.astype(np.float32), cache['features']),
                          'coordinates64': compare(coordinates, cache['coordinates']),
                          'labels': {str(n): compare(labels[str(n)], cache['labels'][:, i]) for i, n in enumerate(cache['label_names'])},
                          'trees': {str(n): compare(trees[str(n)], cache['tree'][:, i]) for i, n in enumerate(cache['tree_names'])},
                          'joint_q95': compare(joint['q95'], cache['joint_q95']),
                          'joint_beta_n': compare(joint['beta_n'], cache['joint_beta_n'])}
            reports[stem] = report
            print(json.dumps({'raw_stage_compared': stem}), flush=True)
        verify()
        with (HERE / 'cross_host_raw_stage_analysis.json').open('x') as stream:
            json.dump({'status': 'all4_worst_discrepancy_raw_stages_compared', 'reports': reports,
                       'root_raw_features_float64_peer_cache_preserves_only_float32_features': True,
                       'no_native_finisher_or_scorer_invoked': True, 'no_new_reference_targets_read': True,
                       'not_candidate_admission_no_tolerance_waiver': True, 'publication': False,
                       'seconds': time.time()-started}, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (HERE / 'cross_host_raw_stage_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
