# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — full69 X integration qualification
"""Require byte equality on a complete native shot, including an inherited fork."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parent / 'D39_joint_finisher_review_2026-10-11T083412Z'
sys.path.insert(0, str(Q))
import joint_prediction as jp
from native_x_intervention import use_x_specialist


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    for path, expected in json.loads((HERE / 'native_integration_sources.json').read_text()).items():
        assert digest(path) == expected, path


def hashes(fields):
    return {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in fields.items()}


def predict(path, share, no_op=False):
    def provider(targets):
        assert no_op, 'zero contribution must never evaluate a specialist'
        return {name: value.copy() for name, value in targets.items()}

    pd = jp.cp.STATE.pd
    original = pd.edited_x_points
    with use_x_specialist(pd, provider, share, half_window=0) as state:
        values, shot = jp.predict(Path(path), None, None)
    assert pd.edited_x_points is original
    fields = values[jp.BASE]
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in fields.values())
    return hashes(fields), shot.n_frames, state


def main():
    started = time.time()
    try:
        verify()
        cases = json.loads((jp.cp.D32 / 'manifest.json').read_text())['cases']
        path = cases[0]['path']
        expected = json.loads((jp.D38 / 'evaluation/qualification.json').read_text())['prediction_hashes'][jp.BASE]
        jp.initialize()
        assert jp.cp.STATE.finish.smooth_axis == 0
        rows = {}
        for name, share, no_op in [('zero', 0., False), ('same_target_noop', .25, True)]:
            actual = predict(path, share, no_op)
            assert actual[0] == expected and actual[1] == 213
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
                inherited = pool.submit(predict, path, share, no_op).result(timeout=1800)
            assert inherited == actual
            rows[name] = {'prediction_hashes': actual[0], 'frames': actual[1], 'state': actual[2],
                          'native_and_real_fork_bitexact': True}
        verify()
        with (HERE / 'native_integration_qualification.json').open('x') as stream:
            json.dump({'status': 'full69_whole213_zero_and_native_X_noop_real_fork_bitexact_qualified',
                       'results': rows, 'baseline_qualification_sha256': digest(jp.D38 / 'evaluation/qualification.json'),
                       'positive_noop_bypasses_arithmetic_only_when_expert_equals_original_targets': True,
                       'nonzero_trained_specialist_not_yet_qualified': True,
                       'base_flags_only_selected_D39_flags_need_separate_qualification': True,
                       'no_optimizer_updates_or_training_or_reference_labels_read': True,
                       'not_candidate_admission': True, 'publication': False,
                       'seconds': time.time()-started}, stream, indent=2)
    except Exception as error:
        with (HERE / 'native_integration_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
