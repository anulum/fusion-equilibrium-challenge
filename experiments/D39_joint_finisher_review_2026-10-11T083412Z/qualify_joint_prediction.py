# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — actual whole-shot central prediction qualification
"""Require unchanged D38 prediction hashes and real inherited-fork parity."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import joint_prediction as jp


def hashes(values):
    return {arm: {name: hashlib.sha256(array.tobytes()).hexdigest()
                  for name, array in fields.items()} for arm, fields in values.items()}


def inherited(path, winner):
    values, _shot = jp.predict(Path(path), winner, {})
    return hashes(values)


def main():
    cases = json.loads((jp.cp.D32 / 'manifest.json').read_text())['cases']
    path = Path(cases[0]['path'])
    expected = json.loads((jp.D38 / 'evaluation/qualification.json').read_text())
    assert expected['status'] == 'full69_native_map_and_scalar_finisher_bitexact_qualified'
    jp.initialize()
    results = {}
    for winner in jp.cp.ARMS[1:]:
        values, shot = jp.predict(path, winner, {})
        actual = hashes(values)
        assert actual[jp.BASE] == expected['prediction_hashes'][jp.BASE]
        assert actual['d38_finalist'] == expected['prediction_hashes'][winner], winner
        assert actual['d39_finalist'] == actual[jp.BASE]
        assert actual['composition'] == actual['d38_finalist']
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
            forked = pool.submit(inherited, str(path), winner).result(timeout=1800)
        assert forked == actual, winner
        results[winner] = actual
    # These are deliberately empty finisher flags: this gate proves transfer
    # arithmetic, not acceptance of a future selected finisher or model.
    result = {'status': 'central_empty_flags_all_six_D38_paths_bitexact_real_fork_qualified',
              'frames': shot.n_frames, 'reference_D38_qualification_sha256': jp.cp.digest(jp.D38 / 'evaluation/qualification.json'),
              'all_six_native_paths_bitexact': True, 'all_six_real_fork_paths_bitexact': True,
              'candidate_admitted': False, 'concrete_D39_flags_still_require_qualification': True,
              'prediction_hashes': results}
    with (jp.HERE / 'central_transfer_qualification.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
