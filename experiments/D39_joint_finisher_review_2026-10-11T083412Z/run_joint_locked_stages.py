# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — supervised locked joint stages
"""Execute reviewed concrete development finalists, then one reserved candidate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PYTHON = '$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def verify():
    for source, expected in json.loads((HERE / 'joint_locked_stage_sources.json').read_text()).items():
        assert digest(source) == expected, source
    review = json.loads((HERE / 'concrete_finalists_integrity_review.json').read_text())
    assert review['individual_development_gates_verified']
    assert review['complete_case_sources_and_results_verified']
    assert review['development_lock_sha256'] == digest(HERE / 'concrete_development_finalists_lock.json')


def run(script, extra=(), timeout=40 * 3600):
    verify()
    label = Path(script).stem + ('_qualification' if '--qualify' in extra else '')
    (HERE / 'joint_stage_state.json').write_text(json.dumps({'stage': label, 'UTC_seconds': time.time()}))
    env = dict(os.environ, JAX_PLATFORMS='cpu', ROCR_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (HERE / (label + '_joint_run.log')).open('x') as log:
        result = subprocess.run([PYTHON, '-B', '-u', str(HERE / script), *extra],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    assert result.returncode == 0, (script, result.returncode)
    verify()


def main():
    started = time.time()
    try:
        verify()
        lock_path = HERE / 'concrete_development_finalists_lock.json'
        lock = json.loads(lock_path.read_text())
        assert lock['individual_development_gates_verified'] and lock['locked_before_central_scoring']
        assert lock['d38_winner'] is not None or lock['d39_change'] is not None
        assert not (HERE / 'central_development_analysis.json').exists()
        assert not (HERE / 'concrete_confirmation_candidate_lock.json').exists()
        run('evaluate_joint_development.py', ('--qualify',), timeout=4 * 3600)
        run('evaluate_joint_development.py', ('--workers', '4'))
        central_path = HERE / 'central_development_analysis.json'
        central = json.loads(central_path.read_text())
        assert central['status'] == 'all358_central_locked_finalists_complete'
        assert central['lock_sha256'] == digest(lock_path)
        winner = central['one_joint_winner']
        assert winner in ('d38_finalist', 'd39_finalist', 'composition')
        candidate = {'one_candidate_only': True, 'locked_before_reserved_candidate_scoring': True,
                     'winner': winner, 'd38_winner': lock['d38_winner'], 'd39_change': lock['d39_change'],
                     'central_analysis_sha256': digest(central_path), 'development_lock_sha256': digest(lock_path),
                     'UTC_seconds': time.time(), 'publication': False}
        candidate_path = HERE / 'concrete_confirmation_candidate_lock.json'
        write(candidate_path, candidate)
        candidate_sha = digest(candidate_path)
        run('evaluate_joint_confirmation.py', ('--qualify',), timeout=4 * 3600)
        run('fit_joint_confirmation_trees.py', timeout=18 * 3600)
        assert digest(candidate_path) == candidate_sha
        run('evaluate_joint_confirmation.py', ('--workers', '4'))
        assert digest(candidate_path) == candidate_sha
        confirmation = json.loads((HERE / 'confirmation/analysis.json').read_text())
        assert confirmation['status'] == 'all357_one_joint_candidate_confirmation_complete'
        outcome = {'status': 'joint_full358_and_single357_confirmation_complete', 'winner': winner,
                   'candidate_admitted_for_build': confirmation['confirmation_gate_pass'],
                   'confirmation_lock_sha256': candidate_sha, 'seconds': time.time() - started,
                   'publication': False, 'human_terminal_integrity_review_required': True}
        write(HERE / 'joint_stage_terminal.json', outcome)
    except Exception as error:
        write(HERE / 'joint_stage_failure.json', {'status': 'joint_stages_failed_not_admitted',
              'type': type(error).__name__, 'error': str(error), 'seconds': time.time() - started,
              'candidate_admitted': False, 'publication': False})
        raise


if __name__ == '__main__':
    main()
