# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — resume D27 without retraining successful models
"""Finish all independent stages before common comparative evaluation."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
REPAIR = Path(__file__).resolve().parent
CAMPAIGN = REPAIR.parent


def state(phase, detail=None):
    value = {'phase': phase, 'detail': detail, 'repair': str(REPAIR),
             'updated_utc_seconds': time.time(), 'intermediate_winner_selection': False}
    for root, name in ((REPAIR, 'state.json'), (CAMPAIGN, 'campaign_state.json')):
        temp = root / (name + '.partial')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(root / name)
    print(json.dumps(value), flush=True)


def verify():
    for manifest in (CAMPAIGN / 'frozen_sources.json', REPAIR / 'frozen_sources.json'):
        for path, expected in json.loads(manifest.read_text()).items():
            with Path(path).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, path


def command(script, arguments=(), gpu=None):
    verify()
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(REPAIR) + ':' + str(CAMPAIGN)
    if gpu is not None:
        environment['ROCR_VISIBLE_DEVICES'] = gpu
    return [sys.executable, '-u', str(script), *arguments], environment


def run(script, arguments=()):
    args, environment = command(script, arguments)
    subprocess.run(args, env=environment, check=True)


def main():
    assert json.loads((REPAIR/'qualification.json').read_text())['status'] == 'axis_cell_reference_flux_qualified'
    neural = json.loads((CAMPAIGN/'fits/receipt.json').read_text())
    assert neural['successful'] == 50 and len(neural['tasks']) == 50
    assert not (CAMPAIGN/'readouts').exists()
    assert (CAMPAIGN/'pressure_factors').is_dir() and not any((CAMPAIGN/'pressure_factors').iterdir())
    kfd = subprocess.run(['fuser','/dev/kfd'],capture_output=True,text=True)
    assert not kfd.stdout.strip(), 'GPU workload already active; do not interfere'
    state('running_pressure_target_repair_and_independent_geometry_readouts', {'neural_fits_complete':50})
    jobs = []
    for name, script, arguments, gpu in (
        ('pressure_supervision', REPAIR/'regenerate_beta_targets.py', ('--out',str(REPAIR/'beta_targets'),'--workers','3'), None),
        ('geometry_readouts', REPAIR/'refit_readouts.py', (), '1')):
        args, env = command(script, arguments, gpu)
        stream = (REPAIR/(name+'.log')).open('x')
        jobs.append((name, subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT), stream))
    terminal = {}
    while len(terminal) != len(jobs):
        for name, process, stream in jobs:
            code = process.poll()
            if code is not None and name not in terminal:
                terminal[name] = code
                stream.close()
                state('repair_and_readout_stage_progress', {'terminal':terminal})
        if len(terminal) != len(jobs):
            time.sleep(30)
    assert all(code == 0 for code in terminal.values()), terminal
    state('running_five_corrected_pressure_factor_fits')
    run(REPAIR/'fit_pressure_factor.py')
    state('building_MAST_scalar_development_predictions_no_MAST_labels')
    run(CAMPAIGN/'mast_factor_predictions.py')
    state('all_experiment_runs_terminal_begin_common_evaluation')
    run(REPAIR/'evaluate_batch.py')
    state('complete_all_comparisons_processed_together')
    with (CAMPAIGN/'campaign_receipt.json').open('x') as stream:
        json.dump({'status':'full_frozen_campaign_complete','comparisons':str(CAMPAIGN/'evaluation/summary.json'),
                   'implementation_repair':str(REPAIR/'amendment.json'),'all_50_neural_fits_preserved':True,
                   'public_upload_or_submission':False,'combined_candidate_built':False},stream,indent=2)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed_closed_requires_implementation_or_resource_review',
              {'type':type(error).__name__,'error':str(error)})
        raise
