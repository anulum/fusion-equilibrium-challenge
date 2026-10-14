# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — finite production69 scalar transfer workflow
"""Finish every fit before the single prospective confirmation comparison."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESTARTS=0


def state(phase: str, detail: dict | None = None) -> None:
    """Retain operational progress atomically without scientific selection."""
    temporary=HERE/'state.partial.json'
    temporary.write_text(json.dumps({'phase':phase,'detail':detail,'utc_seconds':time.time(),
                                    'intermediate_score_selection':False},indent=2))
    temporary.replace(HERE/'state.json')
    print(json.dumps({'phase':phase,'detail':detail}),flush=True)


def verify_sources() -> None:
    """Refuse altered source, recipes, controls or input table bindings."""
    for file,expected in json.loads((HERE/'frozen_sources.json').read_text()).items():
        with Path(file).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,file
    for row in json.loads((HERE/'input_tables.json').read_text()):
        value=Path(row['path']).stat()
        assert value.st_size==row['size'] and value.st_mtime_ns==row['mtime_ns'],row['path']


def wait_resources() -> None:
    """Wait finitely for idle GPUs and 9GiB available RAM without stopping peers."""
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        available=next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        holders=subprocess.run(['fuser','/dev/kfd'],capture_output=True,text=True)
        if not holders.stdout.strip() and available>=9*1024**2:
            return
        state('waiting_idle_GPU_and_RAM',{'available_GiB':available/1024**2})
        time.sleep(60)
    raise TimeoutError('resource deadline')


def run_stage(script: str, receipt: str, expected: str, gpu: bool) -> None:
    """Resume once after an operational interruption, with no numerical seed retry."""
    global RESTARTS
    path=HERE/receipt
    if path.exists():
        assert json.loads(path.read_text())['status']==expected
        return
    if gpu:
        wait_resources()
    while True:
        verify_sources()
        state('running_'+script.removesuffix('.py'),{'operational_restarts_used':RESTARTS})
        environment=dict(os.environ)
        if gpu:
            environment['ROCR_VISIBLE_DEVICES']='1,2,3'
        argv=[sys.executable,'-B','-u',str(HERE/script)]
        if gpu:
            argv+=['--out',str(HERE/'fits')]
        result=subprocess.run(argv,env=environment,timeout=48*3600,check=False)
        if result.returncode==0:
            assert json.loads(path.read_text())['status']==expected
            return
        if RESTARTS>=1:
            raise RuntimeError(f'{script} failed after bounded checkpoint resume: {result.returncode}')
        RESTARTS+=1
        state('operational_checkpoint_restart',{'stage':script,'exit':result.returncode})
        if gpu:
            wait_resources()


def main() -> None:
    """Require actual-path qualifications, finish77 fits, then evaluate all358 shots."""
    verify_sources()
    qualified=json.loads((HERE/'qualification/qualification.json').read_text())
    assert qualified['status']=='full_production69_scalar_ROCm_path_qualified' and qualified['training_rows']==1480465
    assert qualified['scaling_source_parity'] and len(qualified['results'])==3
    assert all(r['CPU_GPU_double_max_abs']<1e-8 and r['checkpoint_exact'] for r in qualified['results'])
    assert json.loads((HERE/'feature_qualification.json').read_text())['status']=='production_features_and_all69_qbeta_bitexact'
    assert json.loads((HERE/'checkpoint_qualification.json').read_text())['status']=='distinct_current_best_optimizer_PRNG_roundtrip_exact'
    run_stage('train_batch.py','fits/receipt.json','all_77_neural_tasks_terminal',True)
    neural=json.loads((HERE/'fits/receipt.json').read_text())
    assert neural['successful']==77,'numerical failure/incomplete arm; no partial comparison'
    run_stage('evaluate_batch.py','evaluation/summary.json','all_scalar_transfer_comparisons_complete',False)
    verify_sources()
    summary=json.loads((HERE/'evaluation/summary.json').read_text())
    receipt={'status':'D35_workflow_complete','utc_seconds':time.time(),'successful_new_fits':77,
        'shots':summary['shots'],'frames':summary['frames'],'operational_restarts':RESTARTS,
        'prospective_transfer_gate_passes':summary['prospective_transfer_gate_passes'],
        'absolute_full_S_computed':False,'public_upload_or_submission':False}
    with (HERE/'terminal_receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    state('complete_requires_interpretation_no_publication',receipt)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        receipt={'status':'D35_failed_closed','utc_seconds':time.time(),'type':type(error).__name__,
                 'error':str(error),'operational_restarts':RESTARTS,'public_upload_or_submission':False}
        if not (HERE/'terminal_receipt.json').exists():
            with (HERE/'terminal_receipt.json').open('x') as stream:
                json.dump(receipt,stream,indent=2)
        state('failed_closed_requires_review',receipt)
        raise
