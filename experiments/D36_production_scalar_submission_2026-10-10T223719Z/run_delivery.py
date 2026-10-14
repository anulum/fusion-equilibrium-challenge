# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bounded production package and fast HF delivery
"""Finish qualified prediction and archive QA before authorized fast private upload."""
import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
R=Path('$HOME/fusion-equilibrium-challenge')
def state(phase):
    value={'phase':phase,'utc_seconds':time.time(),'entrant_HF_and_Codabench_authorized':True}
    (p/'delivery_state.partial.json').write_text(json.dumps(value));(p/'delivery_state.partial.json').replace(p/'delivery_state.json')
    print(json.dumps(value),flush=True)

def run(argv,timeout):
    subprocess.run(argv,check=True,timeout=timeout)

try:
    deadline=time.monotonic()+2*3600;state('waiting_8_full_data_scalar_fits')
    while not(p/'fits/receipt.json').exists():
        result=subprocess.run(['systemctl','--user','show','fec-d36-production-scalar-fit-v1','-p','ActiveState'],capture_output=True,text=True).stdout
        assert 'ActiveState=active'in result,'training stopped without successful receipt'
        assert time.monotonic()<deadline,'fit deadline'
        time.sleep(30)
    assert json.loads((p/'fits/receipt.json').read_text())['successful']==8
    assert json.loads((p/'assembly_precision_qualification.json').read_text())['status']=='archive_precision_qualification_passed'
    state('building_all874_public_scalar_predictions_and_archive')
    run([str(R/'fec_adapter/.venv/bin/python'),'-B','-u',str(p/'build_submission.py')],2*3600)
    state('official_archive_QA')
    run([str(R/'fec_harness/.venv/bin/python'),'-B','-u',str(p/'qa_submission.py')],30*60)
    state('fast_private_HF_Xet_upload')
    assert os.environ['HF_XET_HIGH_PERFORMANCE']=='1'
    upload=Path('$FEC_WORK/first_agent/alldata/upload_archives.py')
    run(['/usr/bin/python3','-B','-u',str(upload),str(p/'submission')],90*60)
    receipt=json.loads((p/'submission_HF/safe_upload_receipt.json').read_text())
    assert receipt['status']=='private_HF_pinned_archives_verified'
    state('HF_verified_pointer_ready_for_Codabench_submission')
except Exception as error:
    (p/'delivery_failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),'utc_seconds':time.time()}))
    state('failed_closed_requires_repair')
    raise
