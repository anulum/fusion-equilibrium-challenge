# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Run the complete predeclared D40 sequence; never bypass a failed scientific gate."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;E=P/'evaluation_v1'
E.mkdir(exist_ok=True)
PY=sys.executable

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def event(value):
 value={'UTC_seconds':time.time(),**value}
 with (E/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 print(json.dumps(value),flush=True)

def verify():
 protocol=json.loads((P/'evaluation_protocol_frozen_v1.json').read_text())
 for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
 return protocol

def step(stage,args,target):
 verify()
 if target.exists():
  event({'stage':stage,'event':'existing_immutable_result_reused','result':str(target)})
  return json.loads(target.read_text())
 event({'stage':stage,'event':'started','argv':args})
 subprocess.run([PY,'-B','-u',*map(str,args)],check=True)
 verify();assert target.exists(),str(target)
 event({'stage':stage,'event':'completed','result':str(target),'result_sha256':digest(target)})
 return json.loads(target.read_text())

def finish(status,**details):
 with (E/'pipeline_terminal.json').open('x') as f:json.dump({'status':status,'UTC_seconds':time.time(),'protocol_sha256':digest(P/'evaluation_protocol_frozen_v1.json'),'publication':False,**details},f,indent=2)
 event({'event':'pipeline_terminal','status':status,**details})

def main():
 verify();activation=P/'training_activation_original6326_v1.json';fits=P/'real_fits_original6326_v1'
 event({'stage':'await_all24_terminal_training','event':'started'})
 deadline=time.monotonic()+7200
 while True:
  reports=sorted(fits.glob('attempt_*.json'))
  if reports:
   report=json.loads(reports[-1].read_text())
   assert report['activation_sha256']==digest(activation)
   assert report['status']=='all24_real_X_fits_terminal','operational/integrity issue requires resolution before evaluation'
   assert len(report['fits'])==24
   break
  state=subprocess.run(['systemctl','--user','show','fec-d40-X-specialists-original6326-v1.service','-p','ActiveState','--value'],capture_output=True,text=True,timeout=10).stdout.strip()
  assert state in ('active','activating'),'training stopped without full terminal receipt'
  assert time.monotonic()<deadline,'training wait exceeded declared2h window'
  time.sleep(20)
 event({'stage':'await_all24_terminal_training','event':'completed','training_receipt':str(reports[-1]),'family_training_quality':report['family_training_quality_pass']})
 common=['--fits',fits,'--activation',activation]
 q=step('actual_positive_portable_native_fork_qualification',[P/'qualify_X_candidate_v1.py',*common,'--out',E/'qualification'],E/'qualification/qualification.json')
 if q['status']=='no_training_admissible_family':finish('all_families_failed_training_quality');return
 assert q['activation_sha256']==digest(activation)
 head=step('all120_complete8_head_pregate',[P/'evaluate_X_head_pregate_v3.py',*common,'--out',E/'head'],E/'head/head_pregate.json')
 assert head['activation_sha256']==digest(activation)
 screen=step('all_head_eligible_full69_screen120',[P/'evaluate_X_official_v1.py','--phase','screen120','--workers','3'],E/'screen120/analysis.json')
 if not screen.get('one_candidate'):finish('no_screen120_candidate',no238_or357_opened=True);return
 winner=screen['one_candidate']
 validation=step('sole_locked_candidate_validation238',[P/'evaluate_X_official_v1.py','--phase','validation238','--workers','3'],E/'validation238/analysis.json')
 if not validation['decisions'][winner]['gate_pass']:finish('sole_candidate_failed_validation238',winner=winner,no357_opened=True);return
 step('four_shared_eligible_tree_refits',[P/'fit_X_confirmation_trees_v1.py'],P/'confirmation_trees/receipt.json')
 confirmation=step('sole_locked_candidate_confirmation357',[P/'evaluate_X_official_v1.py','--phase','confirmation357','--workers','3'],E/'confirmation357/analysis.json')
 passed=confirmation['decisions'][winner]['gate_pass']
 finish('sole_candidate_confirmation_pass' if passed else 'sole_candidate_confirmation_fail',winner=winner,confirmation_gate_pass=passed)

if __name__=='__main__':
 try:main()
 except Exception as error:
  path=E/('pipeline_fault_'+str(time.time_ns())+'.json')
  with path.open('x') as f:json.dump({'status':'operational_or_integrity_resolution_required','UTC_seconds':time.time(),'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'not_scientific_rejection':True},f,indent=2)
  raise
