# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Execute sole238, conditional four trees and sole357, refusing unknown results."""
from pathlib import Path
import json,subprocess,sys,time,hashlib
D=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def emit(value):
 print(json.dumps(value),flush=True)
 with (D/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')

def step(source,argv,result):
 target=D/result
 if not target.exists():
  emit({'stage':source,'event':'started','UTC_seconds':time.time()})
  subprocess.run([sys.executable,'-B','-u',str(D/source),*argv],check=True)
 assert target.exists();emit({'stage':source,'event':'completed','result':str(target),'sha256':sha(target),'UTC_seconds':time.time()})
 record=json.loads(target.read_text())
 if result.endswith('analysis.json'):
  phase=result.split('/')[0];assert record['identity']['protocol_sha256']==sha(D/'protocol_frozen_v1.json') and record['identity']['fixed_primary']=='static' and record['identity']['share']==.5 and record['identity']['phase']==phase
  assert record['status']=='fixed_static50_all_declared_whole_shots_complete' and record['shots']==(238 if phase=='validation238' else 357)
  cases=list((D/phase/'cases').glob('*.json'));assert len(cases)==record['shots'];frames=0
  for p in cases:
   case=json.loads(p.read_text());assert case['identity']==record['identity'] and sha(p.with_suffix('.npz'))==case['arrays_sha256'];frames+=case['frames']
  assert frames==record['frames']
 return record

def main():
 manifest=json.loads((D/'protocol_frozen_v1.json').read_text());activation=json.loads((D/'execution_activation_v1.json').read_text());assert activation['protocol_sha256']==sha(D/'protocol_frozen_v1.json')
 for p,h in manifest['sources_sha256'].items():assert sha(p)==h,p
 validation=step('score_fixed_static_v1.py',['--phase','validation238','--workers','3'],'validation238/analysis.json')
 if not validation['decisions']['static']['gate_pass']:outcome='fixed_static_validation238_rejected_no357_opened'
 else:
  assert (D/'finalist_lock.json').exists()
  step('fit_shared_eligible_trees_v1.py',[],'confirmation_trees/receipt.json')
  confirmation=step('score_fixed_static_v1.py',['--phase','confirmation357','--workers','3'],'confirmation357/analysis.json')
  outcome='fixed_static_sole357_pass_candidate_requires_result_review' if confirmation['decisions']['static']['gate_pass'] else 'fixed_static_sole357_rejected'
 for p,h in manifest['sources_sha256'].items():assert sha(p)==h,p
 with (D/'pipeline_terminal.json').open('x') as f:json.dump({'status':outcome,'UTC_seconds':time.time(),'protocol_sha256':sha(D/'protocol_frozen_v1.json'),'publication':False,'no357_opened':outcome=='fixed_static_validation238_rejected_no357_opened'},f,indent=2)
 emit({'event':'D43_terminal','status':outcome,'UTC_seconds':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as error:
  with (D/('pipeline_fault_'+str(time.time_ns())+'.json')).open('x') as f:json.dump({'status':'operational_or_integrity_fault_not_scientific_rejection','error':type(error).__name__+': '+str(error),'UTC_seconds':time.time()},f,indent=2)
  raise
