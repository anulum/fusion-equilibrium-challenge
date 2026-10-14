# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — bounded D26 output collection and ensemble admission
import json,subprocess,time,hashlib
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
started=time.monotonic();pending=set(range(5));finished=[]
def persist(status,error=None):
 (p/'supervisor_state_v2.json').write_text(json.dumps({'status':status,'pending':sorted(pending),'finished':finished,'elapsed_seconds':time.monotonic()-started,'error':error,'next':'Fold-excluded geometry readout, raw+full-finisher OOF ensembles and unchanged full-score gate; no automatic all-data fit or public submission.'},indent=2)+'\n')
def call(args,timeout):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout
persist('polling')
while pending and time.monotonic()-started<16*3600:
 for k in sorted(pending):
  ref=f'anulum/fec-d26-paired-boundary-clearance-fold{k}'
  try:
   status=call(['kaggle','kernels','status',ref],60)
   print(status.strip(),flush=True)
   if 'COMPLETE' not in status:
    if any(s in status for s in ('ERROR','CANCEL','FAILED')):
     d=p/f'failed_v2_f{k}';d.mkdir(exist_ok=True)
     call(['kaggle','kernels','output',ref,'-p',str(d)],180)
     raise RuntimeError('kernel terminal failure: '+ref)
    continue
   d=p/f'outputs_f{k}';d.mkdir(exist_ok=True)
   call(['kaggle','kernels','output',ref,'-p',str(d)],180)
   zpath=d/f'd26_fold{k}.npz'
   with np.load(zpath,allow_pickle=False) as z:
    rows=z['rows'];assert rows.ndim==1 and len(rows)>0 and len(np.unique(rows))==len(rows)
    for arm in ('p','s'):
     assert z[f'pred_{arm}'].shape==(len(rows),13)
     assert np.isfinite(z[f'pred_{arm}']).all() and np.isfinite(z[f'coord_sse_{arm}']).all()
   manifest=[]
   for seed in (0,1):
    for arm in ('p','s'):
     name=f'discharge_hc_seed{seed}_tc_{arm}_k{k}'
     net=d/'out'/name/'network.npz';report=d/'out'/name/'fit_report.json'
     assert net.is_file() and report.is_file(),name
     with np.load(net,allow_pickle=False) as z:
      desc=json.loads(str(z['description']))
      assert len(desc['label_names'])==79 and 'magnetics_dsep_aux' in desc['label_names']
      assert len(z['feature_mean'])==582
      for field in z.files:
       if np.issubdtype(z[field].dtype,np.number):assert np.isfinite(z[field]).all(),(name,field)
     manifest.append({'name':name,'network_sha256':hashlib.sha256(net.read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()})
   (d/'collection_receipt.json').write_text(json.dumps({'fold':k,'actual_ref':ref,'status':'4_fits_and_OOF_outputs_collected','models':manifest,'rows':len(rows)},indent=2)+'\n')
   pending.remove(k);finished.append(k);persist('polling')
  except Exception as e:
   persist('failed_closed',str(e));raise
 if pending:time.sleep(300)
if pending:
 persist('deadline_expired','No restart; retained sources and partial outputs.');raise SystemExit(1)
persist('all20_fits_collected_full_score_still_pending')
