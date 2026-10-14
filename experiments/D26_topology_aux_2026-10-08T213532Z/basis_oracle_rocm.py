# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — ROCm projection floor for scored LCFS geometry
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import jax,jax.numpy as jnp
p=Path(__file__).resolve().parent
repo=Path('$HOME/fusion-equilibrium-challenge')
sys.path.insert(0,str(repo/'fec_adapter'));sys.path.insert(0,str(repo/'fec_harness/fusion_scoring'))
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
from lcfs import extract_lcfs_with_sign,major_radius
from contour import symmetric_hausdorff
from derive import derive_frame
from common import AXIS_SIGN
assert len(jax.devices())==1 and jax.devices()[0].platform in ('gpu','rocm')
bfile=repo/'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/flux_basis.npz'
basis=FluxBasis.load(bfile)
with np.load(repo/'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge/table_calibration.npz',allow_pickle=False) as z:stems=z['stems'].astype(str)
stems=sorted(stems,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:16]
with np.load(repo/'fec_harness/fusion_scoring/masks/d3d_envelope.npz',allow_pickle=False) as z:R,Z,mask=z['grid_R'],z['grid_Z'],z['mask_coarse'].astype(bool)
data=Path('$FEC_DATA/data/diii_d_train')
records=[];parity=0.;start=time.time()
for stem in stems:
 row=pq.read_table(data/(stem+'.parquet'),columns=['efit_psirz'])['efit_psirz'][0].as_py()
 n=len(row);indices=sorted(set(int(round(q*(n-1))) for q in (.1,.5,.9)))
 truth=np.asarray([row[i] for i in indices],dtype=np.float64);del row
 x=jnp.asarray(truth.reshape(len(indices),-1));mu=jnp.asarray(basis.mean);d=jnp.asarray(basis.directions)
 coord=((x-mu)@d.T).block_until_ready()
 for k in (45,64,128):
  assert k<=len(basis.directions)
  decoded=np.asarray((coord[:,:k]@d[:k]+mu).block_until_ready()).reshape(truth.shape)
  cpu=basis.truncated(k).decode(basis.truncated(k).encode(truth))
  err=float(np.max(np.abs(decoded-cpu)));parity=max(parity,err)
  assert np.allclose(decoded,cpu,rtol=1e-10,atol=1e-11),err
  for i,t,pred in zip(indices,truth,decoded.astype(np.float32)):
   ct,_=extract_lcfs_with_sign(t.astype(np.float32),R,Z,'DIII-D',mask,mask.astype(float))
   cp,_=extract_lcfs_with_sign(pred,R,Z,'DIII-D',mask,mask.astype(float),axis_sign=AXIS_SIGN['DIII-D'])
   gt=derive_frame(t.astype(np.float32),R,Z,'DIII-D',mask,mask.astype(float),contour=ct)
   pp=derive_frame(pred,R,Z,'DIII-D',mask,mask.astype(float),contour=cp,axis_sign=AXIS_SIGN['DIII-D'])
   dist=None if ct is None else 1.0 if cp is None else float(symmetric_hausdorff(ct,cp)/major_radius(ct))
   records.append({'stem':stem,'frame':i,'k':k,'D_LCFS':dist,'prediction_contour_failed':cp is None,'map_sse':float(np.sum((pred.astype(float)-t)**2)),'truth_scalars':gt,'projected_scalars':pp})
 print(json.dumps({'stem_done':stem,'seconds':time.time()-start}),flush=True)
result={'status':'diagnostic_complete','device':str(jax.devices()[0]),'source_basis_sha256':hashlib.sha256(bfile.read_bytes()).hexdigest(),'oracle':True,'selection_or_fit':False,'CPU_GPU_max_abs_diff':parity,'seconds':time.time()-start,'summary':{str(k):{'mean_D_LCFS':float(np.mean([r['D_LCFS'] for r in records if r['k']==k and r['D_LCFS'] is not None])),'frames':sum(r['k']==k for r in records)} for k in (45,64,128)},'records':records}
(p/'basis_oracle_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary']),flush=True)
