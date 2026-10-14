# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fold-excluded D26 geometry readouts on ROCm
import sys,json,hashlib,zipfile,struct,time,gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax
p=Path(__file__).resolve().parent
G=Path('$LAB')
c=G/'work/first_agent'
w=G/'work/second_agent-recovery-20260926'
sys.path.insert(0,str(w/'scratch'));sys.path.insert(0,'$HOME/fusion-equilibrium-challenge/fec_adapter')
from geometry_readout_streaming import validity_groups,cpu_moments,gpu_moment_function,corrected_from_moments
from fec_adapter.diii_d_accuracy.network import Network
storage=Path('$FEC_RUNS')
tables=storage/'ml350/runs/tables28_subframe_20261006'
targetroot=storage/'workstation/second_agent-recovery-20260926_workspace_20261008/results/original_map_geometry_targets_20260927'
originalroot=w/'results/original_map_geometry_targets_20260927'
def digest(f):
 with Path(f).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def mapped_features(path):
 with zipfile.ZipFile(path) as z:
  member=z.getinfo('features.npy');assert member.compress_type==zipfile.ZIP_STORED
 with path.open('rb') as f:
  f.seek(member.header_offset);header=f.read(30);nl,el=struct.unpack('<HH',header[26:30]);f.seek(nl+el,1)
  version=np.lib.format.read_magic(f)
  assert version in ((1,0),(2,0)),version
  shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
  offset=f.tell()
 assert not fortran and dtype==np.dtype('float32')
 return np.memmap(path,mode='r',offset=offset,shape=shape,dtype=dtype)
reserved=set(json.loads((c/'kfold/reserved_for_evaluation.json').read_text())['stems'])
splits=json.loads((c/'kfold/splits.json').read_text())
complete=json.loads((targetroot/'complete_receipt.json').read_text());all_names=complete['label_names'];assert len(all_names)==73
records={r['stem']:r for r in complete['records']}
parts={};totalrows=0
for part in ('training','calibration'):
 path=tables/f'table_{part}.npz'
 with np.load(path,allow_pickle=False) as z:stems=z['stems'].astype(str);shot=z['shot_of_row']
 counts=np.bincount(shot,minlength=len(stems));bounds=np.r_[0,np.cumsum(counts)]
 features=mapped_features(path);assert features.shape==(len(shot),582)
 parts[part]={'features':features,'stems':stems,'counts':counts,'bounds':bounds,'base':totalrows}
 totalrows+=len(shot)
truth=np.full((totalrows,73),np.nan,dtype=np.float32)
for part,d in parts.items():
 for i,stem in enumerate(d['stems']):
  if stem in reserved or stem not in set(splits[part]):continue
  r=records[stem];assert r['part']==part and r['frames']==d['counts'][i],stem
  target=targetroot/Path(r['output_path']).relative_to(originalroot)
  assert digest(target)==r['output_sha256'],stem
  with np.load(target,allow_pickle=False) as z:
   assert str(z['stem'])==stem
   a,b=d['base']+d['bounds'][i],d['base']+d['bounds'][i+1];truth[a:b]=z['labels']
 print(json.dumps({'targets_loaded':part}),flush=True)
global_groups=validity_groups(truth)
devices=jax.devices();assert len(devices)==1 and all(d.platform in ('gpu','rocm') for d in devices)
models=json.loads((p/'readout_input_models.json').read_text())
started=time.time()
def worker(card):
 results=[]
 with jax.default_device(devices[card]):
  for k in range(card,5,len(devices)):
   rowids={}
   for part,d in parts.items():
    spans=[]
    allowed=set(splits[part])
    for i,stem in enumerate(d['stems']):
     if stem in reserved or stem not in allowed or (part=='training' and int(hashlib.sha256(stem.encode()).hexdigest(),16)%5==k):continue
     spans.append(np.arange(d['bounds'][i],d['bounds'][i+1]))
    rowids[part]=np.concatenate(spans)
   for item in [m for m in models if m['fold']==k]:
    path=Path(item['path']);assert digest(path)==item['sha256']
    report=json.loads(path.with_name('fit_report.json').read_text());n=report['training_frames']
    tr=len(rowids['training']);tc=tr+len(rowids['calibration'])
    fitparts=['training'] if n==tr else ['training','calibration'] if n==tc else None
    assert fitparts is not None,(path,n,tr,tc)
    old=Network.load(path);names=[s for s in all_names if s in old.label_names];sel=[all_names.index(s) for s in names]
    groups=global_groups if len(names)==73 else validity_groups(truth[:,sel])
    fn=gpu_moment_function(old,names,groups)
    sample=rowids['training'][:128];f=np.asarray(parts['training']['features'][sample],dtype=float);y=truth[parts['training']['base']+sample][:,sel]
    gpu,bad,maskbad=fn(f,y);assert int(bad)==0 and int(maskbad)==0
    gpu=jax.tree.map(np.asarray,gpu);cpu=cpu_moments(old,f,y,names,groups)
    parity=0.
    for aa,bb in zip(cpu,gpu):
     assert int(aa[0])==int(bb[0])
     for x,z in zip(aa[1:],bb[1:]):
      err=float(np.max(np.abs(np.asarray(x)-np.asarray(z))/(1+np.abs(np.asarray(x)))))
      parity=max(parity,err);assert err<1e-9,(path,err)
    total=None
    for part in fitparts:
     d=parts[part];rows=rowids[part]
     for o in range(0,len(rows),8192):
      ids=rows[o:o+8192];f=np.asarray(d['features'][ids],dtype=float);y=truth[d['base']+ids][:,sel]
      mm,bad,maskbad=fn(f,y);assert int(bad)==0 and int(maskbad)==0
      mm=jax.tree.map(np.asarray,mm)
      total=mm if total is None else tuple(tuple(a+b for a,b in zip(x,z)) for x,z in zip(total,mm))
    new,reports=corrected_from_moments(old,total,names,groups,ridge=.001)
    changed={old.coordinates+old.label_names.index(name) for name in names}
    keep=[i for i in range(len(old.output_mean)) if i not in changed]
    for a,b in zip(old.weights[:-1]+old.biases[:-1],new.weights[:-1]+new.biases[:-1]):assert np.array_equal(a,b)
    assert np.array_equal(old.weights[-1][keep],new.weights[-1][keep]) and np.array_equal(old.biases[-1][keep],new.biases[-1][keep])
    assert np.array_equal(old.linear,new.linear) and np.array_equal(old.feature_mean,new.feature_mean) and np.array_equal(old.feature_scale,new.feature_scale)
    out=p/f'readouts_f{k}'/path.parent.name;out.mkdir(parents=True,exist_ok=False);new.save(out/'network.npz')
    receipt={'fold':k,'role':item['role'],'input':str(path),'input_sha256':item['sha256'],'network_sha256':digest(out/'network.npz'),'fit_parts':fitparts,'training_rows':tr,'calibration_rows':len(rowids['calibration']),'heldfold_reserved_excluded':True,'CPU_GPU_moment_relative_max':parity,'groups':reports,'device':str(devices[card])}
    (out/'fit_report.json').write_text(json.dumps(receipt,indent=2)+'\n');results.append(receipt)
    print(json.dumps({'readout_done':path.parent.name,'fold':k,'device':str(devices[card]),'seconds':time.time()-started}),flush=True)
    del fn,new,old,total;gc.collect()
 return results
with ThreadPoolExecutor(max_workers=len(devices)) as pool:results=[r for chunk in pool.map(worker,range(len(devices))) for r in chunk]
assert len(results)==30
(p/'geometry_readout_receipt.json').write_text(json.dumps({'status':'30_fold_excluded_geometry_readouts_completed','results':results,'seconds':time.time()-started},indent=2)+'\n')
