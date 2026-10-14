# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Use all eight specialists after the unchanged full69 tree-adjusted X targets."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).resolve().parent
Q=P.parent/'D39_joint_finisher_review_2026-10-11T083412Z'
sys.path.insert(0,str(Q))
import joint_prediction as jp
from evaluate_X_head_pregate_v2 import initialize as load_families
from native_x_intervention import use_x_specialist,LABELS
STATE={}
BASE=jp.BASE

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def initialize(fits,activation):
 models,quality=load_families(Path(fits),Path(activation))
 jp.initialize()
 assert len(jp.cp.STATE.maps)==69 and len(jp.cp.STATE.joint)==8
 assert jp.cp.STATE.finish.x_points and jp.cp.STATE.finish.smooth_axis==0
 STATE.update(models=models,quality=quality,activation_sha256=digest(activation))
 return quality

def inputs(path):
 cp=jp.cp;pd=cp.STATE.pd;path=Path(path)
 row=cp.read_inputs(path)
 shot=pd.aligned_shot(row,path.stem,np.asarray(row['efit_times'],dtype=float))
 features=np.concatenate([cp.STATE.layout_features(row,shot,{'discharge'})['discharge'],cp.STATE.subframe.features_of(path).astype(float)],axis=1)
 assert features.shape==(shot.n_frames,582) and np.isfinite(features).all()
 raw=np.load(jp.D38/'temporal/development'/(path.stem+'.npy'),allow_pickle=False)
 assert raw.shape==(shot.n_frames,840) and raw.dtype==np.float16
 return row,shot,features,raw

def experts(features,raw,modes):
 result={}
 for mode in modes:
  networks=STATE['models'][mode];assert len(networks)==8
  values=[net.predict(features,raw) for net in networks]
  result[mode]={n:np.mean([v[n] for v in values],axis=0) for n in LABELS}
  assert all(v.shape==(len(features),) and np.isfinite(v).all() for v in result[mode].values())
 return result

def predict(path,modes,raw_override=None):
 cp=jp.cp;pd=cp.STATE.pd;path=Path(path)
 if raw_override is None:
  row,shot,features,raw=inputs(path)
 else:
  row=cp.read_inputs(path);shot=pd.aligned_shot(row,path.stem,np.asarray(row['efit_times'],dtype=float))
  features=np.concatenate([cp.STATE.layout_features(row,shot,{'discharge'})['discharge'],cp.STATE.subframe.features_of(path).astype(float)],axis=1)
  raw=np.asarray(raw_override);assert raw.shape==(shot.n_frames,840) and raw.dtype==np.float16
  assert features.shape==(shot.n_frames,582) and np.isfinite(features).all()
 coordinates,labels=cp.STATE.together(cp.STATE.maps,{'discharge':features});trees=cp.trees_of(features)
 base=jp.finished(path,row,features,coordinates,labels,trees,{})
 _,joint=cp.STATE.together(cp.STATE.joint,{'discharge':np.pad(features,((0,0),(0,183)))})
 baseline={'psirz':base[0],**jp.blend_scalars(base,joint)}
 result={BASE:baseline}
 for mode,prediction in experts(features,raw,modes).items():
  original=pd.edited_x_points
  with use_x_specialist(pd,prediction,.5,half_window=0):
   final=jp.finished(path,row,features,coordinates,labels,trees,{})
  assert pd.edited_x_points is original
  fields={'psirz':final[0],**jp.blend_scalars(final,joint)}
  assert all(np.array_equal(fields[n],baseline[n]) for n in ('q95','betaN'))
  result[mode]=fields
 assert all(v.dtype==np.float32 and np.isfinite(v).all() for fields in result.values() for v in fields.values())
 return result,shot

def hashes(path,modes):
 fields,shot=predict(Path(path),tuple(modes))
 return {'frames':shot.n_frames,'prediction_hashes':{arm:{n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in f.items()} for arm,f in fields.items()}}
