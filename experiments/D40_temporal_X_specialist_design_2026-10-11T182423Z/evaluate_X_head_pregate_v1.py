# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Evaluate complete eight-member X families on the declared 120 shots only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
P=Path(__file__).resolve().parent
Q=P.parent/'D39_joint_finisher_review_2026-10-11T083412Z'
sys.path.insert(0,str(Q))
import joint_prediction as jp
from evaluate_development import CASES
from x_temporal_model import XTemporalNetwork,LABELS,MODES
from native_x_intervention import use_x_specialist

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def initialize(fits,activation):
 models={};quality={}
 for mode in MODES:
  receipts=[json.loads((fits/(mode+'_'+str(seed))/'fit_report.json').read_text()) for seed in range(500,508)]
  assert all(r['frozen']['activation_sha256']==digest(activation) for r in receipts)
  quality[mode]=all(r['status']=='fit_complete' and r['quality']['seed_quality_pass'] for r in receipts)
  if not quality[mode]:continue
  for seed,r in zip(range(500,508),receipts,strict=True):
   root=fits/(mode+'_'+str(seed))/'export'
   assert digest(root/'network.npz')==r['network_sha256'] and digest(root/'encoder.npz')==r['encoder_sha256']
  models[mode]=[XTemporalNetwork(fits/(mode+'_'+str(seed))/'export') for seed in range(500,508)]
 return models,quality

def summarize(rows,key):
 result={}
 for end,start in [('lower',0),('upper',2)]:
  distances=[];sums=0.;count=0;before=[];after=[]
  for row in rows:
   truth=row['truth'][:,start:start+2]; valid=np.isfinite(truth).all(axis=1)
   if not valid.any():continue
   error=row[key][valid,start:start+2]-truth[valid]
   distance=np.linalg.norm(error,axis=1)
   distances.append(float(np.mean(distance)));sums+=float(distance.sum());count+=len(distance)
   before.append((row['baseline'][valid,start:start+2]-truth[valid]).ravel());after.append(error.ravel())
  assert distances and count>0
  b,a=np.concatenate(before),np.concatenate(after)
  result[end]={'equal_shot_mean_distance_m':float(np.mean(distances)),'pooled_frame_mean_distance_m':sums/count,'eligible_shots':len(distances),'eligible_frames':count,'paired_component_error_correlation':float(np.corrcoef(b,a)[0,1])}
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--fits',type=Path,required=True);a.add_argument('--activation',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 activation=json.loads(args.activation.read_text());assert activation['branch']=='unused357_original6326'
 assert args.out.resolve().is_relative_to(P)
 for p,h in activation['sources_sha256'].items():assert digest(p)==h,p
 models,quality=initialize(args.fits,args.activation)
 assert models,'no complete admissible eight-member family'
 jp.initialize();cp=jp.cp;pd=cp.STATE.pd
 assert cp.STATE.finish.x_points and cp.STATE.finish.smooth_axis==0
 from fec_adapter.diii_d_accuracy.shape_edit import Reader
 from fec_adapter.diii_d_accuracy.x_point import x_point_positions
 selected=[c for c in CASES if c['index']%3==0];assert len(selected)==120
 tasks=[t for t in json.loads((jp.D38/'tasks.json').read_text())['tasks'] if t['family']!='joint']
 recipe=json.loads((P/'training_recipe_frozen_v1.json').read_text())
 names=recipe['head_reporting_additions']['base8_descriptive_member_subset_fixed_now']
 base8=[cp.STATE.maps[next(i for i,t in enumerate(tasks) if t['name']==n)] for n in names]
 assert len(base8)==8
 rows=[];started=time.monotonic()
 for case in selected:
  path=Path(case['path']);row=cp.read_inputs(path);shot=pd.aligned_shot(row,path.stem,np.asarray(row['efit_times'],dtype=float))
  features=np.concatenate([cp.STATE.layout_features(row,shot,{'discharge'})['discharge'],cp.STATE.subframe.features_of(path).astype(float)],axis=1)
  assert features.shape==(shot.n_frames,582) and np.isfinite(features).all()
  raw=np.load(jp.D38/'temporal/development'/(path.stem+'.npy'),allow_pickle=False);assert raw.shape==(shot.n_frames,840) and raw.dtype==np.float16
  coordinates,labels=cp.STATE.together(cp.STATE.maps,{'discharge':features});trees=cp.trees_of(features)
  capture={}
  def provider(targets):
   capture.update({n:np.asarray(targets[n],dtype=float).copy() for n in LABELS});return targets
  with use_x_specialist(pd,provider,.5,half_window=0):
   baseline=jp.finished(path,row,features,coordinates,labels,trees,{})
  assert set(capture)==set(LABELS) and np.isfinite(baseline[0]).all()
  target=np.column_stack([capture[n] for n in LABELS])
  truth_table=pq.read_table(path,columns=['efit_psirz'],use_threads=False)
  flux=np.asarray(truth_table['efit_psirz'][0].as_py(),dtype=np.float32)
  reader=Reader(str(shot.source));truth=np.asarray([x_point_positions(reader,m.astype(float)) for m in flux])
  assert truth.shape==target.shape==(shot.n_frames,4)
  _,heads8=cp.STATE.together(base8,{'discharge':features})
  for n,w in cp.TREE_WEIGHTS.items():
   if n in LABELS:heads8[n]+=w*(trees[n].astype(float)-heads8[n])
  record={'stem':case['stem'],'index':case['index'],'truth':truth,'baseline':target,'descriptive_base8':np.column_stack([heads8[n] for n in LABELS])}
  for mode,networks in models.items():
   values=[np.column_stack([v[n] for n in LABELS]) for v in [m.predict(features,raw) for m in networks]]
   assert len(values)==8 and all(np.isfinite(v).all() and v.shape==target.shape for v in values)
   expert=np.mean(values,axis=0);record[mode+'_expert']=expert;record[mode+'_share050']=.5*target+.5*expert
   for seed,value in zip(range(500,508),values,strict=True):
    record[mode+'_'+str(seed)+'_expert']=value;record[mode+'_'+str(seed)+'_share050']=.5*target+.5*value
  rows.append(record)
  print(json.dumps({'D40_head_pregate_shots':len(rows),'frames':shot.n_frames}),flush=True)
 baseline=summarize(rows,'baseline');report={'baseline':baseline,'descriptive_base8':summarize(rows,'descriptive_base8')};passes={}
 for mode in models:
  expert=summarize(rows,mode+'_expert');mixed=summarize(rows,mode+'_share050')
  ratios={end:mixed[end]['equal_shot_mean_distance_m']/baseline[end]['equal_shot_mean_distance_m'] for end in ('lower','upper')}
  passes[mode]=all(v<=.90 for v in ratios.values())
  report[mode]={'raw_expert':expert,'share050':mixed,'ratios':ratios,'per_seed':{str(seed):{'raw_expert':summarize(rows,mode+'_'+str(seed)+'_expert'),'share050':summarize(rows,mode+'_'+str(seed)+'_share050')} for seed in range(500,508)}}
 args.out.mkdir(exist_ok=True)
 write(args.out/'head_pregate.json',{'status':'all_declared120_complete_X_head_pregate','activation_sha256':digest(args.activation),'source_sha256':digest(__file__),'shots':len(rows),'frames':sum(len(r['truth']) for r in rows),'training_quality_pass':quality,'head_pregate_pass':passes,'reports':report,'share':.5,'all8_members_used':True,'full69_actual_tree_adjusted_smoothed_baseline':True,'no238_or357_targets_read':True,'not_official_S_admission':True,'seconds':time.monotonic()-started})
if __name__=='__main__':main()
