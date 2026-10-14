# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Score fixed complete families and allow only one locked confirmation candidate."""
import argparse,json,pickle,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import d40_prediction_v1 as d40
from evaluate_development import CASES,anatomy,part_numbers
from analyze_development import scores
from campaign_common import DATA
P=d40.P; E=P/'evaluation_v1';STATE={}

def write(path,value):
 with path.open('x') as f:json.dump(anatomy.serial(value),f,indent=2,allow_nan=False)

def verify_sources():
 manifest=json.loads((P/'evaluation_protocol_frozen_v1.json').read_text())
 for path,h in manifest['sources_sha256'].items():assert d40.digest(path)==h,path
 return manifest

def eligible_trees():
 folder=P/'confirmation_trees';receipt=json.loads((folder/'receipt.json').read_text())
 assert receipt['status']=='all_four_eligible_shared_confirmation_trees_complete'
 trees={}
 for name in d40.jp.cp.TREE_WEIGHTS:
  root=folder/name;r=json.loads((root/'fit_report.json').read_text())
  assert r['both358_development_and357_confirmation_excluded']
  assert d40.digest(root/'tree.ubj')==r['model_sha256'] and d40.digest(root/'lookup.npz')==r['lookup_sha256']
  booster=d40.jp.cp.STATE.xgb.Booster();booster.load_model(root/'tree.ubj');booster.set_param({'device':'cpu','nthread':1})
  with np.load(root/'lookup.npz',allow_pickle=False) as z:trees[name]=booster,z['leaf_values'].copy(),z['base_score'].copy()
 d40.jp.cp.STATE.trees=trees
 return d40.digest(folder/'receipt.json')

def truth_of(path):
 import pyarrow.parquet as pq
 table=pq.read_table(path,columns=['efit_psirz','efit_q95','efit_beta_n'],use_threads=False)
 return {'psi':np.asarray(table['efit_psirz'][0].as_py(),dtype=np.float32),'q95':np.asarray(table['efit_q95'][0].as_py(),dtype=float),'betaN':np.asarray(table['efit_beta_n'][0].as_py(),dtype=float)}

def predict_case(case):
 raw=None
 if STATE['phase']=='confirmation357':
  a,b=STATE['window_bounds'][case['stem']];raw=STATE['windows'][a:b]
 return d40.predict(Path(case['path']),STATE['modes'],raw_override=raw)

def one(case):
 root=STATE['root'];p=root/'cases'/(case['stem']+'.json');archive=p.with_suffix('.npz')
 if p.exists():
  r=json.loads(p.read_text());assert r['identity']==STATE['identity'] and d40.digest(archive)==r['arrays_sha256'];return r
 assert not archive.exists(),'orphan case archive requires explicit integrity recovery'
 started=time.monotonic();path=Path(case['path']);truth=truth_of(path)
 reference_path=STATE['reference_root']/(case['stem']+'.pkl');reference=pickle.loads(reference_path.read_bytes())
 predicted,shot=predict_case(case);assert shot.n_frames==len(truth['psi'])
 parts={};arrays={};fingerprints={}
 for arm,fields in predicted.items():
  parts[arm],distance,derived=anatomy.score_maps(truth,reference,fields)
  arrays[arm+'_distance']=distance;arrays[arm+'_derived']=derived
  arrays[arm+'_flux_mse']=np.mean((fields['psirz'].astype(float)-truth['psi'])**2,axis=(1,2))
  arrays[arm+'_q95']=fields['q95'];arrays[arm+'_betaN']=fields['betaN']
  import hashlib
  fingerprints[arm]={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in fields.items()}
 if STATE['phase']!='confirmation357':
  old=json.loads((d40.jp.D38/'evaluation/development'/(case['stem']+'.json')).read_text())
  assert anatomy.serial(parts[d40.BASE])==old['parts'][d40.BASE],'native baseline differs from qualified full69'
 tmp=archive.with_name(case['stem']+'.partial.npz');np.savez_compressed(tmp,**arrays);tmp.replace(archive)
 r={'stem':case['stem'],'frames':shot.n_frames,'index':case.get('index'),'parts':parts,'identity':STATE['identity'],'arrays_sha256':d40.digest(archive),'reference_sha256':d40.digest(reference_path),'prediction_hashes':fingerprints,'seconds':time.monotonic()-started}
 write(p,r);print(json.dumps({'D40_official_phase':STATE['phase'],'shot':case['stem'],'seconds':r['seconds']}),flush=True);return anatomy.serial(r)

def reduce(rows,references,modes,seed):
 arms=(d40.BASE,*modes);parts={a:[part_numbers(r['parts'][a]) for r in rows] for a in arms}
 totals={k:np.sum([r[k] for r in references],axis=0) for k in references[0]}
 exact={};official={}
 for a in arms:
  exact[a]={k:float(v[0]) for k,v in scores(np.ones((1,len(rows))),references,parts[a]).items()}
  acc=anatomy.scorer.Accum();acc.psi_sign=1.
  for p in parts[a]:acc.add(p)
  official[a]=anatomy.scorer.finalize_machine(acc,totals)
  assert not official[a]['psi_sign_probe_disagrees']
  assert all(round(v,6)==official[a][k] for k,v in exact[a].items())
 weights=np.random.default_rng(seed).multinomial(len(rows),np.full(len(rows),1./len(rows)),size=10000).astype(float)
 base_sample=scores(weights,references,parts[d40.BASE])['S'];decisions={}
 for a in modes:
  delta=exact[a]['S']-exact[d40.BASE]['S'];ci=np.quantile(scores(weights,references,parts[a])['S']-base_sample,[.005,.995])
  failures={b:sum(p['lcfs_fails'] for p in parts[b]) for b in (d40.BASE,a)}
  nonfinite=sum(p['psi_nonfinite'] for p in parts[a])
  decisions[a]={'delta_S':delta,'CI99':ci.tolist(),'LCFS_failures':failures,'nonfinite_pixels':nonfinite,'gate_pass':bool(delta>=3e-5 and ci[0]>0 and failures[a]<=failures[d40.BASE] and nonfinite==0)}
 return {'shots':len(rows),'frames':sum(r['frames'] for r in rows),'unrounded_scores':exact,'official_scores':official,'decisions':decisions,'bootstrap_seed':seed,'replicates':10000,'CI_per_contrast_not_familywise':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen120','validation238','confirmation357'],required=True);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
 manifest=verify_sources();activation=P/'training_activation_original6326_v1.json';fits=P/'real_fits_original6326_v1'
 q=json.loads((E/'qualification/qualification.json').read_text());assert q['activation_sha256']==d40.digest(activation)
 assert q['status']=='actual_complete8_families_native_positive_full69_whole213_fork_bitexact_and_checkpoint_portable_qualified'
 for path,h in q['sources_sha256'].items():assert d40.digest(path)==h,path
 head=json.loads((E/'head/head_pregate.json').read_text());assert head['activation_sha256']==d40.digest(activation)
 head_modes=tuple(m for m in manifest['family_order'] if head['head_pregate_pass'].get(m,False))
 root=E/args.phase;root.mkdir(exist_ok=True)
 identity={'protocol_sha256':d40.digest(P/'evaluation_protocol_frozen_v1.json'),'activation_sha256':d40.digest(activation),'phase':args.phase}
 if args.phase=='screen120':
  modes=head_modes
  if not modes:
   write(root/'analysis.json',{'status':'no_head_pregate_passer','one_candidate':None,'no238_or357_targets_read':True,'identity':identity});return
  cases=[c for c in CASES if c['index']%3==0];assert len(cases)==120;seed=3801
 else:
  screen=json.loads((E/'screen120/analysis.json').read_text());lock=json.loads((E/'screen120/candidate_lock.json').read_text())
  assert lock['screen_analysis_sha256']==d40.digest(E/'screen120/analysis.json')
  assert lock['winner']==screen['one_candidate'] and screen['decisions'][lock['winner']]['gate_pass']
  modes=(lock['winner'],);identity['candidate_lock_sha256']=d40.digest(E/'screen120/candidate_lock.json')
  if args.phase=='validation238':
   cases=[c for c in CASES if c['index']%3!=0];assert len(cases)==238;seed=3802
  else:
   validation=json.loads((E/'validation238/analysis.json').read_text());final=json.loads((E/'finalist_lock.json').read_text())
   assert validation['decisions'][modes[0]]['gate_pass'] and final['winner']==modes[0]
   assert final['validation_analysis_sha256']==d40.digest(E/'validation238/analysis.json')
   identity['finalist_lock_sha256']=d40.digest(E/'finalist_lock.json');seed=3802
   cohort=json.loads((d40.jp.D38/'cohort.json').read_text())['reserved_confirmation'];assert len(cohort)==357
   cases=[{'stem':s,'path':str(DATA/(s+'.parquet'))} for s in sorted(cohort)]
 assert set(modes)<=set(q['families'])
 d40.initialize(fits,activation);STATE.update(phase=args.phase,root=root,modes=modes,identity=identity,reference_root=d40.jp.cp.D32/'references')
 if args.phase=='confirmation357':
  # Qualify the locked path on development inputs before opening any357 labels.
  path=Path(CASES[0]['path']);before=d40.hashes(path,modes)
  expected=q['prediction'];assert before=={'frames':expected['frames'],'prediction_hashes':{a:expected['prediction_hashes'][a] for a in (d40.BASE,*modes)}}
  identity['eligible_tree_receipt_sha256']=eligible_trees()
  native=d40.hashes(path,modes)
  with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('fork')) as pool:assert pool.submit(d40.hashes,str(path),modes).result(timeout=1800)==native
  write(root/'native_qualification.json',{'status':'locked_one_candidate_shared_eligible_trees_native_and_FORK_bitexact','identity':identity,'prediction':native,'no357_labels_read':True})
  from window_cache import mapped_array
  aligned=json.loads((d40.jp.D38/'temporal/validation_alignment.json').read_text());bounds={s:tuple(aligned['offsets'][i:i+2]) for i,s in enumerate(aligned['stems'])}
  assert set(bounds)=={c['stem'] for c in cases}
  windows=mapped_array(d40.jp.D38/'temporal/windows/rawwindow_validation.npz','features');assert windows.dtype==np.float16
  STATE.update(window_bounds=bounds,windows=windows,reference_root=root/'references')
  STATE['reference_root'].mkdir(exist_ok=True)
  # The original reference builder uses only its overridden output folder.
  import evaluate_confirmation as original
  # Use root/confirmation references, matching the unchanged reference helper.
  original.HERE=root/'native_reference_builder';native_root=original.HERE/'confirmation/references';native_root.mkdir(parents=True,exist_ok=True)
  STATE['reference_root']=native_root
  with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:refs=list(pool.map(original.reference_of,[c['stem'] for c in cases],chunksize=1))
  references=[r['statistics'] for r in refs];assert sum(r['frames'] for r in refs)==76251
  totals={k:np.sum([r[k] for r in references],axis=0) for k in references[0]}
  anatomy.MEANS=(totals['psi_sum']/totals['psi_n'],totals['scal_sum']/np.maximum(totals['scal_n'],1),totals['cons_sum']/np.maximum(totals['cons_n'],1))
 else:references=[anatomy.STATISTICS[c['index']] for c in cases]
 (root/'cases').mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:rows=list(pool.map(one,cases,chunksize=1))
 assert len(rows)==len(cases)
 result=reduce(rows,references,modes,seed);result.update(status='all_declared_whole_shots_complete',identity=identity,publication=False)
 if args.phase=='screen120':
  passed=[m for m in modes if result['decisions'][m]['gate_pass']]
  winner=max(passed,key=lambda m:(result['unrounded_scores'][m]['S'],-manifest['family_order'].index(m))) if passed else None
  result['one_candidate']=winner
 write(root/'analysis.json',result)
 if args.phase=='screen120' and result['one_candidate']:
  write(root/'candidate_lock.json',{'winner':result['one_candidate'],'screen_analysis_sha256':d40.digest(root/'analysis.json'),'one_candidate_only':True,'locked_before238_scores':True,'identity':identity})
 if args.phase=='validation238' and result['decisions'][modes[0]]['gate_pass']:
  write(E/'finalist_lock.json',{'winner':modes[0],'one_candidate_only':True,'locked_before357_scores':True,'validation_analysis_sha256':d40.digest(root/'analysis.json'),'screen_candidate_lock_sha256':d40.digest(E/'screen120/candidate_lock.json'),'identity':identity})
 verify_sources()
if __name__=='__main__':main()
