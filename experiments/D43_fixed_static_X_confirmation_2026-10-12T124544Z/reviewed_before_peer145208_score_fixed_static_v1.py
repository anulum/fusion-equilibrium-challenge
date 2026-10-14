# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Confirm the fixed static eight-member50% intervention on238, then on sole357."""
import argparse,json,sys,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
D=Path(__file__).resolve().parent
plan=json.loads((D/'study_draft_v1.json').read_text());P=Path(plan['D40_root']);D42=Path(plan['D42_official_root']);sys.path.insert(0,str(P))
import evaluate_X_official_v1 as old
from evaluate_development import CASES,anatomy
from campaign_common import DATA

def verify():
 protocol=json.loads((D/'protocol_frozen_v1.json').read_text());activation=json.loads((D/'execution_activation_v1.json').read_text())
 assert activation['status']=='fixed_static_confirmation_authorized_and_reviewed_source_bound'
 assert activation['protocol_sha256']==old.d40.digest(D/'protocol_frozen_v1.json')
 for path,h in protocol['sources_sha256'].items():assert old.d40.digest(path)==h,path
 old.verify_sources();q=json.loads((P/'evaluation_v1/qualification/qualification.json').read_text())
 assert q['status']=='actual_complete8_families_native_positive_full69_whole213_fork_bitexact_and_checkpoint_portable_qualified'
 assert q['activation_sha256']==old.d40.digest(P/'training_activation_original6326_v1.json')
 for path,h in q['sources_sha256'].items():assert old.d40.digest(path)==h,path
 primary=json.loads((D42/'diagnostic_interpretation_and_future_primary_lock_v1.json').read_text())
 assert primary['primary_family']=='static' and primary['share']==.5 and primary['seeds']==list(range(500,508))
 assert primary['no_selection_among3_by120_S'] and primary['secondary_mean_ordered_diagnostic_only']
 assert protocol['primary_family']=='static' and protocol['share']==.5
 assert protocol['bootstrap_seeds']=={'validation238':3802,'confirmation357':3802}
 assert not any(json.loads((P/'evaluation_v1/head/head_pregate.json').read_text())['head_pregate_pass'].values())
 assert json.loads((D42/'terminal_independent_arithmetic_audit_v1.json').read_text())['status']=='all120_artifact_integrity_unchanged_QB_exact_baseline_and_independent_pooled_metric_bootstrap_audit_PASS'
 return protocol,q

def load_eligible_trees():
 folder=D/'confirmation_trees';receipt=json.loads((folder/'receipt.json').read_text());assert receipt['status']=='all_four_eligible_shared_confirmation_trees_complete'
 provenance=json.loads((folder/'study_receipt_v1.json').read_text());assert provenance['protocol_sha256']==old.d40.digest(D/'protocol_frozen_v1.json') and provenance['receipt_sha256']==old.d40.digest(folder/'receipt.json') and provenance['finalist_lock_sha256']==old.d40.digest(D/'finalist_lock.json')
 trees={}
 for name in old.d40.jp.cp.TREE_WEIGHTS:
  root=folder/name;r=json.loads((root/'fit_report.json').read_text());assert r['both358_development_and357_confirmation_excluded'] and r['export_reload_predictions_bitexact']
  assert old.d40.digest(root/'tree.ubj')==r['model_sha256'] and old.d40.digest(root/'lookup.npz')==r['lookup_sha256']
  booster=old.d40.jp.cp.STATE.xgb.Booster();booster.load_model(root/'tree.ubj');booster.set_param({'device':'cpu','nthread':1})
  with np.load(root/'lookup.npz',allow_pickle=False) as z:trees[name]=booster,z['leaf_values'].copy(),z['base_score'].copy()
 old.d40.jp.cp.STATE.trees=trees
 return old.d40.digest(folder/'receipt.json')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validation238','confirmation357'],required=True);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();protocol,q=verify();root=D/args.phase;root.mkdir(exist_ok=True)
 identity={'protocol_sha256':old.d40.digest(D/'protocol_frozen_v1.json'),'original_training_activation_sha256':old.d40.digest(P/'training_activation_original6326_v1.json'),'phase':args.phase,'fixed_primary':'static','share':.5}
 modes=('static',)
 if args.phase=='validation238':
  cases=[c for c in CASES if c['index']%3!=0];assert len(cases)==238;expected_frames=sum(c['span'][1]-c['span'][0] for c in cases)
 else:
  validation=json.loads((D/'validation238/analysis.json').read_text());lock=json.loads((D/'finalist_lock.json').read_text());assert validation['decisions']['static']['gate_pass']
  assert lock['winner']=='static' and lock['one_candidate_only'] and lock['locked_before357_scores'] and lock['validation_analysis_sha256']==old.d40.digest(D/'validation238/analysis.json')
  identity['finalist_lock_sha256']=old.d40.digest(D/'finalist_lock.json')
  cohort=json.loads((old.d40.jp.D38/'cohort.json').read_text())['reserved_confirmation'];assert len(cohort)==357
  cases=[{'stem':s,'path':str(DATA/(s+'.parquet'))} for s in sorted(cohort)];expected_frames=76251
 old.d40.initialize(P/'real_fits_original6326_v1',P/'training_activation_original6326_v1.json')
 old.STATE.update(phase=args.phase,root=root,modes=modes,identity=identity,reference_root=old.d40.jp.cp.D32/'references')
 if args.phase=='confirmation357':
  # Original-tree path equals existing actual native qualification before any357 labels.
  path=Path(CASES[0]['path']);before=old.d40.hashes(path,modes);expected=q['prediction']
  assert before=={'frames':expected['frames'],'prediction_hashes':{a:expected['prediction_hashes'][a] for a in (old.d40.BASE,*modes)}}
  identity['eligible_tree_receipt_sha256']=load_eligible_trees();native=old.d40.hashes(path,modes)
  with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('fork')) as pool:assert pool.submit(old.d40.hashes,str(path),modes).result(timeout=1800)==native
  old.write(root/'native_qualification.json',{'status':'sole_static50_shared_eligible_four_trees_native_FORK_bitexact_on213','identity':identity,'prediction':native,'no357_labels_read':True})
  from window_cache import mapped_array
  aligned=json.loads((old.d40.jp.D38/'temporal/validation_alignment.json').read_text());bounds={s:tuple(aligned['offsets'][i:i+2]) for i,s in enumerate(aligned['stems'])};assert set(bounds)=={c['stem'] for c in cases}
  windows=mapped_array(old.d40.jp.D38/'temporal/windows/rawwindow_validation.npz','features');assert windows.dtype==np.float16
  old.STATE.update(window_bounds=bounds,windows=windows)
  import evaluate_confirmation as native_ref
  native_ref.HERE=root/'native_reference_builder';refroot=native_ref.HERE/'confirmation/references';refroot.mkdir(parents=True,exist_ok=True);old.STATE['reference_root']=refroot
  with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:refs=list(pool.map(native_ref.reference_of,[c['stem'] for c in cases],chunksize=1))
  references=[r['statistics'] for r in refs];assert sum(r['frames'] for r in refs)==expected_frames
  totals={k:np.sum([r[k] for r in references],axis=0) for k in references[0]};anatomy.MEANS=(totals['psi_sum']/totals['psi_n'],totals['scal_sum']/np.maximum(totals['scal_n'],1),totals['cons_sum']/np.maximum(totals['cons_n'],1))
 else:references=[anatomy.STATISTICS[c['index']] for c in cases]
 (root/'cases').mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:rows=list(pool.map(old.one,cases,chunksize=1))
 assert len(rows)==len(cases) and sum(r['frames'] for r in rows)==expected_frames
 result=old.reduce(rows,references,modes,3802);result.update(status='fixed_static50_all_declared_whole_shots_complete',identity=identity,publication=False,one_candidate_only=True,D40_rejection_unchanged=True,primary_locked_before_D42_aggregate=True,historical_cohort_exposure_disclosed=True)
 fails={arm:sum(sum(r['parts'][arm]['cons_fails']) for r in rows) for arm in (old.d40.BASE,'static')}
 result['decisions']['static']['Consistency_failure_frames']=fails
 result['decisions']['static']['gate_pass']=bool(result['decisions']['static']['gate_pass'] and fails['static']<=fails[old.d40.BASE])
 verify();old.write(root/'analysis.json',result)
 if args.phase=='validation238' and result['decisions']['static']['gate_pass']:
  old.write(D/'finalist_lock.json',{'winner':'static','share':.5,'one_candidate_only':True,'locked_before357_scores':True,'validation_analysis_sha256':old.d40.digest(root/'analysis.json'),'identity':identity})
 print(json.dumps({'D43_scoring_complete':args.phase,'gate':result['decisions']['static']}),flush=True)
if __name__=='__main__':main()
