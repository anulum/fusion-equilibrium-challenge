# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Exploratory all-family official scores on already exposed120 shots, without admission."""
import json,sys,time,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
D=Path(__file__).resolve().parent
manifest=json.loads((D/'protocol_frozen_v1.json').read_text())
P=Path(manifest['D40_root']);sys.path.insert(0,str(P))
import evaluate_X_official_v1 as original
from evaluate_development import CASES,anatomy

def verify():
 for path,h in manifest['sources_sha256'].items():assert original.d40.digest(path)==h,path
 original.verify_sources()
 q=json.loads((P/'evaluation_v1/qualification/qualification.json').read_text())
 assert q['status']=='actual_complete8_families_native_positive_full69_whole213_fork_bitexact_and_checkpoint_portable_qualified'
 for path,h in q['sources_sha256'].items():assert original.d40.digest(path)==h,path
 assert set(manifest['families'])==set(q['families'])
 assert json.loads((P/'evaluation_v1/pipeline_terminal.json').read_text())['status']=='no_screen120_candidate'
 assert not any(json.loads((P/'evaluation_v1/head/head_pregate.json').read_text())['head_pregate_pass'].values())

def main():
 verify();modes=tuple(manifest['families']);cases=[c for c in CASES if c['index']%3==0]
 assert len(cases)==120 and sum(anatomy.STATISTICS[c['index']]['psi_n'] for c in cases)>0
 original.d40.initialize(P/'real_fits_original6326_v1',P/'training_activation_original6326_v1.json')
 identity={'protocol_sha256':original.d40.digest(D/'protocol_frozen_v1.json'),'activation_sha256':original.d40.digest(P/'training_activation_original6326_v1.json'),'phase':'diagnostic120_only_no_admission'}
 original.STATE.update(phase='screen120',root=D,modes=modes,identity=identity,reference_root=original.d40.jp.cp.D32/'references')
 (D/'cases').mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:rows=list(pool.map(original.one,cases,chunksize=1))
 assert len(rows)==120 and sum(r['frames'] for r in rows)==26597
 result=original.reduce(rows,[anatomy.STATISTICS[c['index']] for c in cases],modes,3801)
 result.update(status='exploratory_all3_official_scores120_complete_no_admission',identity=identity,publication=False,original_D40_rejection_unchanged=True,no238_or357_targets_read=True,all_family_scores_reported=True,selected_candidate=None,historical_cohort_exposure_disclosed=True,scientific_limits='Post-head-result exploratory analysis; same120 reused. No confirmation, candidate lock, submission, or proof of generalisation. Per-contrast CIs are not familywise.')
 verify();original.write(D/'analysis.json',result)
 original.write(D/'pipeline_terminal.json',{'status':result['status'],'analysis_sha256':original.d40.digest(D/'analysis.json'),'no238_or357_opened':True,'publication':False})
 print(json.dumps({'D42_diagnostic_terminal':result['status'],'decisions':result['decisions']}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  original.write(D/('pipeline_fault_'+str(time.time_ns())+'.json'),{'status':'operational_or_integrity_fault','error':type(error).__name__+': '+str(error)});raise
