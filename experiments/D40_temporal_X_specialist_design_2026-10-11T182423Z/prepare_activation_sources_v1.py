# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
import hashlib,json,time
from pathlib import Path
import jax,jaxlib,numpy as np
P=Path(__file__).resolve().parent
D=P.parent/'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
R=Path('$HOME/fusion-equilibrium-challenge/fec_adapter/fec_adapter/diii_d_accuracy')
M=Path('$FEC_RUNS/ml350/runs')
G=P.parents[3]
remote=M/'d39_finisher_20261011'
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
paths=set()
for n in ['trainer_v3_review_sources.json','prototype_qualification_sources.json','native_integration_sources.json']:
 v=json.loads((P/n).read_text());v=v.get('sources_sha256',v)
 for p,h in v.items():
  assert digest(p)==h,p
  paths.add(Path(p))
paths.update(P/n for n in ['train_X_specialists_v3.py','x_training_kernel_v2.py','shared_initial_parameters.py','x_temporal_model.py','training_recipe_frozen_v1.json','operational_resume_policy_v2.json','shared_initial_parameters/receipt.json','optimizer_resume_qualification_v2/process_chain_receipt.json','driver_recovery_qualification/receipt.json','data_transfer_qualification/receipt.json','original6326_data/receipt.json','original6326_data/fit_row_plan.json','original6326_data/normalization.npz','native_integration_qualification.json','ROCm_qualification_v2.json'])
paths.update(P/'shared_initial_parameters'/('seed'+str(s)+'.npz') for s in range(500,508))
paths.update(D/n for n in ['cohort.json','supervision/receipt.json','supervision/teacher_mapping.json','campaign_common.py','window_cache.py','training_data.py','candidate_state.py'])
paths.update(R/n for n in ['fit_network.py','fit_on_accelerators.py','network.py'])
for part in ('training','calibration'):
 paths.update([M/'tables28_subframe_20261006'/('table_'+part+'.npz'),D/'supervision'/(part+'_geometry.npy'),D/'supervision'/(part+'_shot_ids.npy'),D/'temporal/windows'/('rawwindow_'+part+'.npz'),D/'temporal'/(part+'_alignment.json')])
paths.update([remote/'src/d39_combination_locked.json',remote/'reports/screen120_all32_selection.json',remote/'src_d41/d41_locked.json',remote/'reports/d41_screen120.json',P.parent/'D39_joint_finisher_review_2026-10-11T083412Z/peer_screen_independent_report.json'])
lock=json.loads((remote/'src/d39_combination_locked.json').read_text());assert lock['no_winner'] and lock['change']=={}
assert digest(remote/'reports/screen120_all32_selection.json')==lock['screening_report_sha256']
l41=json.loads((remote/'src_d41/d41_locked.json').read_text());assert l41['selected'] is None
assert digest(remote/'reports/d41_screen120.json')==l41['screening_report_sha256']
source={}
for p in sorted(paths):
 source[str(p)]=digest(p);print('hashed',p.name,p.stat().st_size,flush=True)
value={'status':'D40_activation_sources_prepared_peer_trainer_review_pending','branch':'unused357_original6326','D39_opened357':False,'D39_opened238':False,'D41_opened238':False,'D39_terminal_outcome_source':str(remote/'src/d39_combination_locked.json'),'fit_row_plan':str(P/'original6326_data/fit_row_plan.json'),'normalization':str(P/'original6326_data/normalization.npz'),'recipe_sha256':digest(P/'training_recipe_frozen_v1.json'),'runtime_versions':{'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__},'sources_sha256':source,'fits':24,'fit_shots':6326,'fit_frames':1404214,'reference_finisher':'deployed_unchanged_D39_D41_no_winner','fit_activation_only':'positive_trained_native_and_realfork_qualification_required_before_score_admission','publication':False,'UTC_seconds':time.time()}
with (P/'activation_sources_prepared_v1.json').open('x') as f:json.dump(value,f,indent=2)
print('source preparation complete',flush=True)
