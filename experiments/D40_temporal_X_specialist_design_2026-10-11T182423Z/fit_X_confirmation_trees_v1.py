# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Fit eligible original four trees only after the sole D40 finalist is locked."""
import json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
D=P.parent/'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0,str(D))
from campaign_common import digest
import fit_confirmation_trees as original

def main():
 protocol=json.loads((P/'evaluation_protocol_frozen_v1.json').read_text())
 for path,h in protocol['sources_sha256'].items():assert digest(path)==h,path
 E=P/'evaluation_v1';lock=json.loads((E/'finalist_lock.json').read_text())
 analysis=json.loads((E/'validation238/analysis.json').read_text())
 assert lock['one_candidate_only'] and lock['locked_before357_scores']
 assert lock['validation_analysis_sha256']==digest(E/'validation238/analysis.json')
 assert analysis['decisions'][lock['winner']]['gate_pass']
 assert json.loads((D/'confirmation_tree_qualification/receipt.json').read_text())['status']=='real_eligible_tree_input_export_path_qualified'
 assert not (P/'confirmation_trees/receipt.json').exists(),'eligible trees already fitted; verify existing receipt before reuse'
 original.HERE=P
 train=original.xgb.train
 class Progress(original.xgb.callback.TrainingCallback):
  def after_iteration(self,model,epoch,evals_log):
   if epoch%50==0:
    print(json.dumps({'eligible_tree_iteration':epoch,'calibration':evals_log.get('cal',{}),'UTC_seconds':time.time()}),flush=True)
   return False
 def logged_train(*args,**kwargs):
  kwargs['callbacks']=[*kwargs.get('callbacks',[]),Progress()]
  return train(*args,**kwargs)
 original.xgb.train=logged_train
 try:original.fit_all(qualify=False)
 finally:original.xgb.train=train
 for path,h in protocol['sources_sha256'].items():assert digest(path)==h,path
if __name__=='__main__':main()
