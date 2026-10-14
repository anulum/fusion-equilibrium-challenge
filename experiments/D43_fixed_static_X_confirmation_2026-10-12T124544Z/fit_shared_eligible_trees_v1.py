# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Fit unchanged four eligible confirmation trees only after fixed static238 passes."""
import json,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
from score_fixed_static_v1 import verify,old

def main():
 verify();analysis=json.loads((D/'validation238/analysis.json').read_text());lock=json.loads((D/'finalist_lock.json').read_text())
 assert analysis['decisions']['static']['gate_pass'] and lock['winner']=='static' and lock['share']==.5
 assert lock['locked_before357_scores'] and lock['validation_analysis_sha256']==old.d40.digest(D/'validation238/analysis.json')
 assert json.loads((old.d40.jp.D38/'confirmation_tree_qualification/receipt.json').read_text())['status']=='real_eligible_tree_input_export_path_qualified'
 assert not (D/'confirmation_trees/receipt.json').exists(),'completed tree fits are verified, never repeated'
 import fit_confirmation_trees as original
 original.HERE=D;train=original.xgb.train
 class Progress(original.xgb.callback.TrainingCallback):
  def after_iteration(self,model,epoch,evals_log):
   if epoch%50==0:print(json.dumps({'eligible_tree_iteration':epoch,'calibration':evals_log.get('cal',{}),'UTC_seconds':time.time()}),flush=True)
   return False
 def logged_train(*args,**kwargs):
  kwargs['callbacks']=[*kwargs.get('callbacks',[]),Progress()];return train(*args,**kwargs)
 original.xgb.train=logged_train
 try:original.fit_all(qualify=False)
 finally:original.xgb.train=train
 verify()
 receipt=D/'confirmation_trees/receipt.json';r=json.loads(receipt.read_text());assert r['status']=='all_four_eligible_shared_confirmation_trees_complete'
 assert {v['target'] for v in r['reports']}==set(original.TARGETS) and all(v['both358_development_and357_confirmation_excluded'] and v['export_reload_predictions_bitexact'] for v in r['reports'])
 old.write(D/'confirmation_trees/study_receipt_v1.json',{'protocol_sha256':old.d40.digest(D/'protocol_frozen_v1.json'),'receipt_sha256':old.d40.digest(receipt),'finalist_lock_sha256':old.d40.digest(D/'finalist_lock.json'),'unchanged_original_four_tree_fitter_source':str(Path(original.__file__)),'fitter_sha256':old.d40.digest(original.__file__)})
if __name__=='__main__':main()
