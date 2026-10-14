# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — calibration forward qualification diagnosis
"""Compare compiled and eager calibration passes on an actual eligible fit."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import HERE, TABLES, mapped_array, write_json
from train_batch import restore_checkpoint
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit
from fec_adapter.diii_d_accuracy.fit_network import FitSettings

report=json.loads((HERE/'qualification/qualification_hx/fit_report.json').read_text())
settings=FitSettings(**report['settings'])
state,key,epoch,best,criterion,best_epoch=restore_checkpoint(HERE/'qualification/qualification_hx/checkpoint.npz',settings)
with np.load(HERE/'scaling_tc.npz',allow_pickle=False) as stored:
    scaling={k:stored[k].copy() for k in stored.files}
raw=mapped_array(TABLES/'table_calibration.npz','features').astype(float)
held=jnp.asarray(np.clip((raw-scaling['feature_mean'])/scaling['feature_scale'],-8.,8.).astype(np.float32))
truth=mapped_array(TABLES/'table_calibration.npz','coordinates')[:,:64].astype(float)
@jax.jit
def compiled(parameters,inputs):
    return fit.forward(parameters,inputs,0.,None)[:,:64]
a=np.asarray(compiled(state[0],held),dtype=float)
b=np.asarray(fit.forward(state[0],held,0.,None),dtype=float)[:,:64]
with np.load(TABLES/'table_calibration.npz',allow_pickle=False) as stored:
    cells=float(stored['shot_cells'].sum());mean=float(stored['shot_map_sum'].sum())/cells
    sums={'flux_squares_about_the_mean':float(stored['shot_squares_total'].sum())-cells*mean**2,
          'flux_squares_left_out_by_the_basis':float(stored['shot_squares_left_out'].sum())+float(np.sum(stored['coordinates'][:,64:].astype(float)**2))}
results={}
for name,predicted in (('compiled',a),('eager',b)):
    error=predicted*scaling['output_scale'][:64]+scaling['output_mean'][:64]-truth
    sse=np.sort(np.sum(error**2,axis=1))[:-max(1,len(error)//fit.TRIMMED_SHARE)].sum()
    r2,_=fit._held_scores(error,sums)
    results[name]={'trimmed_coordinate_SSE':float(sse),'original_fit_criterion':criterion,
                   'criterion_difference':float(sse-criterion),'r2':r2,
                   'source_formula_difference':float(r2-(1.-(sse+sums['flux_squares_left_out_by_the_basis'])/sums['flux_squares_about_the_mean']))}
results['max_forward_difference']=float(np.max(np.abs(a-b)))
results['no_heldout_metrics_computed']=True
write_json(HERE/'qualification/calibration_diagnostic.json',results)
print(json.dumps(results),flush=True)
