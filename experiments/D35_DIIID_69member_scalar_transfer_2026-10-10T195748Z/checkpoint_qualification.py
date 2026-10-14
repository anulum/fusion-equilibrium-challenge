# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — distinct current and selected checkpoint persistence
"""Verify distinct real fit parameter states survive an atomic checkpoint round trip."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from campaign_common import HERE, digest, write_json
from train_batch import restore_checkpoint, save_checkpoint
from fec_adapter.diii_d_accuracy.fit_network import FitSettings
from fec_adapter.diii_d_accuracy.network import Network

report=json.loads((HERE/'qualification/qualification_hx/fit_report.json').read_text())
settings=FitSettings(**report['settings'])
state,key,epoch,best,criterion,best_epoch=restore_checkpoint(HERE/'qualification/qualification_hx/checkpoint.npz',settings)
member=next(t for t in json.loads((HERE/'tasks.json').read_text())['tasks'] if t['family']=='hx')
original=Network.load(member['core_path'])
assert digest(member['core_path'])==member['core_sha256']
current={k:[jnp.asarray(a,dtype=jnp.float32) for a in getattr(original,k)] for k in ('weights','biases')}
current['linear']=jnp.asarray(original.linear,dtype=jnp.float32)
assert not np.array_equal(np.asarray(current['weights'][0]),np.asarray(best['weights'][0]))
probe=HERE/'qualification/distinct_state_roundtrip.npz'
state=(current,*state[1:])
save_checkpoint(probe,state,key,epoch,settings,best,criterion,best_epoch)
restored,restored_key,restored_epoch,selected,restored_criterion,selected_epoch=restore_checkpoint(probe,settings)
assert restored_epoch==epoch and selected_epoch==best_epoch and restored_criterion==criterion
assert np.array_equal(jax.random.key_data(key),jax.random.key_data(restored_key))
for left,right in ((state,restored),(best,selected)):
    assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True))
write_json(HERE/'checkpoint_qualification.json',{'status':'distinct_current_best_optimizer_PRNG_roundtrip_exact',
    'source_production_core':member['core_path'],'source_actual_oneepoch_fit':str(HERE/'qualification/qualification_hx/checkpoint.npz'),
    'checkpoint_sha256':digest(probe),'storage_integrity_probe_only':True,'optimizer_state_used_for_training':False,
    'holdout_labels_accessed':False})
