# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — actual production input and scalar parity
"""Exercise the public feature and 69-member scalar path on an eligible shot."""
import json
import sys

import numpy as np

from campaign_common import DATA, HERE, PKG, digest, write_json
from evaluate_batch import deployment_features, subframe
from fec_adapter.diii_d_accuracy.network import Network, predict_together
from fec_adapter.shot_reader import aligned_shot

sys.path.insert(0,str(PKG/'entrypoints'))
import predict_diii_d as deployed

cohort=json.loads((HERE/'cohort.json').read_text())
stem=cohort['training'][0]
assert stem not in set(cohort['holdout'])
path=DATA/(stem+'.parquet')
features=deployment_features(path)
row=deployed.read_row(path)
shot=aligned_shot(row,stem,np.asarray(row['efit_times'],dtype=float))
actual=deployed.layout_features(row,shot,{'discharge'})['discharge']
actual=np.concatenate([actual,subframe.features_of(path).astype(actual.dtype)],axis=1)
assert np.array_equal(features,actual)
lineage=json.loads((HERE/'core_lineage.json').read_text())
assert lineage['all69_core_to_readout_qbeta_bitexact']
records=[]
for member in lineage['members']:
    print(json.dumps({'qualifying_production_member':member.get('original_name',member.get('name'))}),flush=True)
    core=Network.load(member['core_path'])
    readout_path=member.get('production_readout',member.get('readout_path'))
    assert readout_path is not None
    readout=Network.load(readout_path)
    _,a=core.predict(features); _,b=readout.predict(features)
    assert all(np.array_equal(a[n],b[n]) for n in ('q95','beta_n'))
    records.append({'core':member['core_path'],'readout':readout_path,
                    'core_sha256':digest(member['core_path']),'readout_sha256':digest(readout_path),'qbeta_bitexact':True})
write_json(HERE/'feature_qualification.json',{'status':'production_features_and_all69_qbeta_bitexact',
    'eligible_shot':stem,'frames':len(features),'feature_width':582,'members':records,
    'heldout_labels_accessed':False,'prediction_comparisons_scoreless':True})
