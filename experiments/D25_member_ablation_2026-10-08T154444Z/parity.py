# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — H3 exact pipeline reproduction check
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
root=Path('$HOME/fec_first_agent_d16')
sys.path.insert(0,str(root))
import deploy_build_subframe as b
runs=Path('$FEC_RUNS/ml350/runs')
pkg=runs/'legacy_tree_fec_first_agent_20261005/deploy/pkg'
data=Path('$FEC_DATA/data')
pd=b._load_entrypoint(pkg)
nets=b.verified(root/'networks_v29_h3.sha',[])
a=pd._parser().parse_args(['--data-root',str(data),'--network',*nets,'--basis',str(pkg/'models/flux_basis.npz'),'--out','/dev/null','--split','diii_d_public_test',*b.FLAGS])
finish=pd._finish_from(a)
files=[str(p) for p in pd.split_files(data,'diii_d_public_test',None)][:1]
task=(str(pkg),files,nets,[],str(pkg/'models/flux_basis.npz'),finish,None,str(root/'tree_heads_public_test.npz'),[('li_scored',.26),('r_axis',.14),('kappa',.09),('x_lower_r',.16)],str(root/'out_public_test.npz'))
pred=b._predict_many(task)[0]
with np.load(runs/'builds_v29_20261007/build_H3/diii_d_public_test.npz',allow_pickle=False) as z:
 out={}
 for label,value in zip(('psirz','q95','betaN'),pred):
  actual=z['shot_0000_'+label]
  candidate=np.asarray(value,dtype=actual.dtype)
  out[label]={'shape':list(actual.shape),'dtype':str(actual.dtype),'max_abs_diff':float(np.max(np.abs(candidate-actual))),'array_equal':bool(np.array_equal(candidate,actual))}
print(json.dumps({'status':'complete','stem':Path(files[0]).stem,'parity':out,'builder_sha256':b.digest(b.__file__),'entrypoint_sha256':b.digest(pd.__file__)}))
assert all(v['array_equal'] for v in out.values()),out
