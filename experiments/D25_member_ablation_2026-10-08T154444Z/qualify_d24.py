# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — independent D24 output admission
import hashlib,json
from pathlib import Path
import numpy as np
c=Path('$FEC_WORK/first_agent')
d=c/'d24_subframe_trees'
errors=[];missing=[];checked=[]
for line in (d/'protocol_d24_0.sha256').read_text().splitlines():
 sha,name=line.split(maxsplit=1);p=Path(name)
 if not p.is_absolute():p=d/p
 if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:errors.append('hash mismatch: '+str(p))
with np.load(c/'d9_gbm_heads/nn_oof.npz',allow_pickle=False) as z:
 fold=z['row_fold']
for k in range(5):
 expected=np.flatnonzero(fold==k)
 for part,names in [('a',['q95','beta_n','li_scored','r_axis']),('b',['z_axis','kappa','x_lower_r','x_lower_z'])]:
  p=d/f'kaggle_out_f{k}{part}/gbm_sub_oof_fold{k}{part}.npz'
  if not p.exists():missing.append(str(p));continue
  with np.load(p,allow_pickle=False) as z:
   if int(z['done'])!=4:errors.append(str(p)+': unfinished heads')
   if z['targets'].astype(str).tolist()!=names:errors.append(str(p)+': wrong targets/order')
   if not np.array_equal(z['rows'],expected):errors.append(str(p)+': incomplete, duplicated, reordered or wrong-fold rows')
   if z['pred'].shape!=(len(expected),4) or not np.isfinite(z['pred']).all():errors.append(str(p)+': invalid predictions')
  checked.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(expected)})
actual=set(d.glob('kaggle_out_*/gbm_sub_oof_fold*.npz'))
if len(actual)!=len(checked):errors.append('unexpected fold/part output files')
print(json.dumps({'status':'invalid' if errors else 'pending' if missing else 'qualified_for_fresh_frozen_evaluation','errors':errors,'missing':missing,'checked':checked,'note':'Independent structure/hash gate only. No protocol threshold changes; fresh evaluator success and exact result required separately.'},indent=2))
