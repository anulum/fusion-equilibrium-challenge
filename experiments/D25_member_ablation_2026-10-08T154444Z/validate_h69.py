# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — H69 complete archive admission
import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
partial=p/'diii_d_public_test.partial.npz'
frames=0
with np.load(partial,allow_pickle=False) as z:
 expected={f'shot_{i:04d}_{head}' for i in range(874) for head in ('psirz','q95','betaN')}
 assert set(z.files)==expected,(len(z.files),len(expected))
 for i in range(874):
  psi=z[f'shot_{i:04d}_psirz']
  assert psi.ndim==3 and psi.shape[1:]==(65,65)
  n=psi.shape[0];frames+=n
  for head,value in [('psirz',psi),('q95',z[f'shot_{i:04d}_q95']),('betaN',z[f'shot_{i:04d}_betaN'])]:
   assert value.dtype==np.float32 and np.isfinite(value).all(),(i,head)
   if head!='psirz':assert value.shape==(n,),(i,head)
assert frames==193359,frames
with partial.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
final=p/'diii_d_public_test.npz'
assert not final.exists()
partial.rename(final)
(p/'candidate_receipt.json').write_text(json.dumps({'status':'validated_candidate_not_submitted','archive':str(final),'sha256':digest,'bytes':final.stat().st_size,'shots':874,'frames':frames,'network_count':69,'protocol':'build_protocol.json'},indent=2)+'\n')
