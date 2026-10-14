# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — MAST v12 complete archive admission
import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
partial=p/'mast_public_test.partial.npz'
receipts=[json.loads(line) for line in (p/'build_remote.log').read_text().splitlines() if line.startswith('{') and '"archive_stream_complete"' in line]
assert len(receipts)==1
producer=receipts[0]
assert producer['shots']==1206 and len(producer['per_shot'])==1206
assert len({r[0] for r in producer['per_shot']})==1206
frames=0
with np.load(partial,allow_pickle=False) as z:
 expected={f'shot_{i:04d}_{head}' for i in range(1206) for head in ('psirz','q95','betaN')}
 assert set(z.files)==expected,(len(z.files),len(expected))
 for i in range(1206):
  psi=z[f'shot_{i:04d}_psirz']
  assert psi.ndim==3 and psi.shape[1:]==(65,65)
  n=psi.shape[0];frames+=n
  assert n==producer['per_shot'][i][1],i
  for head,value in [('psirz',psi),('q95',z[f'shot_{i:04d}_q95']),('betaN',z[f'shot_{i:04d}_betaN'])]:
   assert value.dtype==np.float32 and np.isfinite(value).all(),(i,head)
   if head!='psirz':assert value.shape==(n,),(i,head)
assert frames==producer['frames'] and frames>0,frames
with partial.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
final=p/'mast_public_test.npz'
assert not final.exists()
partial.rename(final)
(p/'candidate_receipt.json').write_text(json.dumps({'status':'validated_candidate_not_submitted','archive':str(final),'sha256':digest,'bytes':final.stat().st_size,'shots':1206,'frames':frames,'model':'v12_input_domain','protocol':'build_protocol.json'},indent=2)+'\n')
