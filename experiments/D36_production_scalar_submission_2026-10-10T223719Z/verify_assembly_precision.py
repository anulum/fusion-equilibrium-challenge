# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — qualification of archive scalar blend precision
"""Bound archive rounding on all existing fit-disjoint confirmation predictions."""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
old=p.parent/'D35_DIIID_69member_scalar_transfer_2026-10-10T195748Z'
records=json.loads((old/'evaluation/statistics.json').read_text());sse=[];max_difference=0.
for row in records:
 with np.load(old/'evaluation/shots'/(row['stem']+'.npz'),allow_pickle=False)as z:
  blend=(.75*z['baseline69'].astype(float)+.25*z['joint8'].astype(float)).astype(np.float32)
  max_difference=max(max_difference,float(np.max(np.abs(blend-z['blend69']))))
  valid=np.isfinite(z['truth']);sse.append((np.where(valid,z['truth']-blend.astype(float),0.)**2).sum(axis=0))
sums=np.array([r['sum']for r in records]);sq=np.array([r['sumsq']for r in records]);counts=np.array([r['count']for r in records]);sse=np.array(sse);base=np.array([r['sse'][0]for r in records])
def delta(weights):
 denom=weights@sq-(weights@sums)**2/(weights@counts)
 return .15*float(np.mean((weights@base-weights@sse)/denom))
result=delta(np.ones(358));rng=np.random.default_rng(996)
ci=np.percentile([delta(np.bincount(rng.integers(0,358,358),minlength=358))for _ in range(2000)],[2.5,97.5])
assert result>=1e-5 and ci[0]>0
original=json.loads((old/'evaluation/summary.json').read_text())['comparisons']['69']['delta_S_if_map_identical']
assert abs(result-original)<1e-9
receipt={'status':'archive_precision_qualification_passed','delta_S':result,'ci95':ci.tolist(),
 'max_scalar_absolute_difference':max_difference,'original_delta_difference':result-original,
 'weights_unchanged':True,'purpose':'numericaldeploymentqualification of existingpredictions; noholdoutweighttuning'}
(p/'assembly_precision_qualification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
