# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — timestamp-exact DIII-D auxiliary topology labels
import sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
root=Path('$HOME/fec_kfold')
data=Path('$FEC_DATA/data/diii_d_train')
def one(task):
 stem,n=task
 t=pq.read_table(data/(stem+'.parquet'),columns=['efit_times','magnetics_dsep_times','magnetics_dsep'])
 times=np.asarray(t['efit_times'][0].as_py(),dtype=float)
 dt=np.asarray(t['magnetics_dsep_times'][0].as_py(),dtype=float)
 y=np.asarray(t['magnetics_dsep'][0].as_py(),dtype=np.float32)
 assert len(times)==n and np.array_equal(times,dt) and y.shape==(n,),stem
 return y
arrays={};report={}
for part in ('training','calibration','validation'):
 table=root/'tables'/('table_'+part+'.npz')
 if part=='validation':table=root/'alldata/tables/table_validation.npz'
 with np.load(table,allow_pickle=False) as z:
  stems=z['stems'].astype(str);shot=z['shot_of_row'];counts=np.bincount(shot,minlength=len(stems))
 with ThreadPoolExecutor(max_workers=4) as pool:
  ys=list(pool.map(one,zip(stems,counts)))
 y=np.concatenate(ys);assert len(y)==len(shot)
 arrays[part+'_labels']=y;arrays[part+'_stems']=stems;arrays[part+'_shot_of_row']=shot
 report[part]={'shots':len(stems),'rows':len(y),'finite':int(np.isfinite(y).sum()),'quantiles':np.nanpercentile(y,[0,1,50,99,100]).tolist()}
 print(json.dumps({part:report[part]}),file=sys.stderr,flush=True)
np.savez_compressed(sys.stdout.buffer,**arrays)
print(json.dumps({'status':'labels_extracted_exact_timestamps','parts':report}),file=sys.stderr,flush=True)
