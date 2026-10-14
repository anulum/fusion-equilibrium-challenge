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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
files=[str(p) for p in pd.split_files(data,'diii_d_public_test',None)]
assert len(files)==874
weak=str(root/'v29_readout_s2/discharge_hx_seed2_all/network.npz')
assert nets.count(weak)==1
nets.remove(weak)
original_predict=pd.predict_shot
def predict_logged(path,*args,**kwargs):
 value=original_predict(path,*args,**kwargs)
 print(json.dumps({'shot_done':Path(path).stem}),file=sys.stderr,flush=True)
 return value
pd.predict_shot=predict_logged
bounds=np.linspace(0,len(files),17).astype(int)
tasks=[(str(pkg),files[x:y],nets,[],str(pkg/'models/flux_basis.npz'),finish,None,str(root/'tree_heads_public_test.npz'),[('li_scored',.26),('r_axis',.14),('kappa',.09),('x_lower_r',.16)],str(root/'out_public_test.npz')) for x,y in zip(bounds[:-1],bounds[1:])]
archive=pd.PredictionArchive()
with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
 for predictions in pool.map(b._predict_many,tasks):
  for maps,q95,beta in predictions:
   assert np.isfinite(maps).all() and np.isfinite(q95).all() and np.isfinite(beta).all()
   archive.add(maps,q95,beta)
assert archive.shots==874 and archive.frames==193359 and archive.replaced==0
np.savez_compressed(sys.stdout.buffer,allow_pickle=False,**archive._entries)
sys.stdout.buffer.flush()
print(json.dumps({'status':'archive_stream_complete','shots':archive.shots,'frames':archive.frames,'network_count':len(nets),'excluded':weak,'builder_sha256':b.digest(b.__file__),'entrypoint_sha256':b.digest(pd.__file__)}),file=sys.stderr,flush=True)
