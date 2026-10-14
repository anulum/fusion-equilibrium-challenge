# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Qualify actual complete families, native whole-shot prediction, fork and checkpoints."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json,multiprocessing,time
import numpy as np
import d40_prediction_v1 as d40
P=d40.P

def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--activation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 protocol=json.loads((P/'evaluation_protocol_frozen_v1.json').read_text())
 for path,h in protocol['sources_sha256'].items():assert d40.digest(path)==h,path
 assert args.out.resolve().is_relative_to(P)
 args.out.mkdir(exist_ok=True);started=time.monotonic()
 quality=d40.initialize(args.fits,args.activation);modes=tuple(d40.STATE['models'])
 if not modes:
  write(args.out/'qualification.json',{'status':'no_training_admissible_family','training_quality_pass':quality,'activation_sha256':d40.digest(args.activation),'no_held_out_labels_read':True});return
 from evaluate_development import CASES
 path=Path(CASES[0]['path']);assert CASES[0]['index']%3==0
 actual=d40.hashes(path,modes)
 expected=json.loads((d40.jp.D38/'evaluation/qualification.json').read_text())['prediction_hashes'][d40.BASE]
 assert actual['frames']==213 and actual['prediction_hashes'][d40.BASE]==expected
 for mode in modes:assert actual['prediction_hashes'][mode]['psirz']!=expected['psirz'],'trained positive contribution was inactive'
 with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('fork')) as pool:
  inherited=pool.submit(d40.hashes,str(path),modes).result(timeout=1800)
 assert inherited==actual,'actual complete8/native whole213/FORK mismatch'
 # Initialize JAX CPU only after the forked native path has completed.
 import jax,jax.numpy as jnp
 import x_training_kernel_v2 as kernel
 assert all(d.platform=='cpu' for d in jax.devices())
 row,shot,features,raw=d40.inputs(path)
 parity={}
 activation=json.loads(args.activation.read_text())
 with np.load(activation['normalization'],allow_pickle=False) as saved:normalization={n:saved[n].copy() for n in saved.files}
 for mode in modes:
  parity[mode]={}
  for seed,net in zip(range(500,508),d40.STATE['models'][mode],strict=True):
   root=args.fits/(mode+'_'+str(seed))
   assert np.array_equal(net.network.feature_mean[:582],normalization['feature_mean'])
   assert np.array_equal(net.network.feature_scale[:582],normalization['feature_scale'])
   assert np.array_equal(net.network.feature_mean[582:],np.zeros(32)) and np.array_equal(net.network.feature_scale[582:],np.ones(32))
   assert np.array_equal(net.network.output_mean,normalization['output_mean']) and np.array_equal(net.network.output_scale,normalization['output_scale'])
   assert np.array_equal(net.mean,normalization['temporal_mean']) and np.array_equal(net.scale,normalization['temporal_scale'])
   receipt=json.loads((root/'fit_report.json').read_text());frozen=receipt['frozen']
   template,key=kernel.initial_state(seed,jax.devices()[0])
   state,key,history=kernel.resume(root/'checkpoints/epoch60.npz',template,frozen)
   assert len(history)==60 and int(state[-1])==60*(receipt['training_frames']//2048)
   portable={'linear':net.network.linear,'weights':net.network.weights,'biases':net.network.biases,'conv_weights':net.weights,'conv_biases':net.biases,'readout':net.readout,'readout_bias':net.readout_bias}
   assert all(np.array_equal(np.asarray(a,dtype=float),np.asarray(b,dtype=float)) for a,b in zip(jax.tree.leaves(state[0]),jax.tree.leaves(portable),strict=True))
   normalized=np.clip((features-net.network.feature_mean[:582])/net.network.feature_scale[:582],-8.,8.)
   params=jax.tree.map(lambda a:jnp.asarray(a,dtype=jnp.float64),state[0])
   y=kernel.model.forward(params,jnp.asarray(normalized),jnp.asarray(raw,dtype=jnp.float64),jnp.asarray(net.mean),jnp.asarray(net.scale),mode)
   expected=np.asarray(y)*net.network.output_scale+net.network.output_mean
   values=net.predict(features,raw);observed=np.column_stack([values[n] for n in kernel.model.LABELS])
   difference=float(np.max(np.abs(observed-expected)))
   assert np.allclose(observed,expected,atol=1e-10,rtol=1e-10),(mode,seed,difference)
   parity[mode][str(seed)]={'actual_checkpoint_export_parameters_exact':True,'whole213_portable_CPU_JAX_double_max_abs_m':difference}
  print(json.dumps({'actual8_trained_family_qualified':mode}),flush=True)
 write(args.out/'qualification.json',{'status':'actual_complete8_families_native_positive_full69_whole213_fork_bitexact_and_checkpoint_portable_qualified','training_quality_pass':quality,'families':modes,'activation_sha256':d40.digest(args.activation),'prediction':actual,'actual_checkpoint_portable_parity':parity,'portable_tolerance_unchanged_from_prototype':{'atol':1e-10,'rtol':1e-10},'no_held_out_labels_or_scores_read':True,'not_candidate_admission':True,'seconds':time.monotonic()-started,'sources_sha256':{str(P/n):d40.digest(P/n) for n in ['d40_prediction_v1.py','qualify_X_candidate_v1.py','evaluate_X_head_pregate_v2.py']}})
if __name__=='__main__':main()
