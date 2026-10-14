import sys, json, numpy as np, time
from pathlib import Path
SAS=Path('$FEC_RUNS')
X=SAS/'ml350/runs/d39_finisher_20261011'; PKG=SAS/'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
for p in (PKG/'source/fec_adapter/vendor', PKG/'source/fec_adapter', PKG/'entrypoints'): sys.path.insert(0,str(p))
import predict_diii_d as pd
from fec_adapter.diii_d_accuracy import x_point
from fec_adapter.diii_d_accuracy.shape_edit import Reader
basis=pd.FluxBasis.load(PKG/'models/flux_basis.npz').truncated(64)
cases=json.loads((X/'d32/manifest.json').read_text())['cases']
z=np.load('$HOME/.first_agent/jobs/8a09b158/tmp/xmembers.npz'); H=z['members'].mean(0); T=z['truth']; S=z['shot']
reader=None; MS=[]
for c in cases:
    if c['index']%3: continue
    k=np.load(X/'cache'/(c['stem']+'.npz')); co=k['coordinates'].astype(float).copy()
    if reader is None: reader=Reader(str(k['source']))
    co[:,45:]=0.; co[:,30:]=pd.smooth_in_time(co[:,30:][:,:,None],2)[:,:,0]
    maps=basis.decode(co)[::2]
    MS.append(np.array([x_point.x_point_positions(reader,m) for m in maps]))
MS=np.concatenate(MS); assert len(MS)==len(T)
np.save('$HOME/.first_agent/jobs/8a09b158/tmp/xmap_saddles.npy',MS)
sh=np.unique(S); fit=np.isin(S,sh[::2]); test=~fit
for nm,a in (('lower',0),('upper',2)):
    okT=np.isfinite(T[:,a])&np.isfinite(T[:,a+1]); okM=np.isfinite(MS[:,a])&np.isfinite(MS[:,a+1])
    print('==',nm,'truth-defined',okT.sum(),'map has saddle on truth-defined %.3f'%(okM[okT].mean()),'map has saddle where truth has none %.3f'%(okM[~okT].mean()))
    ok=okT&okM
    dh=np.hypot(H[ok,a]-T[ok,a],H[ok,a+1]-T[ok,a+1])*1000; dm=np.hypot(MS[ok,a]-T[ok,a],MS[ok,a+1]-T[ok,a+1])*1000
    print('  both defined: head mean %.2f med %.2f p90 %.2f | raw-map saddle mean %.2f med %.2f p90 %.2f'%(dh.mean(),np.median(dh),np.quantile(dh,.9),dm.mean(),np.median(dm),np.quantile(dm,.9)))
    eh=np.column_stack([H[ok,a]-T[ok,a],H[ok,a+1]-T[ok,a+1]]); em=np.column_stack([MS[ok,a]-T[ok,a],MS[ok,a+1]-T[ok,a+1]])
    print('  correlation of head and map errors R %.3f Z %.3f'%(np.corrcoef(eh[:,0],em[:,0])[0,1],np.corrcoef(eh[:,1],em[:,1])[0,1]))
    f=fit[ok]; t=test[ok]
    for w in (0,.25,.5,.75,1):
        e=(1-w)*eh+w*em; d=np.hypot(e[:,0],e[:,1])*1000
        print('  blend map weight %.2f: mean distance all %.2f (fit half %.2f, other half %.2f)'%(w,d.mean(),d[f].mean(),d[t].mean()))
    cl=np.clip(em,-.02,.02); # robust: map error clipped is not available at inference; instead clip the map-head difference
    diff=np.column_stack([MS[ok,a]-H[ok,a],MS[ok,a+1]-H[ok,a+1]]); nd=np.hypot(diff[:,0],diff[:,1])
    for cap in (.005,.01,.02):
        sc=np.minimum(1,cap/np.maximum(nd,1e-9))[:,None]; e=eh+.5*sc*diff; d=np.hypot(e[:,0],e[:,1])*1000
        print('  head + half of the map-head difference capped at %g m: mean %.2f'%(cap,d.mean()))
