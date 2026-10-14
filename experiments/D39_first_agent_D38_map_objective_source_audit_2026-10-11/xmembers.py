import sys, json, numpy as np, time
from pathlib import Path
SAS=Path('$FEC_RUNS')
X=SAS/'ml350/runs/d39_finisher_20261011'; PKG=SAS/'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
for p in (PKG/'source/fec_adapter/vendor', PKG/'source/fec_adapter', PKG/'entrypoints'): sys.path.insert(0,str(p))
import pyarrow.parquet as pq
import predict_diii_d as pd
from fec_adapter.diii_d_accuracy import x_point
from fec_adapter.diii_d_accuracy.shape_edit import Reader
tasks=json.loads((X/'d38/tasks.json').read_text())['tasks']
nets=[pd.Network.load(X/'d38/readouts'/t['name']/'network.npz') for t in tasks if t['family']!='joint']
fam=[t['family'] for t in tasks if t['family']!='joint']
cases=json.loads((X/'d32/manifest.json').read_text())['cases']
L=x_point.LABELS; reader=None; M=[];T=[];S=[]
t0=time.time()
for c in cases:
    if c['index']%3: continue
    k=np.load(X/'cache'/(c['stem']+'.npz'))
    f=k['features'].astype(float)[::2]
    if reader is None: reader=Reader(str(k['source']))
    per=np.stack([np.column_stack([n.predict(f)[1][l] for l in L]) for n in nets])   # members, frames, 4
    path=c['path']
    psi=np.asarray(pq.read_table(path,columns=['efit_psirz'],use_threads=False)['efit_psirz'][0].as_py(),dtype=float)[::2]
    T.append(np.array([x_point.x_point_positions(reader,m) for m in psi])); M.append(per); S.append(np.full(len(f),c['index']))
    if len(T)%20==0: print(len(T), time.time()-t0, flush=True)
M=np.concatenate(M,axis=1); T=np.concatenate(T); S=np.concatenate(S)
np.savez_compressed('$HOME/.first_agent/jobs/8a09b158/tmp/xmembers.npz', members=M.astype(np.float32), truth=T, shot=S, family=np.array(fam))
def dist(P):
    out={}
    for nm,a in (('lower',0),('upper',2)):
        ok=np.isfinite(T[:,a])&np.isfinite(T[:,a+1]); d=np.hypot(P[ok,a]-T[ok,a],P[ok,a+1]-T[ok,a+1])*1000
        out[nm]=(d.mean(),np.median(d),np.quantile(d,.9),np.quantile(d,.99))
    return out
est={'mean':M.mean(0),'median':np.median(M,0),'trim20':np.sort(M,0)[14:55].mean(0),
     'hx_mean':M[[i for i,f in enumerate(fam) if f=='hx']].mean(0),'hc_mean':M[[i for i,f in enumerate(fam) if f=='hc']].mean(0)}
print('frames',len(T))
for n,P in est.items():
    d=dist(P); print(f"{n:8s} lower mean {d['lower'][0]:.2f} med {d['lower'][1]:.2f} p90 {d['lower'][2]:.2f} p99 {d['lower'][3]:.1f} | upper mean {d['upper'][0]:.2f} med {d['upper'][1]:.2f} p90 {d['upper'][2]:.2f} p99 {d['upper'][3]:.1f}")
sd=M.std(0)
for nm,a in (('lower',0),('upper',2)):
    ok=np.isfinite(T[:,a])&np.isfinite(T[:,a+1]); P=est['mean']
    d=np.hypot(P[ok,a]-T[ok,a],P[ok,a+1]-T[ok,a+1]); s=np.hypot(sd[ok,a],sd[ok,a+1])
    o=np.argsort(-s); top=o[:len(o)//10]
    print(nm,'spearman(member spread, error)',np.corrcoef(np.argsort(np.argsort(s)),np.argsort(np.argsort(d)))[0,1].round(3),'share of summed error in top-10% spread frames',(d[top].sum()/d.sum()).round(3),'single-member mean error mm',np.mean([dist(M[i])[nm][0] for i in range(0,69,7)]).round(2))
