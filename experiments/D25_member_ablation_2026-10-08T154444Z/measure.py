# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — frozen D25 member contribution diagnostic
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

root = Path('$HOME/fec_first_agent_d16')
sys.path.insert(0, str(root / 'pkg'))
from fec_adapter.diii_d_accuracy.network import Network
started = time.time()
table = Path('$FEC_RUNS/ml350/runs/tables28_subframe_20261006/table_calibration.npz')
with np.load(table, allow_pickle=False) as z:
    stems = z['stems'].astype(str)
    selected = sorted(range(len(stems)), key=lambda i: hashlib.sha256(stems[i].encode()).hexdigest())[:32]
    rows = np.flatnonzero(np.isin(z['shot_of_row'], selected))
    features = z['features'][rows]
    coords = z['coordinates'][rows].astype(float)
    truth = z['labels'][rows].astype(float)
    names = z['label_names'].astype(str).tolist()
    _, shot = np.unique(z['shot_of_row'][rows], return_inverse=True)
listing = (root / 'networks_v29_h3.sha').read_text().splitlines()
assert len(listing) == 70
weak_path = root / 'v29_readout_s2/discharge_hx_seed2_all/network.npz'
acc = {k: {} for k in ('all', 'plain', 'weak')}
count = {k: {} for k in acc}
lineage = []
clip = np.zeros(len(rows), dtype=bool)
for i, line in enumerate(listing):
    sha, name = line.split(maxsplit=1)
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    report = json.loads((path.parent / 'fit_report.json').read_text())
    plain = report['fit_parts'] == ['training']
    net = Network.load(path)
    cs, heads = net.predict(features)
    vals = dict(heads, coordinates=cs)
    groups = ['all'] + (['plain'] if plain else []) + (['weak'] if path == weak_path else [])
    for group in groups:
        for k, val in vals.items():
            acc[group][k] = acc[group].get(k, 0) + val
            count[group][k] = count[group].get(k, 0) + 1
    if path == weak_path:
        clip = np.any(np.abs((features-net.feature_mean)/net.feature_scale) > 8, axis=1)
    lineage.append({'path':name,'sha256':sha,'fit_parts':report['fit_parts']})
    print(f'member {i+1}/70 elapsed {time.time()-started:.1f}s',file=sys.stderr,flush=True)
assert count['weak']['coordinates'] == 1
rng = np.random.default_rng(990)
boot = np.stack([np.bincount(rng.integers(0,32,32),minlength=32) for _ in range(2000)])
result = {}
for group in ('all','plain'):
    result[group] = {}
    for name,total in acc[group].items():
        if name not in acc['weak'] or (name != 'coordinates' and name not in names):
            continue
        n = count[group][name]
        before = total/n
        after = (total-acc['weak'][name])/(n-1)
        y = coords[:,:before.shape[1]] if name == 'coordinates' else truth[:,names.index(name)]
        ok = np.isfinite(y).all(axis=1) if y.ndim==2 else np.isfinite(y)
        assert np.isfinite(before).all() and np.isfinite(after).all()
        eb,ea = (before-y)**2,(after-y)**2
        if y.ndim==2: eb,ea=eb.sum(axis=1),ea.sum(axis=1)
        pb=np.bincount(shot[ok],weights=eb[ok],minlength=32)
        pa=np.bincount(shot[ok],weights=ea[ok],minlength=32)
        ratios=(boot@pa)/(boot@pb)
        result[group][name]={'members':n,'frames':int(ok.sum()),'sse_before':float(pb.sum()),'sse_after':float(pa.sum()),'ratio':float(pa.sum()/pb.sum()),'ci95':np.percentile(ratios,[2.5,97.5]).tolist(),'P_improvement':float(np.mean(ratios<1)),'clipped_frames':int((ok&clip).sum()),'clipped_sse_share_before':float(eb[ok&clip].sum()/pb.sum())}
out={'status':'complete','shots':stems[selected].tolist(),'frames':len(rows),'metrics':result,'lineage':lineage,'seconds':time.time()-started,'limitation':'all=calibration-trained mixture; plain=training excluded calibration but calibration checkpoint selection. Raw outputs only, no full scorer or deployment admission.'}
print(json.dumps(out,indent=2))
