# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — production69 scalar transfer confirmation
"""Compare all declared members on all held-out shots after every fit finishes."""
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from campaign_common import DATA, HERE, PKG, V29, cases, digest, write_json

for path in (PKG/'source/fec_adapter/vendor', PKG/'source/fec_adapter'):
    sys.path.insert(0,str(path))
from fec_adapter.diii_d_accuracy.history import layout_features
from fec_adapter.diii_d_accuracy.network import Network, predict_together
from fec_adapter.shot_reader import aligned_shot, input_columns

spec=importlib.util.spec_from_file_location('production_subframe_features',V29/'subframe_features.py')
subframe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(subframe)
NAMES=('q95','beta_n')
ARMS=('baseline69','joint8','blend69','baseline4','blend4','baseline16','blend16','baseline32','blend32')
STATE={}


def deployment_features(path) -> np.ndarray:
    """Use the package's input-only row reader and production feature functions."""
    columns=input_columns(dict.fromkeys(pq.read_schema(path).names))
    table=pq.read_table(path,columns=columns,use_threads=False)
    assert table.num_rows==1
    row={name:table[name][0].as_py() for name in columns}
    shot=aligned_shot(row,path.stem,np.asarray(row['efit_times'],dtype=float))
    matrix=layout_features(row,shot,{'discharge'})['discharge']
    features=np.concatenate([matrix,subframe.features_of(path).astype(matrix.dtype)],axis=1)
    assert features.shape[1]==582 and np.isfinite(features).all()
    return features


def load_models() -> None:
    """Require all77 successful final recipes and verify every exported checksum."""
    receipt=json.loads((HERE/'fits/receipt.json').read_text())
    assert receipt['status']=='all_77_neural_tasks_terminal' and receipt['successful']==77
    assert len(receipt['tasks'])==77
    networks={}
    for row in receipt['tasks']:
        path=HERE/'fits'/row['name']/'network.npz'
        assert row['status']=='fit_complete' and row['epochs_completed']==60 and digest(path)==row['network_sha256']
        networks[row['name']]=Network.load(path)
    STATE['networks']=networks
    STATE['members']=json.loads((HERE/'diagnostic_members.json').read_text())
    assert len(STATE['members']['69'])==69
    STATE['joint']=[networks[f'joint_seed{seed}'] for seed in range(200,208)]


def predict_case(case: dict) -> dict:
    """Retain raw truth and public double predictions rounded only at archive output."""
    path=HERE/'evaluation/shots'/(case['stem']+'.npz')
    receipt_path=path.with_suffix('.json')
    if receipt_path.exists():
        record=json.loads(receipt_path.read_text())
        assert digest(path)==record['sha256']
        return record
    features=deployment_features(DATA/(case['stem']+'.parquet'))
    assert len(features)==case['frames']
    padded=np.pad(features,((0,0),(0,183)))
    _,joint_labels=predict_together(STATE['joint'],{'discharge':padded})
    joint=np.column_stack([joint_labels[n] for n in NAMES])
    all_predictions={}
    for name in STATE['members']['69']:
        _,labels=STATE['networks'][name].predict(features)
        all_predictions[name]=np.column_stack([labels[n] for n in NAMES])
    values={'joint8':joint}
    for size,names in STATE['members'].items():
        baseline=np.mean([all_predictions[n] for n in names],axis=0)
        values['baseline'+size]=baseline
        values['blend'+size]=.75*baseline+.25*joint
    raw=pq.read_table(DATA/(case['stem']+'.parquet'),columns=['efit_q95','efit_beta_n'],use_threads=False)
    truth=np.column_stack([np.asarray(raw[n][0].as_py(),dtype=float) for n in ('efit_q95','efit_beta_n')])
    assert truth.shape==(case['frames'],2)
    valid=np.isfinite(truth)
    clean=np.where(valid,truth,0.)
    payload={'truth':truth}
    sse=[]
    for arm in ARMS:
        prediction=values[arm].astype(np.float32)
        assert prediction.shape==truth.shape and np.isfinite(prediction).all()
        payload[arm]=prediction
        sse.append(np.where(valid,truth-prediction.astype(float),0.).__pow__(2).sum(axis=0).tolist())
    temporary=path.with_name(case['stem']+'.partial.npz')
    np.savez(temporary,**payload)
    temporary.replace(path)
    record={**case,'count':valid.sum(axis=0).tolist(),'sum':clean.sum(axis=0).tolist(),
            'sumsq':(clean**2).sum(axis=0).tolist(),'sse':sse,'sha256':digest(path)}
    write_json(receipt_path,record)
    print(json.dumps({'confirmation_case_complete':case['stem'],'frames':case['frames']}),flush=True)
    return record


def pooled(records: list, weights: np.ndarray) -> dict:
    """Compute the scorer's pooled finite-truth R2 and clipped scalar contribution."""
    sums=np.asarray([r['sum'] for r in records]); squares=np.asarray([r['sumsq'] for r in records])
    counts=np.asarray([r['count'] for r in records]); sse=np.asarray([r['sse'] for r in records])
    denominator=weights@squares-(weights@sums)**2/(weights@counts)
    assert (denominator>0).all()
    r2=1.-np.einsum('n,nat->at',weights,sse)/denominator
    return {arm:{'r2_q95':float(r2[i,0]),'r2_betaN':float(r2[i,1]),'r2_qb':float(r2[i].mean()),
                 'weighted_scalar_term':.15*max(0.,float(r2[i].mean()))} for i,arm in enumerate(ARMS)}


def main() -> None:
    """Score the prospective 69-member blend; report other sizes as diagnostics."""
    load_models()
    (HERE/'evaluation/shots').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(predict_case,cases()))
    records.sort(key=lambda r:r['stem'])
    assert len(records)==358 and sum(r['frames'] for r in records)==78875
    write_json(HERE/'evaluation/statistics.json',records)
    results=pooled(records,np.ones(358))
    boot={size:[] for size in ('4','16','32','69')}
    rng=np.random.default_rng(996)
    for _ in range(2000):
        weights=np.bincount(rng.integers(0,358,358),minlength=358)
        sampled=pooled(records,weights)
        for size in boot:
            boot[size].append(sampled['blend'+size]['weighted_scalar_term']-sampled['baseline'+size]['weighted_scalar_term'])
    comparisons={}
    for size,samples in boot.items():
        delta=results['blend'+size]['weighted_scalar_term']-results['baseline'+size]['weighted_scalar_term']
        ci=np.percentile(samples,[2.5,97.5]).tolist()
        comparisons[size]={'delta_S_if_map_identical':delta,'ci95':ci,'P_positive':float(np.mean(np.asarray(samples)>0)),
                           'primary':size=='69','diagnostic_only':size!='69'}
    primary=comparisons['69']
    passed=primary['delta_S_if_map_identical']>=1e-5 and primary['ci95'][0]>0
    write_json(HERE/'evaluation/summary.json',{'status':'all_scalar_transfer_comparisons_complete',
        'shots':358,'frames':78875,'results':results,'comparisons':comparisons,
        'prospective_transfer_gate_passes':bool(passed),'D34_prior_gate_unchanged':True,
        'map_outputs_changed':False,'absolute_full_S_computed':False,'historically_exposed_confirmation':True,
        'public_upload_or_submission':False,'production_public_score_improvement_established':False})


if __name__=='__main__':
    main()
