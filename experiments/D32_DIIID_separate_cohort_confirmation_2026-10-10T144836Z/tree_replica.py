# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — exact scalar GPU tree accumulation on CPU
"""Use native leaf routing and the original GPU float32 accumulation order.

XGBoost v3.4.1 gpu_predictor.cu PredictKernel, n_groups==1: start sum at
zero, add leaves sequentially, then add that sum to the initial base margin.
The ordinary CPU predictor adds each leaf to the base margin individually.
This adapter does not change any tree, iteration range, branch or leaf value.
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from campaign_common import HERE, digest, write_json


def lookup_for(name, booster, record):
    path = HERE / 'tree_lookups' / (name + '.npz')
    receipt = path.with_suffix('.json')
    count = record['best_iteration'] + 1
    config = json.loads(booster.save_config())['learner']
    assert config['gradient_booster']['name'] == 'gbtree'
    assert config['objective']['name'] == 'reg:squarederror'
    assert int(config['learner_model_param']['num_target']) == 1
    base = np.float32(json.loads(config['learner_model_param']['base_score'])[0])
    if receipt.exists():
        value = json.loads(receipt.read_text())
        assert value['tree_sha256'] == record['sha256'] and value['trees'] == count
        assert digest(path) == value['lookup_sha256']
        with np.load(path, allow_pickle=False) as stored:
            assert stored['base_score'] == base
            return stored['leaf_values'], base, value
    leaves = []
    for raw in booster.get_dump(dump_format='json')[:count]:
        pending = [json.loads(raw)]
        values = {}
        while pending:
            node = pending.pop()
            if 'leaf' in node:
                values[node['nodeid']] = np.float32(node['leaf'])
            else:
                pending.extend(node['children'])
        leaves.append(values)
    assert len(leaves) == count
    maximum = max(max(values) for values in leaves)
    matrix = np.full((count, maximum + 1), np.nan, dtype=np.float32)
    for tree, values in enumerate(leaves):
        for node, leaf in values.items():
            matrix[tree, node] = leaf
    np.savez(path, leaf_values=matrix, base_score=base)
    value = {'tree_sha256': record['sha256'], 'trees': count,
             'lookup_sha256': digest(path), 'path': str(path), 'base_score': float(base)}
    write_json(receipt, value)
    return matrix, base, value


def predict_like_gpu(booster, inputs, matrix, base):
    count = matrix.shape[0]
    leaves = booster.predict(xgb.DMatrix(inputs, nthread=1),
                             pred_leaf=True, iteration_range=(0, count))
    assert leaves.shape == (len(inputs), count)
    indices = leaves.astype(np.int32)
    assert np.array_equal(indices, leaves)
    assert (indices >= 0).all() and (indices < matrix.shape[1]).all()
    contribution = matrix[np.arange(count)[None, :], indices]
    assert np.isfinite(contribution).all()
    # cumsum preserves scalar tree order in float32; adding base after the sum
    # reproduces the GPU kernel instead of changing its numerical tolerance.
    return np.cumsum(contribution, axis=1, dtype=np.float32)[:, -1] + base
