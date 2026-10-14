# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — sole locked joint candidate confirmation prediction
"""Replace shared trees and return only the locked candidate and comparator."""
from pathlib import Path
import json

import numpy as np

import joint_prediction as jp
from window_cache import mapped_array

STATE = {}


def initialize(original_trees=False):
    lock_path = jp.HERE / 'concrete_confirmation_candidate_lock.json'
    lock = json.loads(lock_path.read_text())
    assert lock['locked_before_reserved_candidate_scoring'] and lock['one_candidate_only']
    assert lock['winner'] in ('d38_finalist', 'd39_finalist', 'composition')
    jp.initialize()
    STATE.update(lock)
    STATE['lock_sha256'] = jp.cp.digest(lock_path)
    if original_trees:
        return
    assert json.loads((jp.D38 / 'confirmation_trees/receipt.json').read_text())['status'] == 'all_four_eligible_shared_confirmation_trees_complete'
    trees = {}
    for name in jp.cp.TREE_WEIGHTS:
        folder = jp.D38 / 'confirmation_trees' / name
        receipt = json.loads((folder / 'fit_report.json').read_text())
        assert receipt['both358_development_and357_confirmation_excluded']
        assert jp.cp.digest(folder / 'tree.ubj') == receipt['model_sha256']
        assert jp.cp.digest(folder / 'lookup.npz') == receipt['lookup_sha256']
        booster = jp.cp.STATE.xgb.Booster(); booster.load_model(folder / 'tree.ubj')
        booster.set_param({'device': 'cpu', 'nthread': 1})
        with np.load(folder / 'lookup.npz', allow_pickle=False) as saved:
            trees[name] = booster, saved['leaf_values'].copy(), saved['base_score'].copy()
    jp.cp.STATE.trees = trees
    if lock['d38_winner'] is not None and lock['d38_winner'].startswith('scalar_'):
        alignment = json.loads((jp.D38 / 'temporal/validation_alignment.json').read_text())
        bounds = alignment['offsets']
        STATE['bounds'] = {stem: (bounds[i], bounds[i + 1]) for i, stem in enumerate(alignment['stems'])}
        STATE['windows'] = mapped_array(jp.D38 / 'temporal/windows/rawwindow_validation.npz', 'features')


def predict(path: Path):
    raw = None
    if path.stem in STATE.get('bounds', {}):
        start, end = STATE['bounds'][path.stem]
        raw = STATE['windows'][start:end]
    values, shot = jp.predict(path, STATE['d38_winner'], STATE['d39_change'], temporal_raw=raw)
    return {arm: values[arm] for arm in (jp.BASE, STATE['winner'])}, shot
