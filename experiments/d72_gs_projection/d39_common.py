# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D39 finisher study: shared paths, cache and official scoring
"""Paths on ML350, the cached raw answers of the held-out 69, and the organisers' per-shot score."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAS = Path('$SAS')
X = SAS / 'ml350/runs/d39_finisher_20261011'
PKG = SAS / 'ml350/runs/legacy_tree_first_agent_20261005/deploy/pkg'
V29 = SAS / 'ml350/runs/ensembles_20261007/v29'
R = Path.home() / 'fusion-equilibrium-challenge'
TREE_WEIGHTS = {'li_scored': .26, 'r_axis': .14, 'kappa': .09, 'x_lower_r': .16}
JOINT_SHARE = .25

for _path in (PKG / 'source/fec_adapter/vendor', PKG / 'source/fec_adapter', PKG / 'entrypoints',
              X / 'd32/vendor', R / 'fec_harness'):
    sys.path.insert(0, str(_path))


def remap(path: str) -> str:
    """Return a workstation path of the shared SAS mount as this machine's path."""
    return path.replace('$HOME/remote_data_sas', '$SAS_ROOT')


def digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cases() -> list[dict]:
    """Return the 358 development shots of D38, fit-disjoint from every network used here."""
    listed = json.loads((X / 'd32/manifest.json').read_text())['cases']
    development = sorted(json.loads((X / 'd38/cohort.json').read_text())['development'])
    assert [c['stem'] for c in listed] == development and len(listed) == 358
    return [{**c, 'path': remap(c['path'])} for c in listed]


def statistics() -> tuple[list, dict, tuple]:
    rows = json.loads((X / 'd32/reference_statistics.json').read_text())
    total = {key: sum(np.asarray(row[key]) for row in rows) for key in rows[0]}
    means = (total['psi_sum'] / total['psi_n'], total['scal_sum'] / np.maximum(total['scal_n'], 1),
             total['cons_sum'] / np.maximum(total['cons_n'], 1))
    return rows, total, means


def load_cache(stem: str) -> SimpleNamespace:
    with np.load(X / 'cache' / (stem + '.npz'), allow_pickle=False) as stored:
        return SimpleNamespace(**{name: stored[name] for name in stored.files})


def reference_of(stem: str):
    return pickle.loads((X / 'd32/references' / (stem + '.pkl')).read_bytes())


def score_maps(scorer, means, truth: dict, reference: tuple, prediction: dict):
    """The organisers' unmodified per-shot score, with the per-frame distances kept (as in D37)."""
    radius, height, coarse, mask = scorer._grid()
    contours, derived, defined, rgeo = reference
    eligible = iter(k for k in range(len(contours)) if contours[k] is not None or defined[k].any())
    distances = np.full(len(contours), np.nan)
    values = np.full_like(derived, np.nan, dtype=float)
    active = [-1]
    local = scorer.local_score
    extract, hausdorff, derive = local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame

    def extracted(*args, **kwargs):
        index = next(eligible)
        active[0] = index
        result = extract(*args, **kwargs)
        if result[0] is None and contours[index] is not None:
            distances[index] = 1.
        return result

    def distance(*args, **kwargs):
        result = hausdorff(*args, **kwargs)
        distances[active[0]] = min(1., result / rgeo if rgeo > 0 else 1.)
        return result

    def derived_frame(*args, **kwargs):
        result = derive(*args, **kwargs)
        values[active[0]] = [result[name] for name in local.CONS_SCALARS]
        return result

    local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame = extracted, distance, derived_frame
    try:
        result = local.score_shot(truth, reference, prediction, radius, height, coarse, mask, 1., means)
    finally:
        local.extract_lcfs_with_sign, local.symmetric_hausdorff, local.derive_frame = extract, hausdorff, derive
    assert next(eligible, None) is None
    if result['shot_dlcfs'] is not None:
        assert abs(np.nanmean(distances) - result['shot_dlcfs']) < 1e-14
    return result, distances, values


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value
