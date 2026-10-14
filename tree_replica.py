# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — exact scalar GPU tree accumulation on CPU
"""Portable D32 accumulation helper; exact original function retained."""
import numpy as np
import xgboost as xgb


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
