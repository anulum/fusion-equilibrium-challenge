# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D9-0b: tree settings A and B compared by their nested blend with the network heads
"""Per target of setting B: out-of-fold R2 of the trees and of the nested blend, for A and for B (protocol_d9_0b).

    python d9_b_eval.py --nn nn_oof.npz --a kaggle_out --b kaggle_out_b --out d9_b_result.json

Same rows, network heads and nested weight as d9_gate_eval.py. The setting with the higher blend R2 is the one the
deployment uses for that target; the all-fold least-squares weight of each setting is reported for it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FOLDS = 5


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def trees(directory, row_fold, shape):
    out, names = np.full(shape, np.nan), None
    for k in range(FOLDS):
        with np.load(directory / f'gbm_oof_fold{k}.npz', allow_pickle=False) as z:
            names = z['targets'].astype(str).tolist()
            assert np.array_equal(z['rows'], np.flatnonzero(row_fold == k))
            out[z['rows'], :len(names)] = z['pred']
    return out, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nn', type=Path, required=True)
    ap.add_argument('--a', type=Path, required=True)
    ap.add_argument('--b', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.nn, allow_pickle=False) as z:
        nn, labels, row_fold = z['nn'].astype(np.float64), z['labels'].astype(np.float64), z['row_fold']
        targets = z['targets'].astype(str).tolist()
    tree_a, names_a = trees(a.a, row_fold, nn.shape)
    tree_b, names_b = trees(a.b, row_fold, nn.shape)
    result = {}
    for name in names_b:
        j = targets.index(name)
        y, p = labels[:, j], nn[:, j]
        ok = np.isfinite(y) & (row_fold >= 0)
        entry = {'frames': int(ok.sum()), 'nn_r2': r2(y[ok], p[ok])}
        for key, source, names in (('A', tree_a, names_a), ('B', tree_b, names_b)):
            g = source[:, names.index(name)]
            assert np.isfinite(g[ok]).all()
            blend, weights = np.full(len(y), np.nan), []
            for k in range(FOLDS):
                fit, use = ok & (row_fold != k), ok & (row_fold == k)
                d = g[fit] - p[fit]
                weights.append(float(np.clip(np.dot(d, y[fit] - p[fit]) / np.dot(d, d), 0.0, 1.0)))
                blend[use] = p[use] + weights[-1] * (g[use] - p[use])
            d = g[ok] - p[ok]
            entry[key] = {'tree_r2': r2(y[ok], g[ok]), 'blend_r2': r2(y[ok], blend[ok]),
                          'error_correlation': float(np.corrcoef(y[ok] - p[ok], y[ok] - g[ok])[0, 1]),
                          'nested_weights': weights,
                          'all_fold_weight': float(np.clip(np.dot(d, y[ok] - p[ok]) / np.dot(d, d), 0.0, 1.0))}
        entry['chosen'] = 'B' if entry['B']['blend_r2'] > entry['A']['blend_r2'] else 'A'
        result[name] = entry
    a.out.write_text(json.dumps(result, indent=1))
    for name, e in result.items():
        print(f"{name:10s} NN {e['nn_r2']:.5f} | A tree {e['A']['tree_r2']:.5f} blend {e['A']['blend_r2']:.5f} "
              f"w {e['A']['all_fold_weight']:.3f} | B tree {e['B']['tree_r2']:.5f} blend {e['B']['blend_r2']:.5f} "
              f"w {e['B']['all_fold_weight']:.3f} corr {e['B']['error_correlation']:.3f} -> {e['chosen']}")


if __name__ == '__main__':
    main()
