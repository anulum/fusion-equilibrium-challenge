# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D53 gate: do the most similar other shots share a shot's head offset?
"""Shot-level offsets of the out-of-fold heads, predicted from the nearest other shots (protocol_d53.json).

Descriptor of a shot: mean and standard deviation over its frames of the 367 table features (inputs only),
standardised over shots, 64 principal directions. For a shot of fold k the candidates are the shots of the other
four folds; its predicted offset per head is the mean, over the K nearest candidates, of their mean out-of-fold
residual (label - head). Output per fold: head + predicted offset, in the layout d9_gate_eval.py reads, whose
nested weight then shrinks the correction.

    python sibling_offsets.py TABLE.npz NN_OOF.npz OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

FOLDS, DIRECTIONS, PRIMARY, OTHERS = 5, 64, 5, (1, 3, 10, 20)


def main():
    table, heads_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with np.load(heads_file, allow_pickle=False) as z:
        nn, labels, row_fold, shot = z['nn'].astype(float), z['labels'].astype(float), z['row_fold'], z['shot_of_row']
        targets = z['targets'].astype(str).tolist()
    with np.load(table, allow_pickle=False) as t:
        assert np.array_equal(t['shot_of_row'], shot)
        features = t['features'].astype(np.float64)
    shots = int(shot.max()) + 1
    count = np.bincount(shot, minlength=shots).astype(float)
    mean = np.stack([np.bincount(shot, weights=features[:, j], minlength=shots) for j in range(features.shape[1])], axis=1) / count[:, None]
    square = np.stack([np.bincount(shot, weights=features[:, j] ** 2, minlength=shots) for j in range(features.shape[1])], axis=1) / count[:, None]
    del features
    descriptor = np.concatenate([mean, np.sqrt(np.maximum(square - mean ** 2, 0.0))], axis=1)
    fold_of_shot = np.full(shots, -1)
    fold_of_shot[shot[row_fold >= 0]] = row_fold[row_fold >= 0]
    used = fold_of_shot >= 0
    scale = descriptor[used].std(axis=0)
    keep = scale > 1e-12
    z = (descriptor[:, keep] - descriptor[used][:, keep].mean(axis=0)) / scale[keep]
    _, _, vt = np.linalg.svd(z[used], full_matrices=False)
    code = z @ vt[:DIRECTIONS].T
    residual = np.where(np.isfinite(labels), labels - nn, 0.0)
    valid = np.isfinite(labels).astype(float)
    n_valid = np.stack([np.bincount(shot, weights=valid[:, j], minlength=shots) for j in range(len(targets))], axis=1)
    offset = np.stack([np.bincount(shot, weights=residual[:, j], minlength=shots) for j in range(len(targets))], axis=1) / np.maximum(n_valid, 1)
    has = n_valid > 0
    report = {'shots': int(used.sum()), 'directions': DIRECTIONS, 'nearest_distance_quantiles': None, 'by_k': {}}
    for k_near in (PRIMARY, *OTHERS):
        predicted = np.zeros_like(offset)
        nearest = np.zeros(shots)
        for k in range(FOLDS):
            mine = np.flatnonzero(fold_of_shot == k)
            other = np.flatnonzero(used & (fold_of_shot != k))
            d2 = (code[mine] ** 2).sum(1)[:, None] + (code[other] ** 2).sum(1)[None, :] - 2 * code[mine] @ code[other].T
            order = np.argpartition(d2, k_near, axis=1)[:, :k_near] if k_near < len(other) else np.argsort(d2, axis=1)
            neighbours = other[order]
            nearest[mine] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
            weight = has[neighbours].astype(float)
            predicted[mine] = (offset[neighbours] * weight).sum(axis=1) / np.maximum(weight.sum(axis=1), 1)
        corr = {name: float(np.corrcoef(predicted[used & has[:, j], j], offset[used & has[:, j], j])[0, 1]) for j, name in enumerate(targets)}
        report['by_k'][str(k_near)] = {'correlation_of_predicted_and_true_shot_offset': corr}
        if k_near == PRIMARY:
            report['nearest_distance_quantiles'] = np.quantile(nearest[used], [.05, .25, .5, .75, .95]).tolist()
        folder = out / f'k{k_near}'
        folder.mkdir(parents=True, exist_ok=True)
        for k in range(FOLDS):
            rows = np.flatnonzero(row_fold == k)
            np.savez(folder / f'gbm_oof_fold{k}.npz', pred=(nn[rows] + predicted[shot[rows]]).astype(np.float32), rows=rows,
                     targets=np.asarray(targets))
    (out / 'sibling_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
