# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D9-0 gate: the S16 fold networks' heads on every held-out training frame
"""Out-of-fold network heads on the table rows the boosted heads are predicted on (protocol_d9_0_gate.json).

For fold k the 16 S16 V0 networks of kfold/oof_fold{k}/receipt.json (listed with their sha256 in
lists/s16_fold{k}.sha, paths relative to --nets) answer together on the frames of the fold's held-out shots.
Writes the heads, the labels of the same rows and each row's fold and shot, so the evaluation needs no table.

    python d9_nn_heads.py --pkg DIR --table table_training.npz --kfold DIR --lists DIR --nets DIR --out FILE
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TARGETS = ['q95', 'beta_n', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume',
           'x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
FOLDS = 5


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', type=Path, required=True, help='directory holding the fec_adapter package')
    ap.add_argument('--table', type=Path, required=True)
    ap.add_argument('--kfold', type=Path, required=True, help='directory with splits.json and reserved_for_evaluation.json')
    ap.add_argument('--lists', type=Path, required=True)
    ap.add_argument('--nets', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    started = time.time()
    sys.path.insert(0, str(a.pkg))
    from fec_adapter.diii_d_accuracy.network import Network, predict_together
    splits = json.loads((a.kfold / 'splits.json').read_text())
    reserved = set(json.loads((a.kfold / 'reserved_for_evaluation.json').read_text())['stems'])
    with np.load(a.table, allow_pickle=False) as t:
        names = t['label_names'].astype(str).tolist()
        F, L = t['features'], t['labels'][:, [names.index(n) for n in TARGETS]]
        stems, shot_of_row, layout = t['stems'].astype(str), t['shot_of_row'], str(t['feature_layout'])
    assert set(stems) == set(splits['training'])
    stem_fold = np.array([-1 if s in reserved else int(hashlib.sha256(s.encode()).hexdigest(), 16) % FOLDS
                          for s in stems])
    row_fold = stem_fold[shot_of_row]
    pred = np.full(L.shape, np.nan, dtype=np.float32)
    used = {}
    for k in range(FOLDS):
        files = []
        for line in (a.lists / f's16_fold{k}.sha').read_text().splitlines():
            sha, rel = line.split(maxsplit=1)
            assert digest(a.nets / rel) == sha, rel
            files.append(rel)
        networks = [Network.load(a.nets / rel) for rel in files]
        assert len(networks) == 16 and {n.feature_layout for n in networks} == {layout}
        held = row_fold == k
        _, labels = predict_together(networks, {layout: F[held]})
        for j, name in enumerate(TARGETS):
            pred[held, j] = labels[name]
        used[k] = files
        print(json.dumps({'fold': k, 'rows': int(held.sum()), 'seconds': round(time.time() - started)}), flush=True)
    np.savez(a.out, nn=pred, labels=L.astype(np.float32), row_fold=row_fold, shot_of_row=shot_of_row,
             targets=np.asarray(TARGETS))
    Path(str(a.out) + '.json').write_text(json.dumps({
        'table_sha256': digest(a.table), 'networks': used, 'rows': int(len(L)), 'layout': layout,
        'this_script_sha256': digest(__file__), 'seconds': round(time.time() - started)}, indent=1))
    print('NN_HEADS_DONE', flush=True)


if __name__ == '__main__':
    main()
