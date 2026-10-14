# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D4c: score the lcfs-label-weight OOF folds against V0 (protocol_d4c_lcfs_label_weight.json)
"""Scores d4c/oof_fold{k}/diii_d_train.npz as R8 scores its variants and compares the pooled folds with
reeval/r8/results/rows_V0_fold{k}.json by a paired shot bootstrap (20000, seed 944).

    PYTHONPATH=R/fec_adapter python d4c_score.py 0 1
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R8 = HERE.parent / 'reeval' / 'r8'
sys.path.insert(0, str(HERE.parent / 'reeval'))
sys.path.insert(0, str(R8))
import r2_stage_attribution as r2  # noqa: E402
from r8_finisher_retune import score_from, sums  # noqa: E402


def main():
    folds = [int(a) for a in sys.argv[1:]]
    rows, base = [], []
    (HERE / 'results').mkdir(exist_ok=True)
    for k in folds:
        receipt = json.loads((HERE / f'oof_fold{k}' / 'receipt.json').read_text())
        base_receipt = json.loads((HERE.parent / 'kfold' / f'oof_fold{k}' / 'receipt.json').read_text())
        assert receipt['stems'] == base_receipt['stems'], 'the D4c and V0 folds must hold the same shots in the same order'
        cache = HERE / 'results' / f'rows_D4c_fold{k}.json'
        if cache.is_file():
            part = json.loads(cache.read_text())
        else:
            archive = HERE / f'oof_fold{k}' / 'diii_d_train.npz'
            with Pool(3) as pool:
                parts = pool.map(r2.score_shot, [(str(archive), i, s) for i, s in enumerate(receipt['stems'])])
            part = [dict(r, shot=f'{k}:{i}') for i, p in enumerate(parts) for r in p]
            cache.write_text(json.dumps(part))
        rows += part
        base += json.loads((R8 / 'results' / f'rows_V0_fold{k}.json').read_text())
    shots = sorted({r['shot'] for r in base})
    tmean = np.nanmean(np.asarray([r['t'] for r in base], float), axis=0)
    zn, zb = sums(rows, shots, tmean), sums(base, shots, tmean)
    rng = np.random.default_rng(944)
    boots = np.empty(20000)
    for b in range(20000):
        w = np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)).astype(float)
        boots[b] = score_from(zn, w) - score_from(zb, w)
    new, v0 = r2.pooled(rows), r2.pooled(base)
    ds = new['S'] - v0['S']
    result = {'folds': folds, 'frames_scored': len(rows), 'shots': len(shots), 'S_D4c': new['S'], 'S_V0': v0['S'], 'dS': ds,
              'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'P_positive': float((boots > 0).mean()),
              'terms': {k: {'D4c': new[k], 'V0': v0[k]} for k in ('flux_r2', 'qb', 'D', 'consistency')},
              'scalars': {k: {'D4c': new['scalar_r2'][k], 'V0': v0['scalar_r2'][k]} for k in v0['scalar_r2']},
              'screen_passes': bool(ds >= 2e-5 and (boots > 0).mean() >= 0.8),
              'adoption_passes': bool(len(folds) == 5 and ds >= 2e-5 and (boots > 0).mean() >= 0.9)}
    name = 'summary_folds_' + ''.join(str(k) for k in folds) + '.json'
    (HERE / 'results' / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
