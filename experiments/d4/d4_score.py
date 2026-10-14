# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D4: score the label-weight fold-0 OOF maps against V0 (protocol_d4_label_weight_fold0.json)
"""Scores d4/oof_fold0/diii_d_train.npz exactly as R8 scores its variants (r2.score_shot, every 2nd frame) and
compares it with reeval/r8/results/rows_V0_fold0.json by the R8 paired shot bootstrap (20000, seed 930).

    PYTHONPATH=R/fec_adapter python d4_score.py
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
from r8_finisher_retune import bootstrap, score_from, sums  # noqa: E402

FOLD = 0


def main():
    receipt = json.loads((HERE / f'oof_fold{FOLD}' / 'receipt.json').read_text())
    base_receipt = json.loads((HERE.parent / 'kfold' / f'oof_fold{FOLD}' / 'receipt.json').read_text())
    assert receipt['stems'] == base_receipt['stems'], 'the D4 and V0 folds must hold the same shots in the same order'
    archive = HERE / f'oof_fold{FOLD}' / 'diii_d_train.npz'
    cache = HERE / 'results' / f'rows_D4_fold{FOLD}.json'
    cache.parent.mkdir(exist_ok=True)
    if cache.is_file():
        rows = json.loads(cache.read_text())
    else:
        with Pool(3) as pool:
            parts = pool.map(r2.score_shot, [(str(archive), i, s) for i, s in enumerate(receipt['stems'])])
        rows = [dict(r, shot=f'{FOLD}:{i}') for i, part in enumerate(parts) for r in part]
        cache.write_text(json.dumps(rows))
    base = json.loads((R8 / 'results' / f'rows_V0_fold{FOLD}.json').read_text())
    shots = sorted({r['shot'] for r in base})
    tmean = np.nanmean(np.asarray([r['t'] for r in base], float), axis=0)
    check = abs(r2.pooled(base)['S'] - float(score_from(sums(base, shots, tmean), np.ones(len(shots)))))
    assert check < 1e-6, check
    boots = bootstrap(rows, base, shots, tmean, 20000)
    d4, v0 = r2.pooled(rows), r2.pooled(base)
    ds = d4['S'] - v0['S']
    result = {'fold': FOLD, 'frames_scored': len(rows), 'shots': len(shots), 'S_D4': d4['S'], 'S_V0': v0['S'], 'dS': ds,
              'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'P_positive': float((boots > 0).mean()),
              'terms': {k: {'D4': d4[k], 'V0': v0[k]} for k in ('flux_r2', 'qb', 'D', 'consistency')},
              'scalars': {k: {'D4': d4['scalar_r2'][k], 'V0': v0['scalar_r2'][k]} for k in v0['scalar_r2']}}
    result['extend_to_folds_1_4'] = bool(ds >= 2e-5 and result['P_positive'] >= 0.8)
    (HERE / 'results' / f'summary_fold{FOLD}.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
