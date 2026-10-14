# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D9: the networks' feature matrix of every shot of a split, as the predictor builds it
"""Write the 'discharge' feature matrix of a split's shots exactly as predict_diii_d.finished_maps reads it.

    python d9_test_features.py --pkg DIR --data-root DIR --split diii_d_public_test --out FILE
        [--check-table table_training.npz --check 3] [--workers N]

The boosted heads are predicted from this file. With --check-table the same code is first run on a few training
shots and compared with the rows of the fitted table (as float32), which is what the trees were trained on.
Only predictor inputs are read (read_row), never a label.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _entrypoint(pkg):
    for p in (Path(pkg) / 'entrypoints', Path(pkg) / 'source/fec_adapter/vendor', Path(pkg) / 'source/fec_adapter'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import predict_diii_d as pd
    return pd


def features_of(task):
    pkg, path = task
    pd = _entrypoint(pkg)
    from fec_adapter.diii_d_accuracy.history import DISCHARGE_LAYOUT
    path = Path(path)
    row = pd.read_row(path)
    shot = pd.aligned_shot(row, path.stem, np.asarray(row['efit_times'], dtype=np.float64))
    return np.asarray(pd.layout_features(row, shot, {DISCHARGE_LAYOUT})[DISCHARGE_LAYOUT], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--split', default='diii_d_public_test')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check-table', type=Path, default=None)
    ap.add_argument('--check', type=int, default=3)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    started = time.time()
    pd = _entrypoint(a.pkg)
    report = {'split': a.split}
    if a.check_table:
        with np.load(a.check_table, allow_pickle=False) as t:
            stems, shot_of_row = t['stems'].astype(str), t['shot_of_row']
            picks = np.linspace(0, len(stems) - 1, a.check).astype(int)
            rows = {int(i): t['features'][shot_of_row == i] for i in picks}
        worst, worst_relative, close = 0.0, 0.0, True
        for i, table_rows in rows.items():
            mine = features_of((str(a.pkg), str(a.data_root / 'diii_d_train' / f'{stems[i]}.parquet')))
            assert mine.shape == table_rows.shape, (stems[i], mine.shape, table_rows.shape)
            gap = np.abs(mine.astype(np.float64) - table_rows)
            worst = max(worst, float(gap.max()))
            worst_relative = max(worst_relative, float(np.max(gap / np.maximum(np.abs(table_rows), 1.0))))
            close = close and bool(np.allclose(mine, table_rows, rtol=1e-5, atol=1e-5))
        report['table_check'] = {'shots': [str(stems[i]) for i in rows], 'max_abs_difference': worst,
                                 'max_difference_relative_to_max_abs_value_or_one': worst_relative,
                                 'allclose_rtol_atol_1e-5': close}
        print(json.dumps(report), flush=True)
        # The table was written on the workstation; another machine rounds the same arithmetic differently.
        assert close, 'the predictor features differ from the fitted table beyond rounding'
    files = [str(f) for f in pd.split_files(a.data_root, a.split, None)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        blocks = list(pool.map(features_of, [(str(a.pkg), f) for f in files], chunksize=8))
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    features = np.concatenate(blocks)
    assert np.isfinite(features).all()
    np.savez(a.out, features=features, offsets=offsets, stems=np.asarray([Path(f).stem for f in files]))
    with a.out.open('rb') as stream:
        report.update({'shots': len(files), 'frames': int(len(features)), 'width': int(features.shape[1]),
                       'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
                       'seconds': round(time.time() - started)})
    Path(str(a.out) + '.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
