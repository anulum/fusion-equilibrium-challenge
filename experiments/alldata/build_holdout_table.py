# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data retrain: the holdout part's table (entrant-approved opening)
"""Build `table_holdout.npz` in the lcfs-table format, for the entrant-approved retrain on every labelled
DIII-D shot (entrant 2026-09-29: "Pretrenuj."; then, told what the holdout is, explicitly: "neblokujme
sa ... chceme čo najlepšie výsledky, ... v súlade s ich pravidlami"). The holdout is our own internal
5 % split of the organisers' labelled training release, not challenge test data.

The package refuses holdout shots by design; this script lifts that guard at module level (so the
spawned workers inherit it) and nowhere else. It runs the repository's own steps, unchanged:
extraction (aligned shot + labels), discharge-record extraction, table rows on the EXISTING lcfs
flux basis (no new basis), then the shape, X-point and contour label steps in that order.

Control: five training shots go through the identical steps; their rows must equal the rows of
the existing training table bitwise (features, coordinates, labels) or the holdout table is
refused as built by a different code state.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import fec_adapter.split as split
from fec_adapter.diii_d_accuracy import extract, shape_labels, table

split.refuse_holdout = lambda stem: None          # entrant-approved opening of the holdout part
extract.pending = lambda files, out: ([p for p in files if not (out / f'{p.stem}.npz').exists()], 0)

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$FEC_DATA/data/diii_d_train')
LCFS = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge'
WORK = Path('$DISK/_offroot_cache/fec_alldata_work')
OUT = HERE / 'tables_holdout'


def main():
    started = time.time()
    splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
    holdout = sorted(splits['holdout'])
    assert len(holdout) == 358 and all(split.bucket_of(s) == 'holdout' for s in holdout)
    control = sorted(splits['training'], key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:5]
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'entrant_decision': '2026-09-29 "Pretrenuj." + explicit go for the holdout ("neblokujme sa")',
              'holdout_shots': len(holdout), 'control_training_shots': control}
    for tag, stems in (('control', control), ('holdout', holdout)):
        files = [DATA / f'{s}.parquet' for s in stems]
        ex, hi = WORK / f'extracted_{tag}', WORK / f'history_{tag}'
        extract.extract_all(files, ex, 6)
        extract.extract_all(files, hi, 6, history=True)
        base = WORK / f'table_{tag}_base.npz'
        found = sorted(ex.glob('*_shot_*.npz'))
        report[f'{tag}_rows'] = table.build_part(found, LCFS / 'flux_basis.npz', base, 6, hi)
        a = WORK / f'table_{tag}_shape.npz'
        b = WORK / f'table_{tag}_xpoint.npz'
        c = OUT / f'table_{tag}.npz'
        shape_labels.labelled_table(base, LCFS / 'flux_basis.npz', a, 'DIII-D', 6)
        shape_labels.labelled_table(a, LCFS / 'flux_basis.npz', b, 'DIII-D', 6, x_points=True)
        shape_labels.labelled_table(b, LCFS / 'flux_basis.npz', c, 'DIII-D', 6, contour=True)
        print(json.dumps({'done': tag, 'minutes': round((time.time() - started) / 60, 1)}), flush=True)
    # bitwise control against the existing training table
    with np.load(LCFS / 'table_training.npz') as ref, np.load(OUT / 'table_control.npz') as new:
        rs, ns = ref['stems'].astype(str), new['stems'].astype(str)
        rc = np.bincount(ref['shot_of_row'], minlength=len(rs)); rst = np.concatenate(([0], np.cumsum(rc)))
        nc = np.bincount(new['shot_of_row'], minlength=len(ns)); nst = np.concatenate(([0], np.cumsum(nc)))
        where = {s: i for i, s in enumerate(rs)}
        assert list(new['label_names']) == list(ref['label_names'])
        check = {}
        for j, s in enumerate(ns):
            i = where[s]
            sl_r, sl_n = slice(rst[i], rst[i + 1]), slice(nst[j], nst[j + 1])
            check[s] = {k: bool(np.array_equal(ref[k][sl_r], new[k][sl_n], equal_nan=True))
                        for k in ('features', 'coordinates', 'labels')}
        report['control_bitwise'] = check
        report['control_all_equal'] = all(all(v.values()) for v in check.values())
    report['seconds'] = time.time() - started
    (OUT / 'build_report.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({'control_all_equal': report['control_all_equal']}), flush=True)
    if not report['control_all_equal']:
        (OUT / 'REFUSED_code_state_differs').write_text(json.dumps(check, indent=1))
        sys.exit(2)


if __name__ == '__main__':
    main()
