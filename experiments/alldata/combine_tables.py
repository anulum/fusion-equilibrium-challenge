# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data retrain (minus holdout): one training table from training + validation
"""Concatenate the existing lcfs training and validation tables (same basis, same code) into a new
`table_training.npz`; calibration and validation tables and the basis are copied unchanged (the
validation table stays only for the fit's own quote, now in-sample). Entrant decision 2026-09-29."""
import json, shutil, hashlib
from pathlib import Path
import numpy as np
SRC = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge')
OUT = Path('$HOME/fusion-equilibrium-challenge/kaggle_upload/fec-tables-alldata')
OUT.mkdir(parents=True, exist_ok=True)
tr, va = np.load(SRC / 'table_training.npz'), np.load(SRC / 'table_validation.npz')
assert list(tr['label_names']) == list(va['label_names']) and str(tr['feature_layout']) == str(va['feature_layout'])
assert not set(tr['stems'].astype(str)) & set(va['stems'].astype(str))
out = {}
for k in tr.files:
    if k in ('label_names', 'feature_layout'):
        out[k] = tr[k]
    elif k == 'shot_of_row':
        out[k] = np.concatenate([tr[k], va[k] + len(tr['stems'])])
    else:
        out[k] = np.concatenate([tr[k], va[k]])
np.savez(OUT / 'table_training.npz', **out)
for f in ('table_calibration.npz', 'table_validation.npz', 'flux_basis.npz'):
    shutil.copyfile(SRC / f, OUT / f)
rep = {'source': str(SRC), 'training_shots': int(len(tr['stems'])), 'validation_shots_added': int(len(va['stems'])),
       'rows': int(len(out['features'])), 'note': 'training table = lcfs training + lcfs validation; holdout NOT included (guard kept)'}
(OUT / 'combine_report.json').write_text(json.dumps(rep, indent=1))
print(rep)
