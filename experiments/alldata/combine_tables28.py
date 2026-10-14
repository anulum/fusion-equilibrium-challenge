# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data v28: the local training table = v27 all-data table + holdout table
"""Same concatenation as the fec-alldata28 kernels (training, validation, then holdout); written locally for the
v28 readout refit. Calibration and basis copied unchanged. Entrant go for the holdout 2026-09-29."""
import hashlib, json, shutil
from pathlib import Path
import numpy as np
SRC = Path('$HOME/fusion-equilibrium-challenge/kaggle_upload/fec-tables-alldata')
HO = Path(__file__).resolve().parent / 'tables_holdout/table_holdout.npz'
OUT = Path('$HOME/fusion-equilibrium-challenge/kaggle_upload/fec-tables-alldata28')
OUT.mkdir(parents=True, exist_ok=True)
tr, ho = np.load(SRC / 'table_training.npz'), np.load(HO)
assert set(tr.files) == set(ho.files) and list(tr['label_names']) == list(ho['label_names'])
assert str(tr['feature_layout']) == str(ho['feature_layout']) and not set(tr['stems'].astype(str)) & set(ho['stems'].astype(str))
out = {}
for k in tr.files:
    if k in ('label_names', 'feature_layout'): out[k] = tr[k]
    elif k == 'shot_of_row': out[k] = np.concatenate([tr[k], ho[k] + len(tr['stems'])])
    else: out[k] = np.concatenate([tr[k], ho[k]])
np.savez(OUT / 'table_training.npz', **out)
for f in ('table_calibration.npz', 'table_validation.npz', 'flux_basis.npz'):
    shutil.copyfile(SRC / f, OUT / f)
rep = {'training_shots': int(len(out['stems'])), 'rows': int(len(out['features'])),
       'array_sha256': {k: hashlib.sha256(np.ascontiguousarray(out[k]).tobytes()).hexdigest() for k in out}}
(OUT / 'combine_report.json').write_text(json.dumps(rep, indent=1)); print(rep['training_shots'], rep['rows'])
