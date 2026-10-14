# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — verify family tables are label subsets of the lcfs table (training part)
"""For c, h and hx: the family training table equals the lcfs training table with only that
family's label columns (labels NaN-aware equal, every other array bitwise equal)."""
import hashlib, json
from pathlib import Path
import numpy as np
RUNS = Path('$HOME/fusion-equilibrium-challenge/fec_adapter/runs/diii_d_accuracy')
H = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
ref = np.load(RUNS / 'tables_lcfs_discharge/table_training.npz')
names = ref['label_names'].astype(str).tolist()
ref_hash = {k: H(ref[k]) for k in ref.files if k not in ('labels', 'label_names')}
ref_labels = ref['labels']
out = {}
for fam, d in (('c', 'tables_discharge'), ('h', 'tables_shape_discharge'), ('hx', 'tables_xpoint_discharge')):
    t = np.load(RUNS / d / 'table_training.npz')
    ln = t['label_names'].astype(str).tolist()
    same = {k: H(t[k]) == ref_hash[k] for k in t.files if k not in ('labels', 'label_names')}
    lab = bool(np.array_equal(t['labels'], ref_labels[:, [names.index(n) for n in ln]], equal_nan=True))
    out[fam] = {'labels_equal': lab, 'arrays_equal': same, 'all_ok': lab and all(same.values())}
    print(fam, out[fam], flush=True)
Path(__file__).with_name('verify_tables_training_receipt.json').write_text(json.dumps(out, indent=1))
