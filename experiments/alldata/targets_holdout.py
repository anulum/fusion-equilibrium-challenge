# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data refit v28: original-map geometry targets for the 358 holdout shots
"""The same 73-label derivation as targets_extra.py (imported, unchanged) for the holdout part, opened for
training by the entrant on 2026-09-29 ("neblokujme sa"). Writes targets_extra/holdout/*.npz and
targets_extra/holdout_receipt.json; the validation/reserved receipt is not touched."""
import json, multiprocessing
from concurrent.futures import ProcessPoolExecutor
import targets_extra as te
def main():
    splits = json.loads((te.REPO / 'fec_adapter/splits.json').read_text())
    tasks = [(s, 'holdout') for s in sorted(splits['holdout'])]
    assert len(tasks) == 358 and len(te.NAMES) == 73
    (te.OUT / 'holdout').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(5, mp_context=multiprocessing.get_context('spawn'), initializer=te.init) as pool:
        recs = list(pool.map(te.derive, tasks, chunksize=2))
    (te.OUT / 'holdout_receipt.json').write_text(json.dumps({'status': 'complete', 'records': recs, 'label_names': te.NAMES}, indent=1))
    print(len(recs), sum(r['frames'] for r in recs))
if __name__ == '__main__':
    main()
