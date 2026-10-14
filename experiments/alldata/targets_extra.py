# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data refit: original-map geometry targets for validation + reserved18
"""The 73 original-map geometry labels (same derivation as second_agent-recovery build_original_map_geometry_targets.py:
scorer reading of the raw float32 map; shape values, boundary saddles, 64 radii) for the shots the existing
cache left out: all 357 validation shots and the 18 reserved training shots. Entrant decision 2026-09-29."""
import hashlib, json, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy import contour_edit, x_point
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.diii_d_accuracy.shape_labels import SHAPE_LABELS
REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$FEC_DATA/data/diii_d_train')
OUT = Path(__file__).resolve().parent / 'targets_extra'
NAMES = [*SHAPE_LABELS, *x_point.LABELS, *contour_edit.label_names()]
READER = None
def init():
    global READER; READER = Reader('DIII-D')
def derive(task):
    stem, part = task
    t = pq.read_table(DATA / f'{stem}.parquet', columns=['efit_times', 'efit_psirz'])
    times = np.asarray(t['efit_times'][0].as_py(), float)
    maps = np.asarray(t['efit_psirz'][0].as_py(), dtype=np.float32)
    labels = np.full((len(times), len(NAMES)), np.nan, dtype=np.float32)
    for f, stored in enumerate(maps):
        flux = np.asarray(stored, float); r = READER.read(flux)
        labels[f, :5] = [r.values[n] for n in SHAPE_LABELS.values()]
        if r.contour is None: continue
        s = x_point.boundary_saddles(READER, flux, r)
        for i, end in enumerate(x_point.ENDS):
            if s[end] is not None: labels[f, 5 + 2 * i:7 + 2 * i] = s[end].position
        labels[f, 9:] = contour_edit.radii(r.contour, r.axis)
    out = OUT / part / f'{stem}.npz'
    np.savez_compressed(out, labels=labels, label_names=np.asarray(NAMES), times_ms=times, stem=np.asarray(stem))
    return {'stem': stem, 'part': part, 'frames': int(len(times)), 'output_path': str(out),
            'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
def main():
    splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
    reserved = json.loads((REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems']
    tasks = [(s, 'validation') for s in sorted(splits['validation'])] + [(s, 'training') for s in sorted(reserved) if s in splits['training']]
    assert len(tasks) == 375 and len(NAMES) == 73
    for p in ('validation', 'training'): (OUT / p).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool:
        recs = list(pool.map(derive, tasks, chunksize=2))
    (OUT / 'complete_receipt.json').write_text(json.dumps({'status': 'complete', 'records': recs, 'label_names': NAMES}, indent=1))
    print(len(recs), sum(r['frames'] for r in recs))
if __name__ == '__main__':
    main()
