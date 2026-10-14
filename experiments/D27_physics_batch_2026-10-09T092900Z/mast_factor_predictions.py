# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — label-free MAST pressure-factor development predictions
"""Apply DIII-D factors without MAST targets or organiser envelope masks."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from campaign_common import HERE, ROOT, digest, write_json
from physics_features import read_inputs, full_inputs, factor_features
from fit_pressure_factor import predict_factor
from fec_adapter.input_features import profile_summary

ARCHIVE = ROOT / 'the internal record'
EXPECTED_SHA = '2febdf3e6baed5cefa4e48cf09a2a5a5d24f6830e1e55c2eee30db2b2acaed51'
DATA = Path('$FEC_DATA/data/mast_public_test')


def predict_case(task):
    index, path = task
    row = read_inputs(path)
    shot, positions, window = full_inputs(row, path.stem)
    valid = np.isfinite(shot.thomson.te) & np.isfinite(shot.thomson.ne) & (shot.thomson.te > 0) & (shot.thomson.ne > 0)
    pressure = np.where(valid, shot.thomson.te * shot.thomson.ne * 1.602176634e-19, np.nan)
    summary = profile_summary(pressure)
    features = factor_features(shot, summary, window)[positions]
    predicted = [predict_factor(HERE / f'pressure_factors/factor_fold{k}.npz', features) for k in range(5)]
    factor = np.mean(predicted, axis=0)
    usable = valid[positions].sum(axis=1) >= 5
    with np.load(ARCHIVE, allow_pickle=False) as stored:
        baseline = stored[f'shot_{index:04d}_betaN'].astype(float)
    assert baseline.shape == factor.shape and np.isfinite(baseline).all()
    alternative = np.where(usable, baseline * factor / 2.25, baseline)
    assert np.isfinite(alternative).all()
    print(json.dumps({'MAST_factor_prediction': index, 'frames': len(baseline)}), flush=True)
    return path.stem, alternative.astype(np.float32), factor.astype(np.float32), usable


def main():
    assert json.loads((HERE / 'pressure_factors/receipt.json').read_text())['status'] == 'all_five_factor_fits_complete'
    assert digest(ARCHIVE) == EXPECTED_SHA
    files = sorted(DATA.glob('*.parquet'))
    assert len(files) == 1206
    out = HERE / 'mast_factor_development'
    out.mkdir(exist_ok=False)
    stems, values, factors, usability, offsets = [], [], [], [], [0]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        for stem, beta, factor, usable in pool.map(predict_case, enumerate(files)):
            stems.append(stem)
            values.append(beta)
            factors.append(factor)
            usability.append(usable)
            offsets.append(offsets[-1] + len(beta))
    file = out / 'scalar_predictions.npz'
    np.savez_compressed(file, stems=np.asarray(stems), offsets=np.asarray(offsets),
                        betaN=np.concatenate(values), pressure_factor=np.concatenate(factors),
                        usable_pressure=np.concatenate(usability))
    write_json(out / 'receipt.json', {'status': 'all_MAST_scalar_development_predictions_complete',
        'shots': len(stems), 'frames': offsets[-1], 'source_MAST_v12_sha256': EXPECTED_SHA,
        'prediction_sha256': digest(file), 'MAST_labels_used': False, 'MAST_score_computed': False,
        'organiser_MAST_envelope_used': False, 'public_submission_archive_built': False,
        'limits': 'Scaling of existing input-only v12 betaN by DIII-D-trained factor/2.25; existing filled/fallback scalar lineage is retained, not a new direct ion-pressure measurement. Missing paired Thomson rows retain baseline.'})


if __name__ == '__main__':
    main()
