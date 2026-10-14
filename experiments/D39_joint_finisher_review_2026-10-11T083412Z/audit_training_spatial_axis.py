# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — training-only raw spatial decoder axis intervention
"""Locate the axis before and after adding each learned residual, without fitting."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import TABLES, PKG, REPO, metadata, mapped_array, digest
from fec_adapter.diii_d_accuracy.network import Network
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis
sys.path.insert(0, str(REPO / 'fec_harness'))
from fusion_scoring.derive import magnetic_axis


def verify():
    for source, expected in json.loads((HERE / 'training_spatial_axis_sources.json').read_text()).items():
        assert digest(source) == expected, source


def main():
    started = time.time()
    try:
        verify()
        protocol = json.loads((HERE / 'training_spatial_axis_protocol.json').read_text())
        cohort = json.loads((D38 / 'cohort.json').read_text())
        allowed = set(cohort['training'])
        assert len(allowed) == 5954 and not (allowed & (set(cohort['development']) | set(cohort['reserved_confirmation'])))
        meta = metadata('training')
        ids = np.asarray([(meta['bounds'][i]+meta['bounds'][i+1])//2 for i, s in enumerate(meta['stems']) if s in allowed], dtype=np.int64)
        assert len(ids) == 5954
        assert hashlib.sha256(ids.tobytes()).hexdigest() == json.loads((HERE / 'training_anchor_component_audit.json').read_text())['sample_ids_sha256']
        table = TABLES / 'table_training.npz'
        features = mapped_array(table, 'features')[ids].astype(float)
        labels = mapped_array(table, 'labels')[ids].astype(float)
        names = list(meta['label_names'])
        truth = labels[:, [names.index('r_axis'), names.index('z_axis')]]
        assert np.isfinite(truth).all()
        basis = FluxBasis.load(PKG / 'models/flux_basis.npz').truncated(45)
        mask_path = REPO / 'fec_harness/fusion_scoring/masks/d3d_envelope.npz'
        with np.load(mask_path, allow_pickle=False) as mask_file:
            R, Z, mask = mask_file['grid_R'], mask_file['grid_Z'], mask_file['mask_coarse']
        assert mask.shape == (65, 65)
        reports = {}
        arrays = {'training_row_ids': ids, 'training_axis_labels': truth}
        for seed in protocol['seeds']:
            folder = D38 / 'map_fits' / f'spatial_{seed}'
            network = Network.load(folder / 'network.npz')
            coordinates, heads = network.predict(features)
            latent = np.column_stack([heads[f'spatial_{i:02d}'] for i in range(16)])
            decoder = np.load(folder / 'decoder.npy', allow_pickle=False)
            without, with_residual = [], []
            for start in range(0, len(ids), 64):
                stop = min(start+64, len(ids))
                field = basis.decode(coordinates[start:stop, :45])
                residual = (latent[start:stop] @ decoder).reshape(-1, 65, 65)
                for plain, amended in zip(field, field+residual, strict=True):
                    without.append(magnetic_axis(plain, R, Z, 'DIII-D', mask)[:2])
                    with_residual.append(magnetic_axis(amended, R, Z, 'DIII-D', mask)[:2])
            without, with_residual = np.asarray(without), np.asarray(with_residual)
            assert without.shape == with_residual.shape == (5954, 2)
            assert np.isfinite(without).all() and np.isfinite(with_residual).all()
            shift = with_residual-without
            distance = np.sqrt(np.sum(shift**2, axis=1))
            reports[str(seed)] = {
                'raw_axis_shift_R_Z_RMSE_m': np.sqrt(np.mean(shift**2, axis=0)).tolist(),
                'raw_axis_displacement_quantiles_m': np.quantile(distance, [.5, .9, .95, .99, 1.]).tolist(),
                'without_residual_vs_training_axis_labels_R_Z_RMSE_m': np.sqrt(np.mean((without-truth)**2, axis=0)).tolist(),
                'with_residual_vs_training_axis_labels_R_Z_RMSE_m': np.sqrt(np.mean((with_residual-truth)**2, axis=0)).tolist(),
                'frames_with_displacement_gt1cm': int((distance>.01).sum()),
                'frames_with_displacement_gt5cm': int((distance>.05).sum()),
                'network_sha256': digest(folder / 'network.npz'), 'decoder_sha256': digest(folder / 'decoder.npy')}
            arrays[f'without_{seed}'], arrays[f'with_{seed}'] = without, with_residual
            print(json.dumps({'raw_spatial_axis_audit_completed_seed': seed}), flush=True)
        verify()
        archive = HERE / 'training_spatial_axis_arrays.npz'
        assert not archive.exists()
        np.savez_compressed(archive, **arrays)
        with (HERE / 'training_spatial_axis_analysis.json').open('x') as stream:
            json.dump({'status': 'all8_all5954_training_midpoints_raw_spatial_axis_intervention_complete',
                       'reports': reports, 'shots': 5954, 'native_axis_extractor': True,
                       'grid_mask_and_machine_orientation_from_native_scorer': True,
                       'arrays_sha256': digest(archive), 'protocol_sha256': digest(HERE / 'training_spatial_axis_protocol.json'),
                       'single_export_raw_fields_not_deployed_H69_25percent_blend_or_finished_maps': True,
                       'axis_labels_are_original_training_EFIT_labels_not_new_reference_derivations': True,
                       'no358_or357_labels_read': True, 'no_fit_or_model_changes': True,
                       'no_claim_about_rho_band_residual_RMS': True, 'not_candidate_admission': True,
                       'publication': False, 'seconds': time.time()-started}, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (HERE / 'training_spatial_axis_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
