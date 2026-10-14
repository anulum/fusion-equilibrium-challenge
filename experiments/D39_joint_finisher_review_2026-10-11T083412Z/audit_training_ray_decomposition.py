# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — training-only ray target decomposition
"""Measure projection aliasing without fitting or accessing held-out labels."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'
sys.path.insert(0, str(D38))
from campaign_common import TABLES, PKG, metadata, mapped_array, digest
from fec_adapter.diii_d_accuracy.network import Network
from fec_adapter.diii_d_accuracy.flux_basis import FluxBasis


def sample(channels, points):
    xy = np.clip(np.nan_to_num(points), 0., 64.)
    low = np.clip(np.floor(xy).astype(np.int64), 0, 63)
    fraction = xy - low
    x, y = low[:, :, 0], low[:, :, 1]
    dx, dy = fraction[:, :, 0:1], fraction[:, :, 1:2]
    table = channels.T
    return ((1-dx)*(1-dy)*table[y*65+x] + dx*(1-dy)*table[y*65+x+1]
            + (1-dx)*dy*table[(y+1)*65+x] + dx*dy*table[(y+1)*65+x+1])


def verify():
    for source, expected in json.loads((HERE / 'training_ray_audit_sources.json').read_text()).items():
        assert digest(source) == expected, source


def main():
    started = time.time()
    try:
        verify()
        protocol = json.loads((HERE / 'training_ray_audit_protocol.json').read_text())
        cohort = json.loads((D38 / 'cohort.json').read_text())
        allowed = set(cohort['training'])
        assert len(allowed) == 5954
        assert not (allowed & (set(cohort['development']) | set(cohort['reserved_confirmation']) | set(cohort['calibration'])))
        meta = metadata('training')
        ids, point_ids, offset, stems = [], [], 0, []
        for i, stem in enumerate(meta['stems']):
            if stem in allowed:
                lo, hi = meta['bounds'][i:i+2]
                ids.append((lo+hi)//2)
                point_ids.append(offset+(hi-lo)//2)
                offset += hi-lo
                stems.append(stem)
        assert len(ids) == 5954 and set(stems) == allowed
        ids = np.asarray(ids, dtype=np.int64)
        original = json.loads((HERE / 'training_anchor_component_audit.json').read_text())
        assert hashlib.sha256(ids.tobytes()).hexdigest() == original['sample_ids_sha256']
        table = TABLES / 'table_training.npz'
        features = mapped_array(table, 'features')[ids].astype(float)
        teacher = mapped_array(table, 'coordinates')[ids, :128].astype(float)
        labels = mapped_array(table, 'labels')[ids].astype(float)
        points = np.load(D38 / 'supervision/training_points.npy', mmap_mode='r', allow_pickle=False)[point_ids].astype(float)
        assert points.shape == (5954, 43, 2) and np.isfinite(teacher).all()
        directions = np.asarray(FluxBasis.load(PKG / 'models/flux_basis.npz').directions[:128]).reshape(128, 4225)
        wanted = np.column_stack([teacher[:, :64], labels])
        weights = np.r_[np.ones(64), np.full(78, .05)]
        reports = {}
        for seed in protocol['seeds']:
            reports[str(seed)] = {}
            for family in protocol['families']:
                path = D38 / 'map_fits' / f'{family}_{seed}' / 'network.npz'
                model = Network.load(path)
                receipt = json.loads(path.with_name('fit_report.json').read_text())
                assert digest(path) == receipt['network_sha256']
                coordinates, heads = model.predict(features)
                prediction = np.column_stack([coordinates, *[heads[n] for n in model.label_names]])
                assert np.isfinite(prediction).all()
                assert np.unique(model.output_scale[:64]).size == 1
                sigma = float(model.output_scale[0])
                valid_anchor = np.isfinite(wanted)
                anchor = np.where(valid_anchor, (prediction[:, :142]-np.nan_to_num(wanted))/model.output_scale[:142], 0.)
                per_output = np.mean(anchor**2*weights, axis=0)
                sums = np.zeros(5)
                valid_shots = 0
                decoder = None
                if family == 'spatial':
                    decoder_path = path.with_name('decoder.npy')
                    assert digest(decoder_path) == receipt['decoder_sha256']
                    decoder = np.load(decoder_path, allow_pickle=False)
                    assert decoder.shape == (16, 4225)
                for start in range(0, len(ids), 128):
                    stop = min(start+128, len(ids))
                    p = points[start:stop]
                    basis_points = sample(directions, p)
                    delta = basis_points[:, 1:] - basis_points[:, :1]
                    target45 = np.einsum('nc,npc->np', teacher[start:stop, :45]/sigma, delta[:, :, :45])
                    remainder = np.einsum('nc,npc->np', teacher[start:stop, 45:]/sigma, delta[:, :, 45:])
                    predicted = np.einsum('nc,npc->np', coordinates[start:stop, :45]/sigma, delta[:, :, :45])
                    if decoder is not None:
                        spatial_points = sample(decoder, p)
                        predicted += np.einsum('nc,npc->np', prediction[start:stop, 142:]/sigma,
                                               spatial_points[:, 1:] - spatial_points[:, :1])
                    error45 = predicted-target45
                    full_error = error45-remainder
                    valid = np.isfinite(p[:, 1:]).all(axis=2) & np.isfinite(p[:, :1]).all(axis=2)
                    counts = valid.sum(axis=1)
                    chosen = counts > 0
                    terms = [error45**2, remainder**2, -2*error45*remainder, full_error**2,
                             np.abs(full_error**2-(error45**2+remainder**2-2*error45*remainder))]
                    sums += np.asarray([np.sum(np.sum(np.where(valid, v, 0.), axis=1)[chosen]/counts[chosen]) for v in terms])
                    valid_shots += int(chosen.sum())
                assert valid_shots > 0
                means = sums/valid_shots
                assert np.isclose(means[3], means[:3].sum(), rtol=1e-12, atol=1e-12)
                result = {'model_sha256': digest(path), 'valid_midpoints': valid_shots,
                          'ray_MSE_teacher45': float(means[0]), 'ray_MSE_teacher_omitted45_to127': float(means[1]),
                          'ray_cross_term': float(means[2]), 'ray_MSE_teacher128': float(means[3]),
                          'decomposition_identity_absolute_residual': float(means[4]),
                          'weighted_anchor_per_output': per_output.tolist(),
                          'normalized_coordinate_MSE64': per_output[:64].tolist(),
                          'spatial_residual_included': decoder is not None}
                if family == 'control':
                    expected = original['results'][str(seed)]['weighted_anchor_error_components']['coordinates64']
                    assert np.isclose(per_output[:64].sum(), expected, rtol=1e-12, atol=1e-12)
                reports[str(seed)][family] = result
                print(json.dumps({'ray_audit_completed': family, 'seed': seed}), flush=True)
        changes = {str(seed): {family: (np.asarray(reports[str(seed)][family]['normalized_coordinate_MSE64'])
                                  - np.asarray(reports[str(seed)]['control']['normalized_coordinate_MSE64'])).tolist()
                               for family in protocol['families'] if family != 'control'} for seed in protocol['seeds']}
        verify()
        outcome = {'status': 'all32_all5954_training_midpoint_ray_decomposition_complete',
                   'shots': 5954, 'one_midpoint_per_training_shot': True,
                   'sample_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(), 'reports': reports,
                   'coordinate_MSE64_delta_vs_matched_control': changes,
                   'equal_valid_midpoint_mean_not_training_frame_weighted_loss': True,
                   'double_export_diagnostic_not_actual_float32_gradient': True,
                   'no_fit_no_model_changes_no358_or357_labels': True, 'not_candidate_admission': True,
                   'protocol_sha256': digest(HERE / 'training_ray_audit_protocol.json'), 'seconds': time.time()-started,
                   'publication': False}
        with (HERE / 'training_ray_audit_analysis.json').open('x') as stream:
            json.dump(outcome, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (HERE / 'training_ray_audit_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time()-started}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
