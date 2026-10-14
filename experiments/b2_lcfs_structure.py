# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2 diagnostic: structure of the v25 boundary error
"""Decompose the v25 finished-map LCFS error on out-of-sample validation shots.

Diagnosis only: nothing here is fitted or selected. Every one of the 65 v25
members was trained on the training split and 50 of them on calibration too,
so validation is the only labelled part that is out-of-sample for the whole
ensemble. The 22 reserved stems inside validation are excluded.

Per sampled frame the scorer's boundary is read from the truth map and from
the v25 map, and the error is split three ways:

* angular Fourier modes of the radius error about the TRUE axis, and the
  boundary distance left if modes up to m were removed exactly by moving the
  v25 boundary points radially (an oracle ceiling for a low-parameter boundary
  readout; a polar polygon of the TRUE boundary is itself worse than v25 near
  the X-point, so the polygon is reported only as a floor);
* the boundary flux level: the v25 map's own flux sampled on the true
  boundary, relative to the level its extracted boundary sits at (uniform
  offset versus spread along the boundary, in normalised flux);
* where the Hausdorff maximum sits (angle, and whether it is at an X-point end).
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import directed_hausdorff

from fec_adapter.diii_d_accuracy.contour_edit import radii
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.scoring_kit.common import AXIS_SIGN

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$SAS/DATASETS/'
            'fusion-equilibrium-challenge/data/diii_d_train')
V25 = Path('$LAB/'
           'work/second_agent-recovery-20260926/results/'
           'fixed_axis_volume_full357_56cb')
OUT = HERE / 'results' / 'b2_lcfs_structure_v25_validation'
FRAMES_PER_SHOT = 16
ANGLES = 256
MODE_CUTS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 32)
BANDS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (9, 32), (33, 128))


def hausdorff(a, b):
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def worst_point(a, b):
    """The point (of either set) at which the symmetric Hausdorff maximum is attained."""
    d1, i1, _ = directed_hausdorff(a, b)
    d2, i2, _ = directed_hausdorff(b, a)
    return a[i1] if d1 >= d2 else b[i2]


def polygon(axis, r):
    theta = np.linspace(-np.pi, np.pi, r.size, endpoint=False)
    return np.column_stack((axis[0] + r * np.cos(theta), axis[1] + r * np.sin(theta)))


def highpass(delta, cut):
    spectrum = np.fft.rfft(delta)
    spectrum[:cut + 1] = 0.0
    return np.fft.irfft(spectrum, n=delta.size)


def sample(phi, reader, pts):
    step_r = float(reader.r_grid[1] - reader.r_grid[0])
    step_z = float(reader.z_grid[1] - reader.z_grid[0])
    rows = (pts[:, 1] - reader.z_grid[0]) / step_z
    cols = (pts[:, 0] - reader.r_grid[0]) / step_r
    return map_coordinates(phi, [rows, cols], order=1, mode='nearest')


def displaced(points, axis, delta, cut):
    """Move each point radially about ``axis`` by minus the modes up to ``cut`` of ``delta``."""
    low = delta - highpass(delta, cut)
    grid = np.linspace(-np.pi, np.pi, delta.size, endpoint=False)
    theta = np.arctan2(points[:, 1] - axis[1], points[:, 0] - axis[0])
    shift = np.interp(theta, np.concatenate((grid - 2 * np.pi, grid, grid + 2 * np.pi)),
                      np.tile(low, 3))
    return points - shift[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))


def frame_row(reader, truth, pred):
    rt, rp = reader.read(truth), reader.read(pred)
    if rt.contour is None or rp.contour is None:
        return None
    ct, cp = np.asarray(rt.contour, float), np.asarray(rp.contour, float)
    axis = rt.axis
    rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
    d = hausdorff(cp, ct) / rgeo
    r_t, r_p = radii(ct, axis, ANGLES), radii(cp, axis, ANGLES)
    delta = r_p - r_t
    floor = hausdorff(polygon(axis, r_t), ct) / rgeo
    d_cut = [hausdorff(displaced(cp, axis, delta, m), ct) / rgeo for m in MODE_CUTS]
    theta = np.linspace(-np.pi, np.pi, delta.size, endpoint=False)
    translation = 2.0 * np.asarray([np.mean(delta * np.cos(theta)), np.mean(delta * np.sin(theta))])
    d_translate = hausdorff(cp - translation, ct) / rgeo
    power = np.abs(np.fft.rfft(delta)) ** 2
    power[1:] *= 2.0
    total = float(power.sum())
    bands = [float(power[lo:hi + 1].sum()) / total if total > 0 else 0.0 for lo, hi in BANDS]
    worst = worst_point(cp, ct)
    angle = float(np.arctan2(worst[1] - axis[1], worst[0] - axis[0]))
    zlow, zhigh = ct[np.argmin(ct[:, 1])], ct[np.argmax(ct[:, 1])]
    near_x = min(np.hypot(*(worst - zlow)), np.hypot(*(worst - zhigh))) / rgeo
    sign = AXIS_SIGN['DIII-D']
    phi = sign * np.asarray(pred, float)
    level = float(np.mean(sample(phi, reader, cp)))
    on_true = sample(phi, reader, ct)
    axis_value = float(sample(phi, reader, np.asarray([rp.axis]))[0])
    span = axis_value - level
    offset = float(np.mean(on_true) - level)
    return [d, floor, d_translate, *d_cut, *bands, angle, near_x,
            offset / span if span > 0 else np.nan,
            float(np.std(on_true - level)) / span if span > 0 else np.nan,
            float(np.hypot(*translation)),
            float(np.sqrt(np.mean(delta ** 2))), float(np.max(np.abs(delta)))]


COLUMNS = (['D', 'D_polar256_floor', 'D_after_m1_translation']
           + [f'D_without_modes_le_{m}' for m in MODE_CUTS]
           + [f'power_m{lo}_{hi}' for lo, hi in BANDS]
           + ['worst_angle', 'worst_to_xpoint_end_over_rgeo', 'level_offset_psiN',
              'level_spread_psiN', 'm1_translation_m', 'radius_rms_m', 'radius_max_m'])


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        sys.path.insert(0, str(REPO / 'fec_harness'))
        import run_parallel_score as scoring
        stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
        build = json.loads((V25 / 'build_report.json').read_text())
        assert build['part'] == 'validation' and build['shots'] == 357 == len(stems)
        reserved = set(json.loads((REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems'])
        splits = json.loads((REPO / 'fec_adapter/splits.json').read_text())
        assert set(stems) == set(splits['validation'])
        reader = Reader('DIII-D')
        rows, keys, skipped = [], [], []
        stage = 'frames'
        with np.load(V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
            for index, stem in enumerate(stems):
                if stem in reserved:
                    skipped.append(stem)
                    continue
                maps = archive[f'shot_{index:04d}_psirz']
                table = pq.read_table(DATA / f'{stem}.parquet', columns=['efit_psirz'])
                truth = np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float64)
                assert truth.shape == maps.shape, stem
                picks = np.unique(np.linspace(0, len(maps) - 1, min(FRAMES_PER_SHOT, len(maps)),
                                              dtype=int))
                for frame in picks:
                    row = frame_row(reader, truth[frame], np.asarray(maps[frame], float))
                    if row is None:
                        continue
                    rows.append(row)
                    keys.append((index, int(frame)))
                if (index + 1) % 25 == 0:
                    print(json.dumps({'shots': index + 1, 'frames': len(rows),
                                      'seconds': round(time.time() - started)}), flush=True)
        stage = 'summary'
        data = np.asarray(rows, dtype=np.float64)
        np.savez_compressed(OUT / 'frames.npz', data=data, columns=np.asarray(COLUMNS),
                            keys=np.asarray(keys))
        col = {name: data[:, i] for i, name in enumerate(COLUMNS)}
        summary = {
            'status': 'diagnostic_complete', 'selection_or_fit': False,
            'reserved_excluded': sorted(skipped), 'shots': len(stems) - len(skipped),
            'frames': len(rows), 'frames_per_shot_max': FRAMES_PER_SHOT,
            'mean': {name: float(np.nanmean(col[name])) for name in COLUMNS},
            'median': {name: float(np.nanmedian(col[name])) for name in COLUMNS},
            'D_quantiles_50_90_99': np.quantile(col['D'], [.5, .9, .99]).tolist(),
            'share_worst_within_0p1_rgeo_of_xpoint_end': float(np.mean(
                col['worst_to_xpoint_end_over_rgeo'] < 0.1)),
            'worst_angle_histogram_16': np.histogram(col['worst_angle'], 16,
                                                     (-np.pi, np.pi))[0].tolist(),
            'share_m1_translation_improves_D': float(np.mean(col['D_after_m1_translation'] < col['D'])),
            'share_level_offset_negative': float(np.nanmean(col['level_offset_psiN'] < 0)),
            'MAST_targets_used': False, 'v25_archive_sha256_from': str(V25 / 'SHA256SUMS'),
        }
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary['mean'], indent=1), flush=True)
        code, stage = 0, 'complete'
    except Exception:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (OUT / 'terminal.json').write_text(json.dumps(
            {'actual_exit_code': code, 'stage': stage, 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
