# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B2c oracle: impose the true saddle flux gap on v25 maps
"""Ceiling of a saddle-balance edit on out-of-sample validation frames (oracle, diagnosis only).

B2b found the heavy tail of the boundary distance in near-double-null frames, where the v25
map ranks its two X-points wrongly. Here each v25 map gets the TRUE signed gap between the
flux at its lower and upper saddle, normalised by its own axis-to-boundary flux span, by a
Gaussian bump centred on each saddle (half the correction each, opposite signs; a bump peaked
on a saddle leaves the saddle where it is to first order). Three rounds re-find the saddles.
Truth enters only as the oracle target; nothing is fitted or selected. The frames are the B2
frames (335 validation shots outside the reserved list, at most 16 per shot).

``closure`` is (boundary level - highest saddle phi) / span: about zero when the scorer's
boundary closes at the X-point, clearly positive when it closes earlier by touching the
envelope. Saddles are searched in each half of the vessel independently of the boundary: grid local
minima of |grad psi|^2 with a saddle Hessian and an in-cell Newton step, inside the envelope,
0.6 < |Z| < 1.5 m, 0.95 < R < 1.8 m; the one with the highest phi (closest to the plasma)
is the half's X-point.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.scoring_kit.common import AXIS_SIGN, CONS_SCALARS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lcfs_structure as b2  # noqa: E402

OUT = HERE / 'results' / 'b2c_saddle_gap_oracle_v25_validation'
SIGN = AXIS_SIGN['DIII-D']
WIDTH = 0.15
ROUNDS = 3
SHARES = (1.0, 0.5)
NEAR_DN = 0.01


def saddles(reader, flux):
    """Return {half: (position, phi)} for 'lower' and 'upper', or None where none is found."""
    phi = SIGN * flux
    step_r = float(reader.r_grid[1] - reader.r_grid[0])
    step_z = float(reader.z_grid[1] - reader.z_grid[0])
    gz, gr = np.gradient(phi, step_z, step_r)
    flat = gr ** 2 + gz ** 2
    out = {}
    for half, zsel in (('lower', lambda z: (z < -0.6) & (z > -1.5)),
                       ('upper', lambda z: (z > 0.6) & (z < 1.5))):
        best = None
        for i in range(1, flat.shape[0] - 1):
            z = reader.z_grid[i]
            if not zsel(z):
                continue
            for j in range(1, flat.shape[1] - 1):
                r = reader.r_grid[j]
                if not (0.95 < r < 1.8) or not reader.coarse[i, j]:
                    continue
                if flat[i, j] > flat[i - 1:i + 2, j - 1:j + 2].min():
                    continue
                h_rr = (phi[i, j + 1] - 2 * phi[i, j] + phi[i, j - 1]) / step_r ** 2
                h_zz = (phi[i + 1, j] - 2 * phi[i, j] + phi[i - 1, j]) / step_z ** 2
                h_rz = (phi[i + 1, j + 1] - phi[i + 1, j - 1] - phi[i - 1, j + 1]
                        + phi[i - 1, j - 1]) / (4 * step_r * step_z)
                hess = np.array([[h_rr, h_rz], [h_rz, h_zz]])
                if not np.all(np.isfinite(hess)) or np.linalg.det(hess) >= 0:
                    continue
                newton = np.linalg.solve(hess, np.array([gr[i, j], gz[i, j]]))
                if abs(newton[0]) > step_r or abs(newton[1]) > step_z:
                    continue
                pos = np.array([r, z]) - newton
                value = float(b2.sample(phi, reader, pos[None])[0])
                if best is None or value > best[1]:
                    best = (pos, value)
        out[half] = best
    return out


def gap_of(reader, flux, reading=None):
    reading = reader.read(flux) if reading is None else reading
    s = saddles(reader, flux)
    if reading.contour is None or s['lower'] is None or s['upper'] is None:
        return None
    phi = SIGN * flux
    level = float(np.mean(b2.sample(phi, reader, np.asarray(reading.contour))))
    axis = float(b2.sample(phi, reader, np.asarray([reading.axis]))[0])
    span = axis - level
    if not span > 0:
        return None
    closure = (level - max(s['lower'][1], s['upper'][1])) / span
    return (s['lower'][1] - s['upper'][1]) / span, span, s, closure


def edited(reader, flux, target_gap, share):
    current = gap_of(reader, flux)
    if current is None:
        return None
    aim = current[0] + share * (target_gap - current[0])
    for _ in range(ROUNDS):
        gap, span, s, _ = current
        delta = (aim - gap) * span
        if abs(aim - gap) < 1e-5:
            break
        bump = np.zeros_like(flux)
        for half, sign in (('lower', 0.5), ('upper', -0.5)):
            pos = s[half][0]
            bump += sign * delta * np.exp(-((reader.rr - pos[0]) ** 2 + (reader.zz - pos[1]) ** 2)
                                          / WIDTH ** 2)
        trial = flux + SIGN * bump
        new = gap_of(reader, trial)
        if new is None:
            break
        flux, current = trial, new
    return flux


def measures(reader, truth_reading, ct, rgeo, flux, truth):
    r = reader.read(flux)
    if r.contour is None:
        return None
    d = b2.hausdorff(np.asarray(r.contour, float), ct) / rgeo
    cons = [(r.values[n] - truth_reading.values[n]) ** 2 for n in CONS_SCALARS]
    return [d, *cons, float(np.sum((flux - truth) ** 2))]


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        sys.path.insert(0, str(b2.REPO / 'fec_harness'))
        import run_parallel_score as scoring
        stems = [p.stem for p in scoring.chosen_files('validation', 0, 357)]
        reserved = set(json.loads((b2.REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems'])
        keys = np.load(b2.OUT / 'frames.npz')['keys']
        reader = Reader('DIII-D')
        rows, skipped = [], {'no_gap_truth': 0, 'no_gap_pred': 0, 'edit_lost': 0}
        stage = 'frames'
        with np.load(b2.V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
            for shot in np.unique(keys[:, 0]):
                stem = stems[shot]
                assert stem not in reserved
                maps = archive[f'shot_{shot:04d}_psirz']
                truth_maps = np.asarray(pq.read_table(b2.DATA / f'{stem}.parquet', columns=['efit_psirz'])
                                        ['efit_psirz'][0].as_py(), dtype=np.float64)
                for frame in keys[keys[:, 0] == shot, 1]:
                    truth, pred = truth_maps[frame], np.asarray(maps[frame], np.float64)
                    rt = reader.read(truth)
                    gt = gap_of(reader, truth, rt)
                    if gt is None:
                        skipped['no_gap_truth'] += 1
                        continue
                    gp = gap_of(reader, pred)
                    if gp is None:
                        skipped['no_gap_pred'] += 1
                        continue
                    ct = np.asarray(rt.contour, float)
                    rgeo = 0.5 * (ct[:, 0].max() + ct[:, 0].min())
                    base = measures(reader, rt, ct, rgeo, pred, truth)
                    row = [shot, frame, gt[0], gp[0], gt[3], gp[3], *base]
                    for share in SHARES:
                        out = edited(reader, pred, gt[0], share)
                        m = measures(reader, rt, ct, rgeo, out, truth) if out is not None else None
                        if m is None:
                            skipped['edit_lost'] += 1
                            m = base
                        row += m
                    rows.append(row)
                if len(rows) and shot % 25 == 0:
                    print(json.dumps({'shot_index': int(shot), 'frames': len(rows),
                                      'seconds': round(time.time() - started)}), flush=True)
        stage = 'summary'
        per = ['D', *[f'cons_se_{n}' for n in CONS_SCALARS], 'flux_sse']
        columns = ['shot', 'frame', 'gap_true', 'gap_v25', 'closure_true', 'closure_v25',
                   *[f'base_{c}' for c in per]]
        for share in SHARES:
            columns += [f'share{share}_{c}' for c in per]
        data = np.asarray(rows, dtype=np.float64)
        np.savez_compressed(OUT / 'frames.npz', data=data, columns=np.asarray(columns))
        col = {c: data[:, i] for i, c in enumerate(columns)}
        tail = col['base_D'] > 0.02
        near = np.abs(col['gap_true']) < NEAR_DN
        near_pred = np.abs(col['gap_v25']) < NEAR_DN
        flip = np.sign(col['gap_true']) != np.sign(col['gap_v25'])

        def block(mask):
            out = {'frames': int(mask.sum())}
            for tag in ['base', *[f'share{s}' for s in SHARES]]:
                out[f'{tag}_D_mean'] = float(col[f'{tag}_D'][mask].mean()) if mask.any() else None
            for tag in [f'share{s}' for s in SHARES]:
                out[f'{tag}_cons_se_ratio'] = {n: float(col[f'{tag}_cons_se_{n}'][mask].sum()
                                                        / max(col[f'base_cons_se_{n}'][mask].sum(), 1e-300))
                                               for n in CONS_SCALARS} if mask.any() else None
                out[f'{tag}_flux_sse_ratio'] = float(col[f'{tag}_flux_sse'][mask].sum()
                                                     / col['base_flux_sse'][mask].sum()) if mask.any() else None
            return out

        # Realisable gating: apply the oracle edit only where the v25 map itself is near double null.
        gated = {}
        for share in SHARES:
            d = np.where(near_pred, col[f'share{share}_D'], col['base_D'])
            gated[f'share{share}_D_mean_all_frames'] = float(d.mean())
        summary = {'status': 'diagnostic_complete', 'oracle': True, 'selection_or_fit': False,
                   'frames': len(rows), 'skipped': skipped, 'width_m': WIDTH, 'rounds': ROUNDS,
                   'near_double_null_threshold_psiN': NEAR_DN,
                   'flip_share': float(flip.mean()), 'flip_share_in_tail': float(flip[tail].mean()),
                   'all': block(np.ones(len(rows), bool)), 'tail_D_gt_0p02': block(tail),
                   'true_near_double_null': block(near), 'flipped': block(flip),
                   'edit_only_where_v25_near_double_null': gated,
                   'v25_near_double_null_share': float(near_pred.mean()),
                   'closure_above_saddle_quantiles_true_tail_vs_rest': [
                       np.quantile(col['closure_true'][tail], [.1, .5, .9]).tolist(),
                       np.quantile(col['closure_true'][~tail], [.1, .5, .9]).tolist()],
                   'closure_above_saddle_quantiles_v25_tail_vs_rest': [
                       np.quantile(col['closure_v25'][tail], [.1, .5, .9]).tolist(),
                       np.quantile(col['closure_v25'][~tail], [.1, .5, .9]).tolist()],
                   'MAST_targets_used': False, 'reserved_excluded': True}
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=1), flush=True)
        code, stage = 0, 'complete'
    except Exception:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (OUT / 'terminal.json').write_text(json.dumps(
            {'actual_exit_code': code, 'stage': stage, 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
