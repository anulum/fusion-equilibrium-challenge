# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — M3c-0 step 2: FreeGS vs v7 plasma position against the Thomson edge (inputs + our maps)
"""Protocol protocol_m3c0_freegs_position_feasibility.json. Runs in the fec_adapter venv.

    PYTHONPATH=R/fec_adapter python m3c0_score.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from fec_adapter.diii_d_accuracy.shape_edit import Reader
from fec_adapter.shot_reader import read_shot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v8b'))
from edge_boundary import OFFSET, V7, crossings, markers  # noqa: E402

DATA = '$DISK/_offroot_cache/fec_data_local/mast_public_test/'


def position(reader, flux, z0, target_in, target_out):
    reading = reader.read(flux)
    if reading.contour is None:
        return None
    inner, outer = crossings(reading.contour, z0)
    if inner is None:
        return None
    return {'cross_in': inner, 'cross_out': outer, 'err_in_cm': 100 * (inner - target_in),
            'err_out_cm': 100 * (outer - target_out)}


def main():
    rows = json.loads((HERE / 'm3c0_freegs_rows.json').read_text())
    files = sorted(glob.glob(DATA + '*.parquet'))
    reader = Reader('MAST')
    out = []
    with np.load(HERE / 'm3c0_freegs_maps.npz') as freegs_maps, np.load(V7) as v7:
        for row in rows:
            record = {k: row[k] for k in ('shot', 'frame', 'converged')}
            if row['converged']:
                record['ip_ratio'] = row['ip_solved_A'] / row['ip_input_A']
            shot = read_shot(Path(files[row['shot']]))
            order = np.argsort(shot.thomson.r)
            r, te = shot.thomson.r[order], shot.thomson.te[row['frame'], order]
            ok = np.isfinite(te) & (te > 0)
            m_in, m_out = markers(r[ok], te[ok])
            z0 = float(np.median(shot.thomson.z))
            if m_in is None or m_out is None:
                out.append(record)
                continue
            target_in, target_out = m_in - OFFSET, m_out + OFFSET
            record['v7'] = position(reader, v7[f'shot_{row["shot"]:04d}_psirz'][row['frame']].astype(np.float64),
                                    z0, target_in, target_out)
            key = f'shot_{row["shot"]:04d}_frame_{row["frame"]:03d}'
            record['freegs'] = (position(reader, freegs_maps[key], z0, target_in, target_out)
                                if key in freegs_maps.files else None)
            out.append(record)
    paired = [r for r in out if r.get('v7') and r.get('freegs')]
    summary = {'frames': len(out), 'converged': sum(r['converged'] for r in out), 'paired': len(paired)}
    for arm in ('v7', 'freegs'):
        for side in ('out', 'in'):
            values = [r[arm][f'err_{side}_cm'] for r in paired]
            summary[f'{arm}_err_{side}_cm_median'] = float(np.median(values)) if values else None
            summary[f'{arm}_abs_err_{side}_cm_median'] = float(np.median(np.abs(values))) if values else None
    summary['ip_ratio_median'] = float(np.median([r['ip_ratio'] for r in out if 'ip_ratio' in r])) if paired else None
    better = (paired and summary['converged'] >= 10
              and summary['freegs_abs_err_out_cm_median'] <= summary['v7_abs_err_out_cm_median'] - 2.0)
    summary['verdict'] = 'house_solver_artefact_build_M3c' if better else 'shared_by_both_solvers_or_not_shown'
    (HERE / 'm3c0_result.json').write_text(json.dumps({'summary': summary, 'rows': out}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
