# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B5 diagnostic: time structure of the v25 errors
"""How much of the v25 error on out-of-sample validation shots is constant within a shot.

Diagnosis only, nothing fitted or selected; the 22 reserved validation stems are
excluded. For the flux map and for the two submitted scalars the per-frame
error is split into the shot-mean error (persists through the discharge) and
the remainder, and the remainder's autocorrelation at lags of 1, 2, 5 and 10
frames is measured. A large shot-mean share says the error is a per-shot bias
that only shot-level information can remove; a remainder with low lag-1
autocorrelation is frame noise that smoothing in time could remove; a remainder
with high autocorrelation is slow drift that neither removes.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
DATA = Path('$SAS/DATASETS/'
            'fusion-equilibrium-challenge/data/diii_d_train')
V25 = Path('$LAB/'
           'work/second_agent-recovery-20260926/results/'
           'fixed_axis_volume_full357_56cb')
OUT = HERE / 'results' / 'b5_time_structure_v25_validation'
LAGS = (1, 2, 5, 10)
CHANNELS = ('flux', 'q95', 'betaN')


def split(error):
    """Return (total SS, shot-mean SS, remainder, remainder autocorrelation sums per lag)."""
    flat = error.reshape(len(error), -1)
    mean = flat.mean(axis=0)
    rest = flat - mean
    total = float(np.square(flat).sum())
    bias = float(np.square(mean).sum()) * len(flat)
    lagged = []
    for lag in LAGS:
        if len(rest) > lag:
            lagged.append((float(np.sum(rest[lag:] * rest[:-lag])),
                           float(np.sum(np.square(rest[lag:]))),
                           float(np.sum(np.square(rest[:-lag])))))
        else:
            lagged.append((0.0, 0.0, 0.0))
    return total, bias, lagged


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
        acc = {c: {'total': 0.0, 'bias': 0.0, 'lag': np.zeros((len(LAGS), 3))} for c in CHANNELS}
        shots, per_shot, steps = [], [], []
        stage = 'shots'
        with np.load(V25 / 'diii_d_train.npz', allow_pickle=False) as archive:
            for index, stem in enumerate(stems):
                if stem in reserved:
                    continue
                table = pq.read_table(DATA / f'{stem}.parquet',
                                      columns=['efit_times', 'efit_psirz', 'efit_q95', 'efit_beta_n'])
                times = np.asarray(table['efit_times'][0].as_py(), dtype=np.float64)
                truth = {'flux': np.asarray(table['efit_psirz'][0].as_py(), dtype=np.float64),
                         'q95': np.asarray(table['efit_q95'][0].as_py(), dtype=np.float64),
                         'betaN': np.asarray(table['efit_beta_n'][0].as_py(), dtype=np.float64)}
                pred = {'flux': archive[f'shot_{index:04d}_psirz'].astype(np.float64),
                        'q95': archive[f'shot_{index:04d}_q95'].astype(np.float64),
                        'betaN': archive[f'shot_{index:04d}_betaN'].astype(np.float64)}
                steps.extend(np.diff(times).tolist())
                row = {'stem': stem, 'frames': len(times)}
                for channel in CHANNELS:
                    error = pred[channel] - truth[channel]
                    if channel != 'flux':
                        keep = np.isfinite(error)
                        error = error[keep]
                        if len(error) < 2:
                            continue
                    total, bias, lagged = split(error)
                    acc[channel]['total'] += total
                    acc[channel]['bias'] += bias
                    acc[channel]['lag'] += np.asarray(lagged)
                    row[f'{channel}_bias_share'] = bias / total if total > 0 else None
                per_shot.append(row)
                shots.append(stem)
                if len(shots) % 50 == 0:
                    print(json.dumps({'shots': len(shots), 'seconds': round(time.time() - started)}),
                          flush=True)
        stage = 'summary'
        summary = {'status': 'diagnostic_complete', 'selection_or_fit': False,
                   'shots': len(shots), 'reserved_excluded': len(stems) - len(shots),
                   'frame_step_ms_quantiles_5_50_95': np.quantile(steps, [.05, .5, .95]).tolist(),
                   'lags_frames': list(LAGS), 'MAST_targets_used': False}
        for channel in CHANNELS:
            a = acc[channel]
            corr = [s / np.sqrt(x * y) if x > 0 and y > 0 else None for s, x, y in a['lag']]
            summary[channel] = {'pooled_bias_share': a['bias'] / a['total'],
                                'remainder_autocorrelation_by_lag': corr,
                                'median_shot_bias_share': float(np.median(
                                    [r[f'{channel}_bias_share'] for r in per_shot
                                     if r.get(f'{channel}_bias_share') is not None]))}
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        (OUT / 'per_shot.json').write_text(json.dumps(per_shot))
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
