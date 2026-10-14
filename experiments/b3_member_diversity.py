# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — B3 diagnostic: how independent the 65 v25 members are
"""Measure the error correlation between the 65 v25 members on out-of-sample shots.

Diagnosis only, nothing fitted or selected; the 22 reserved validation stems are
excluded and every eighth frame is used. Each member's raw output (before any
map finishing) is compared with the table's truth: the leading flux coordinates
(orthonormal basis, so squared coordinate error is squared map error), q95,
betaN, li and the 64 boundary radii. For every channel the report gives the
mean member error, the error of the member mean, the mean pairwise correlation
of member errors, and the part of the error that more members of the same kind
could still remove (the ambiguity term divided by the member count) against the
part they share, which only a different kind of model can reach.
"""
import json
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

from fec_adapter.diii_d_accuracy.network import Network

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
TABLES = REPO / 'fec_adapter/runs/diii_d_accuracy/tables_lcfs_discharge'
RECIPE = Path('$LAB/'
              'work/second_agent-recovery-20260926/results/'
              'fixed_axis_volume_full357_56cb/network_recipe.sha')
OUT = HERE / 'results' / 'b3_member_diversity_v25_validation'
STRIDE = 8
FLUX_COORDINATES = 45
LABELS = {'q95': ['q95'], 'betaN': ['beta_n'], 'li': ['li_scored'],
          'boundary_radii': [f'lcfs_r_{i:02d}' for i in range(64)]}


def family(path):
    name = Path(path).parent.name
    for key in ('discharge_hx', 'discharge_hc', 'discharge_h', 'discharge_c'):
        if key in name:
            return key
    return 'other'


def channel_report(errors, names):
    """errors: (members, rows, width) with NaN where a member lacks the output."""
    have = np.all(np.isfinite(errors), axis=(1, 2))
    e = errors[have]
    members = len(e)
    if members < 2:
        return {'members': int(members)}
    individual = np.mean(np.square(e), axis=(1, 2))
    mean_error = np.mean(np.square(e.mean(axis=0)))
    flat = e.reshape(members, -1)
    norms = np.linalg.norm(flat, axis=1)
    gram = flat @ flat.T / np.outer(norms, norms)
    upper = gram[np.triu_indices(members, 1)]
    fams = [names[i] for i in np.nonzero(have)[0]]
    within, across = [], []
    for i, j in combinations(range(members), 2):
        (within if fams[i] == fams[j] else across).append(gram[i, j])
    ambiguity = float(individual.mean() - mean_error)
    return {'members': int(members), 'mean_member_mse': float(individual.mean()),
            'best_member_mse': float(individual.min()), 'ensemble_mean_mse': float(mean_error),
            'ambiguity_term': ambiguity,
            'mean_pairwise_error_correlation': float(upper.mean()),
            'min_pairwise_error_correlation': float(upper.min()),
            'within_family_correlation': float(np.mean(within)) if within else None,
            'across_family_correlation': float(np.mean(across)) if across else None,
            'shared_error_share_of_ensemble': float(upper.mean() * individual.mean() / mean_error),
            'removable_by_infinite_same_kind_members_share': float(
                max(0.0, mean_error - upper.mean() * individual.mean()) / mean_error)}


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    code, stage = 1, 'setup'
    try:
        reserved = set(json.loads((REPO / 'fec_adapter/reserved_for_evaluation.json').read_text())['stems'])
        with np.load(TABLES / 'table_validation.npz', allow_pickle=False) as table:
            stems = table['stems'].astype(str)
            shot = table['shot_of_row']
            keep = ~np.isin(stems[shot], sorted(reserved))
            rows = np.nonzero(keep)[0][::STRIDE]
            features = table['features'][rows]
            coords = table['coordinates'][rows, :FLUX_COORDINATES].astype(np.float64)
            label_names = table['label_names'].astype(str).tolist()
            labels = table['labels'][rows].astype(np.float64)
            layout = str(table['feature_layout'])
        paths = [line.split(maxsplit=1)[1] for line in RECIPE.read_text().splitlines()]
        assert len(paths) == 65
        stage = 'members'
        errors = {'flux': np.full((65, len(rows), FLUX_COORDINATES), np.nan)}
        for key, cols in LABELS.items():
            errors[key] = np.full((65, len(rows), len(cols)), np.nan)
        names, skipped = [], []
        for m, path in enumerate(paths):
            net = Network.load(Path(path))
            names.append(family(path))
            if net.feature_layout != layout:
                skipped.append({'member': m, 'layout': net.feature_layout})
                continue
            predicted, outputs = net.predict(features[:, :net.feature_mean.size])
            k = min(FLUX_COORDINATES, predicted.shape[1])
            errors['flux'][m, :, :k] = predicted[:, :k] - coords[:, :k]
            if k < FLUX_COORDINATES:
                errors['flux'][m, :, k:] = -coords[:, k:]
            for key, cols in LABELS.items():
                if all(c in outputs for c in cols) and all(c in label_names for c in cols):
                    truth = labels[:, [label_names.index(c) for c in cols]]
                    errors[key][m] = np.column_stack([outputs[c] for c in cols]) - truth
            print(json.dumps({'member': m, 'family': names[-1],
                              'seconds': round(time.time() - started)}), flush=True)
        stage = 'summary'
        finite_rows = np.all(np.isfinite(labels[:, [label_names.index(c) for c in
                                                    sum(LABELS.values(), [])]]), axis=1)
        report = {'status': 'diagnostic_complete', 'selection_or_fit': False,
                  'rows': int(len(rows)), 'rows_with_all_labels': int(finite_rows.sum()),
                  'stride': STRIDE, 'reserved_excluded': True, 'layout': layout,
                  'skipped_members': skipped,
                  'families': {f: names.count(f) for f in sorted(set(names))},
                  'MAST_targets_used': False}
        report['flux'] = channel_report(errors['flux'], names)
        for key in LABELS:
            report[key] = channel_report(errors[key][:, finite_rows], names)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=1), flush=True)
        code, stage = 0, 'complete'
    except Exception:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (OUT / 'terminal.json').write_text(json.dumps(
            {'actual_exit_code': code, 'stage': stage, 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
