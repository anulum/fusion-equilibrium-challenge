# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D55: compare a rebuilt prediction archive with the submitted one, array by array
"""python compare_archives.py NEW.npz REFERENCE.npz OUT.json"""
import json
import sys

import numpy as np


def main():
    new, ref = np.load(sys.argv[1]), np.load(sys.argv[2])
    keys_new, keys_ref = sorted(new.files), sorted(ref.files)
    report = {'same_keys': keys_new == keys_ref, 'keys': len(keys_ref), 'shots': 0, 'frames': 0, 'non_finite': 0,
              'scalar_arrays_identical': 0, 'scalar_arrays': 0, 'map_arrays_identical': 0, 'maps_within_1e-6': 0,
              'map_max_abs': 0.0, 'worst': []}
    rows = []
    for key in keys_ref:
        if key not in new.files:
            continue
        a, b = new[key], ref[key]
        assert a.shape == b.shape and a.dtype == b.dtype, (key, a.shape, b.shape, a.dtype, b.dtype)
        report['non_finite'] += int((~np.isfinite(a)).sum())
        if key.endswith('_psirz'):
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            report['shots'] += 1
            report['frames'] += int(a.shape[0])
            report['map_arrays_identical'] += int(np.array_equal(a, b))
            report['maps_within_1e-6'] += int(d.max() <= 1e-6)
            report['map_max_abs'] = max(report['map_max_abs'], float(d.max()))
            rows.append((float(d.max()), float(np.sqrt(np.mean(d ** 2))), key))
        else:
            report['scalar_arrays'] += 1
            report['scalar_arrays_identical'] += int(np.array_equal(a, b))
    rows.sort(reverse=True)
    report['worst'] = [{'key': k, 'max_abs': m, 'rms': r} for m, r, k in rows[:10]]
    report['map_max_abs_quantiles_over_shots'] = np.quantile([m for m, _, _ in rows], [0, .5, .9, .99, 1]).tolist() if rows else None
    report['map_rms_median_over_shots'] = float(np.median([r for _, r, _ in rows])) if rows else None
    report['A'] = bool(report['same_keys'] and report['non_finite'] == 0 and report['scalar_arrays_identical'] == report['scalar_arrays'])
    report['B'] = bool(rows and report['maps_within_1e-6'] == report['shots'])
    report['QUALIFIED'] = bool(report['A'] and report['B'])
    open(sys.argv[3], 'w').write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
