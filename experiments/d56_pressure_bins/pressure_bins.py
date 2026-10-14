# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D56: electron-pressure profile on the table's own position bins
"""The channel product n_e T_e averaged into the same fixed bins the table uses for T_e and n_e separately.

Per frame, on the pulse the package aligns to the frame: 12 height bins of the vertical laser and 6 radius bins of
the other one (mean of the channel products that fired jointly, and a flag), and peak, mean and fired share of the
product per laser: 18 + 18 + 6 = 42 columns. Inputs only.

    PYTHONPATH=PKG/source/fec_adapter python pressure_bins.py SPLIT_DIR STEMS.json BASICS.npz OUT.npz WORKERS
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fec_adapter import shot_reader as reader
from fec_adapter.diii_d_accuracy import features as f
from fec_adapter.input_features import profile_summary

WIDTH = 42


def features_of(path):
    inputs = f.AlignedInputs.from_shot(reader.read_shot(Path(path)))
    te, ne = inputs.thomson_te / 1.0e3, inputs.thomson_ne / 1.0e19
    joint = np.isfinite(te) & np.isfinite(ne) & (te > 0.0) & (ne > 0.0)
    product = np.where(joint, te * ne, np.nan)
    frames = inputs.times_ms.size
    if inputs.thomson_r.size == 0:
        return np.zeros((frames, WIDTH), dtype=np.float32)
    vertical = np.abs(inputs.thomson_r - np.median(inputs.thomson_r)) < f.VERTICAL_TOLERANCE
    blocks = []
    for member, positions, edges in ((vertical, inputs.thomson_z, f.HEIGHT_EDGES), (~vertical, inputs.thomson_r, f.RADIUS_EDGES)):
        mean, fired = f.binned_profile(product, positions, member, edges)
        blocks += [mean, fired]
    for member in (vertical, ~vertical):
        blocks.append(profile_summary(product[:, member]) if member.any() else np.zeros((frames, 3)))
    out = np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape == (frames, WIDTH), out.shape
    return out.astype(np.float32)


def main():
    split, stems_file, basics_file, out, workers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
    stems = json.loads(stems_file.read_text())
    if len(sys.argv) > 6:
        stems = stems[:int(sys.argv[6])]
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        blocks = list(pool.map(features_of, [str(split / f'{stem}.parquet') for stem in stems], chunksize=8))
    features = np.concatenate(blocks)
    shot_of_row = np.repeat(np.arange(len(blocks)), [len(b) for b in blocks])
    assert np.isfinite(features).all()
    with np.load(basics_file, allow_pickle=False) as z:
        basics = z['basics'][:len(features)]
        assert np.array_equal(z['shot_of_row'][:len(features)], shot_of_row)
    np.savez(out, features=features, basics=basics, shot_of_row=shot_of_row, stems=np.asarray(stems))
    print(json.dumps({'DONE': True, 'shots': len(blocks), 'frames': int(len(features)), 'width': int(features.shape[1]),
                      'zero_share_by_column': np.round((features == 0).mean(axis=0), 3).tolist(),
                      'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
