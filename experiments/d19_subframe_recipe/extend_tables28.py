# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — D19: the all-data tables (training + validation + holdout; calibration) with sub-frame features
"""Append the 215 sub-frame features to tables28 for the readout refit of the v29 bank members.

    python extend_tables28.py TABLES28_DIR SUBFRAME_DIR OUT_DIR

Blocks are matched by stem; the frames per shot must equal the table's. Writes table_training.npz,
table_calibration.npz and SHA256SUMS (data with a manifest, as the entrant's rule for the DATA_ area asks).
"""
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np


def main():
    tables, sub, out = (Path(a) for a in sys.argv[1:4])
    blocks = {}
    for path in sorted(glob.glob(str(sub / 'out/part_*.npz')) + glob.glob(str(sub / 'out_calibration.npz'))
                       + glob.glob(str(sub / 'out_validation.npz')) + glob.glob(str(sub / 'out_holdout_*.npz'))):
        with np.load(path, allow_pickle=False) as z:
            f, o = z['features'], z['offsets']
            for i, stem in enumerate(z['stems'].astype(str)):
                assert stem not in blocks, stem
                blocks[stem] = f[o[i]:o[i + 1]]
    out.mkdir(parents=True, exist_ok=False)
    lines = []
    for part in ('training', 'calibration'):
        with np.load(tables / f'table_{part}.npz', allow_pickle=False) as t:
            arrays = {k: t[k] for k in t.files}
        counts = np.bincount(arrays['shot_of_row'], minlength=len(arrays['stems']))
        extra = []
        for stem, n in zip(arrays['stems'].astype(str), counts):
            assert blocks[stem].shape[0] == n, (stem, blocks[stem].shape, n)
            extra.append(blocks[stem])
        extra = np.concatenate(extra)
        assert len(extra) == len(arrays['features']) and np.isfinite(extra).all()
        arrays['features'] = np.concatenate([arrays['features'], extra], axis=1)
        np.savez(out / f'table_{part}.npz', **arrays)
        with (out / f'table_{part}.npz').open('rb') as s:
            lines.append(f"{hashlib.file_digest(s, 'sha256').hexdigest()}  table_{part}.npz")
        print(part, arrays['features'].shape, len(arrays['stems']), flush=True)
    (out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


main()
