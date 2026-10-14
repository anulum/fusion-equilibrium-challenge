# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D61: map-network and joint-network scalars of the 874 public shots, side by side
"""python extract_scalars.py H69_ARCHIVE.npz D36_PUBLIC_PREDICTIONS_DIR STEMS.txt OUT.npz"""
import hashlib
import sys
from pathlib import Path

import numpy as np

archive, joint_dir, stems_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == '99a7df91679012e982f106daff0aa264da395c4f67362abe649d203fd556f901'
stems = stems_file.read_text().split()
assert len(stems) == 874 and stems == sorted(stems)
store = {}
with np.load(archive, allow_pickle=False) as old:
    for i, stem in enumerate(stems):
        with np.load(joint_dir / (stem + '.npz'), allow_pickle=False) as joint:
            for name in ('q95', 'betaN'):
                base = old[f'shot_{i:04d}_{name}']
                assert base.dtype == np.float32 and base.shape == joint[name].shape and joint[name].dtype == np.float64
                store[f'shot_{i:04d}_{name}_map'] = base
                store[f'shot_{i:04d}_{name}_joint'] = joint[name]
np.savez_compressed(out, **store)
print('arrays', len(store), 'frames', sum(v.size for k, v in store.items() if k.endswith('q95_map')))
