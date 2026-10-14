# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — assemble the DIII-D archive: maps of a new build, scalars of the D36 archive unchanged
"""Write a DIII-D public-test archive with ``psirz`` from a new map build and ``q95``/``betaN`` from the D36 archive.

The X-point blend changes maps only; the submitted scalars stay those of submission 975851, array for array.
Every array of the result is compared byte for byte with its source after the archive is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHOTS, FRAMES = 874, 193359


def digest(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', type=Path, required=True, help='archive of the new map build')
    parser.add_argument('--scalars', type=Path, required=True, help='the D36 DIII-D archive')
    parser.add_argument('--scalars-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.scalars) == args.scalars_sha256
    assert not args.out.exists()
    entries, frames, changed = {}, 0, 0
    with np.load(args.maps, allow_pickle=False) as new, np.load(args.scalars, allow_pickle=False) as old:
        assert len(new.files) == len(old.files) == 3 * SHOTS and set(new.files) == set(old.files)
        for i in range(SHOTS):
            prefix = f'shot_{i:04d}_'
            maps, q95, beta = new[prefix + 'psirz'], old[prefix + 'q95'], old[prefix + 'betaN']
            assert maps.dtype == q95.dtype == beta.dtype == np.float32
            assert maps.shape == (len(q95), 65, 65) and q95.shape == beta.shape == (len(q95),)
            assert old[prefix + 'psirz'].shape == maps.shape
            assert np.isfinite(maps).all() and np.isfinite(q95).all() and np.isfinite(beta).all()
            changed += int(not same(maps, old[prefix + 'psirz']))
            entries[prefix + 'psirz'], entries[prefix + 'q95'], entries[prefix + 'betaN'] = maps, q95, beta
            frames += len(q95)
        assert frames == FRAMES
        np.savez_compressed(args.out, allow_pickle=False, **entries)
        with np.load(args.out, allow_pickle=False) as written:
            assert len(written.files) == 3 * SHOTS
            for i in range(SHOTS):
                prefix = f'shot_{i:04d}_'
                assert same(written[prefix + 'psirz'], new[prefix + 'psirz'])
                assert same(written[prefix + 'q95'], old[prefix + 'q95'])
                assert same(written[prefix + 'betaN'], old[prefix + 'betaN'])
    receipt = {'status': 'assembled_maps_new_scalars_D36', 'shots': SHOTS, 'frames': frames,
               'maps_source': str(args.maps), 'maps_source_sha256': digest(args.maps),
               'scalars_source': str(args.scalars), 'scalars_source_sha256': args.scalars_sha256,
               'shots_with_changed_maps': changed, 'archive_sha256': digest(args.out),
               'every_array_byte_equal_to_its_source': True, 'this_script_sha256': digest(__file__)}
    with args.out.with_suffix('.receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=1)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
