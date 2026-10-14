# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D61: candidate archive with the least-squares joint shares for q95 and betaN
"""Reproduce the deployed scalars from their two parts, then write the candidate scalars into a copy of the archive.

    python build_candidate.py SCALARS.npz DEPLOYED.npz OUT_DIR
"""
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

SHARE = {'q95': 0.46, 'betaN': 0.87}
DEPLOYED_SHARE = 0.25
DEPLOYED_SHA = 'e1efabfcd60ddfa7b5668dda3cd5b6e9b32826203b44d36c383c85ba2795ae32'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def blend(base, joint, share):
    return ((1.0 - share) * base.astype(float) + share * joint).astype(np.float32)


def main():
    parts_file, deployed, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    assert digest(deployed) == DEPLOYED_SHA
    out_dir.mkdir(parents=True, exist_ok=False)
    out = out_dir / 'diii_d_public_test.npz'
    changed = {name: 0 for name in SHARE}
    frames = 0
    largest = {name: 0.0 for name in SHARE}
    new = {}
    with np.load(parts_file, allow_pickle=False) as parts, np.load(deployed, allow_pickle=False) as old:
        for i in range(874):
            for name in SHARE:
                key = f'shot_{i:04d}_{name}'
                base, joint = parts[key + '_map'], parts[key + '_joint']
                assert np.array_equal(blend(base, joint, DEPLOYED_SHARE), old[key]), key   # the parts are the deployed ones
                value = blend(base, joint, SHARE[name])
                assert value.shape == old[key].shape and np.isfinite(value).all()
                new[key] = value
                changed[name] += int((value != old[key]).sum())
                largest[name] = max(largest[name], float(np.max(np.abs(value.astype(float) - old[key].astype(float)))))
            frames += len(new[f'shot_{i:04d}_q95'])
    shutil.copyfile(deployed, out)
    with zipfile.ZipFile(out, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        maps = {x.filename: (x.CRC, x.file_size, x.compress_size, x.header_offset) for x in archive.infolist() if x.filename.endswith('_psirz.npy')}
        assert len(maps) == 874
        archive.filelist = [x for x in archive.filelist if x.filename.endswith('_psirz.npy')]
        archive.NameToInfo = {x.filename: x for x in archive.filelist}
        for key, value in new.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            archive.writestr(key + '.npy', buffer.getvalue())
    with zipfile.ZipFile(out) as archive:
        entries = archive.infolist()
        assert len(entries) == 2622 and len({x.filename for x in entries}) == 2622
        for name, values in maps.items():
            entry = archive.getinfo(name)
            assert (entry.CRC, entry.file_size, entry.compress_size, entry.header_offset) == values
    with np.load(out, allow_pickle=False) as check:
        assert len(check.files) == 2622
        for key, value in new.items():
            assert np.array_equal(check[key], value)
    receipt = {'status': 'candidate_built_not_submitted', 'shares': SHARE, 'deployed_share': DEPLOYED_SHARE, 'shots': 874, 'frames': frames,
               'deployed_scalars_reproduced_bit_exactly': True, 'map_records_unchanged': True, 'frames_changed': changed,
               'largest_change': largest, 'deployed_sha256': DEPLOYED_SHA, 'candidate_sha256': digest(out),
               'parts_sha256': digest(parts_file), 'this_script_sha256': digest(__file__), 'public_labels_accessed': False}
    (out_dir / 'build_receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
