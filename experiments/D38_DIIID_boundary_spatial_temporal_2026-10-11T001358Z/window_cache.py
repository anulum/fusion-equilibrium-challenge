# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — native C and Fortran window-cache memory maps
"""Read uncompressed NPY members in their declared storage order."""
from pathlib import Path
import struct
import zipfile
import numpy as np


def mapped_array(path: Path, name: str) -> np.memmap:
    """Map numeric cache members without loading the full ZIP payload."""
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(name + '.npy')
        assert member.compress_type == zipfile.ZIP_STORED
    with path.open('rb') as stream:
        stream.seek(member.header_offset)
        name_size, extra_size = struct.unpack('<HH', stream.read(30)[26:30])
        stream.seek(name_size + extra_size, 1)
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
    assert not dtype.hasobject
    return np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=shape, order='F' if fortran else 'C')
