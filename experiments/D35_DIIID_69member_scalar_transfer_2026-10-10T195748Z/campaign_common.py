# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — frozen physics campaign inputs
"""Resolve the existing data and keep campaign writes on Samsung."""

import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('$LAB')
REPO = Path('$HOME/fusion-equilibrium-challenge')
SEAT = ROOT / 'work/first_agent'
SAS = Path('$FEC_RUNS')
TABLES = SAS / 'ml350/runs/tables28_subframe_20261006'
PKG = SAS / 'ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg'
DATA = Path('$FEC_DATA/data/diii_d_train')
D26 = HERE.parent / 'D26_topology_aux_2026-10-08T213532Z'
PARTS = ('training', 'calibration')
V29 = SAS / 'ml350/runs/ensembles_20261007/v29'
D27 = HERE.parent / 'D27_physics_batch_2026-10-09T092900Z'
D31 = HERE.parent / 'D31_DIIID_learning_rate_stability_2026-10-10T085649Z'
FEATURES = 765
sys.path.insert(0, str(REPO / 'fec_adapter/vendor'))
sys.path.insert(0, str(REPO / 'fec_adapter'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def mapped_array(path, name):
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(name + '.npy')
        assert member.compress_type == zipfile.ZIP_STORED
    with Path(path).open('rb') as stream:
        stream.seek(member.header_offset)
        name_size, extra_size = struct.unpack('<HH', stream.read(30)[26:30])
        stream.seek(name_size + extra_size, 1)
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
    assert not fortran and not dtype.hasobject
    return np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=shape)


def metadata(part: str) -> dict:
    with np.load(TABLES / f'table_{part}.npz', allow_pickle=False) as stored:
        stems = stored['stems'].astype(str)
        counts = np.bincount(stored['shot_of_row'], minlength=len(stems))
        labels = tuple(stored['label_names'].astype(str))
        assert str(stored['feature_layout']) == 'discharge'
    bounds = np.r_[0, np.cumsum(counts)]
    return {'stems': stems.tolist(), 'bounds': bounds.tolist(), 'rows': int(bounds[-1]), 'label_names': labels}


def base_maps() -> dict:
    return {part: {name: mapped_array(TABLES / f'table_{part}.npz', name)
                   for name in ('features', 'coordinates', 'labels')} for part in PARTS}


def rows_for(meta: dict, allowed: set) -> np.ndarray:
    return np.concatenate([np.arange(meta['bounds'][i], meta['bounds'][i+1], dtype=np.int64)
                           for i, stem in enumerate(meta['stems']) if stem in allowed])


def cases() -> list:
    cohort = json.loads((HERE / 'cohort.json').read_text())
    meta = metadata('training')
    held = set(cohort['holdout'])
    out = [{'stem': stem, 'frames': meta['bounds'][i+1] - meta['bounds'][i]}
           for i, stem in enumerate(meta['stems']) if stem in held]
    assert len(out) == 358 and sum(c['frames'] for c in out) == 78875
    return sorted(out, key=lambda c: c['stem'])
