# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — public DIII-D archive of the v25+v28 ensemble (entrant: "submit", 2026-09-29)
"""Run the #949223 package's own DIII-D entrypoint, with its exact flags, on the 874 public-test shots, with the
130 networks: the 65 package networks (v25, checksums verified against the manifest) followed by the 65 v28 networks
(networks_v28.sha verified). The MAST archive is the unchanged #949223 one (hard link after SHA check).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = Path('$LAB/agentic_group_workspace/'
           'second_agent-recovery-20260926/results/joint_fixed_axis_volume_reproduction_56cb')
MAST = PKG.parent / 'joint_submission_56cb/mast_public_test.npz'
MAST_SHA = '50b50184'
DATA = Path('$DISK/_offroot_cache/fec_data_local')
OUT = HERE / 'submission_v25v28'
FLAGS = ['--kept-coordinates', '45', '--smooth-from', '30', '--smooth-axis', '0', '--shape', 'kappa', 'tri_top', 'tri_bot',
         'volume', 'li', '--x-points', '--reshape-after-x-points', '--geometric-share', '0.5', '--li-share', '0.6',
         '--edit-margin', '1', '--contour', '--contour-first', '--contour-share', '0.5', '--contour-rounds', '1',
         '--scalar-share', 'kappa=0.5', 'volume=0.5', 'tri_top=0.25', 'tri_bot=0.25']


def digest(p):
    with Path(p).open('rb') as s:
        return hashlib.file_digest(s, 'sha256').hexdigest()


def main():
    manifest = json.loads((PKG / 'package_manifest.json').read_text())
    v25 = []
    for item in manifest['recipe']:
        p = PKG / item['path']
        assert digest(p) == item['sha256'], p
        v25.append(str(p))
    v28 = []
    for line in (HERE / 'networks_v28.sha').read_text().splitlines():
        sha, p = line.split(maxsplit=1)
        assert digest(p) == sha, p
        v28.append(p)
    assert len(v25) == 65 and len(v28) == 65
    basis = PKG / manifest.get('basis_path', 'models/flux_basis.npz')
    assert digest(basis) == manifest['basis_sha256']
    assert digest(MAST).startswith(MAST_SHA)
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(PKG / 'source/fec_adapter') + os.pathsep + str(PKG / 'source/fec_adapter/vendor'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1')
    cmd = [sys.executable, str(PKG / 'entrypoints/predict_diii_d.py'), '--data-root', str(DATA), '--out', str(OUT),
           '--workers', '6', '--split', 'diii_d_public_test', '--network', *v25, *v28, '--basis', str(basis), *FLAGS]
    t0 = time.time()
    code = subprocess.run(cmd, env=env).returncode
    receipt = {'command_flags': FLAGS, 'networks_v25': v25, 'networks_v28': v28, 'basis_sha256': manifest['basis_sha256'],
               'data_root': str(DATA), 'exit_code': code, 'seconds': time.time() - t0}
    if code == 0:
        if not (OUT / 'mast_public_test.npz').exists():
            os.link(MAST, OUT / 'mast_public_test.npz')
        receipt['diii_d_sha256'] = digest(OUT / 'diii_d_public_test.npz')
        receipt['mast_sha256'] = digest(OUT / 'mast_public_test.npz')
    (OUT / 'build_receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ('exit_code', 'seconds')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
