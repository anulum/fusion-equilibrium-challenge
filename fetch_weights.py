# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge entry — fetch and verify the trained weights
"""Download ``models/`` from the public weight repository at its pinned revision and verify every file.

The weights (about 760 MB) are not stored in this Git repository. ``WEIGHTS.json`` names the public repository and
the pinned revision; ``WEIGHTS.sha256`` lists the SHA-256 of every file. Verification alone:

    python fetch_weights.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify() -> list[str]:
    """Return the listed files that are missing or differ from their recorded SHA-256."""
    wrong = []
    for line in (HERE / 'WEIGHTS.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = HERE / name
        if not path.is_file() or digest(path) != expected:
            wrong.append(name)
    return wrong


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='check the files already present; download nothing')
    args = parser.parse_args()
    if not args.verify_only:
        source = json.loads((HERE / 'WEIGHTS.json').read_text())
        if not source.get('revision'):
            print('WEIGHTS.json carries no pinned revision yet', file=sys.stderr)
            return 2
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=source['repo_id'], repo_type=source['repo_type'], revision=source['revision'],
                          allow_patterns=['models/*', 'models/**'], local_dir=HERE)
    wrong = verify()
    if wrong:
        print(f'{len(wrong)} weight files are missing or differ, first: {wrong[0]}', file=sys.stderr)
        return 1
    print('all weight files present and verified')
    return 0


if __name__ == '__main__':
    sys.exit(main())
