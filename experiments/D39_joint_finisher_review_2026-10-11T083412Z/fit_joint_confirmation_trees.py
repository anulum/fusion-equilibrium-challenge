# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — eligible trees for the locked joint candidate
"""Replace the D38-only entry guard with the joint lock, retaining tree fitting."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D38 = HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z'


def main():
    lock_path = HERE / 'concrete_confirmation_candidate_lock.json'
    lock = json.loads(lock_path.read_text())
    analysis_path = HERE / 'central_development_analysis.json'
    analysis = json.loads(analysis_path.read_text())
    assert lock['one_candidate_only'] and lock['locked_before_reserved_candidate_scoring']
    assert lock['winner'] == analysis['one_joint_winner']
    assert lock['central_analysis_sha256'] == hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    assert analysis['status'] == 'all358_central_locked_finalists_complete'
    assert json.loads((D38 / 'confirmation_tree_qualification/receipt.json').read_text())['status'] == 'real_eligible_tree_input_export_path_qualified'
    assert not (HERE / 'confirmation/analysis.json').exists()
    assert not (D38 / 'confirmation_trees/receipt.json').exists()
    sys.path.insert(0, str(D38))
    import fit_confirmation_trees as original
    original.fit_all(qualify=False)


if __name__ == '__main__':
    main()
