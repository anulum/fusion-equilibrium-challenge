# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — repository licence/provenance contract tests
"""Check the actual public projection and refuse lost or incorrectly assigned rights."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reuse", "--root", str(root), "lint", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.fixture
def projection(tmp_path: Path) -> Path:
    """Copy only Git's actual reviewed public candidates, including pending metadata."""
    destination = tmp_path / "public"
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    for name in files:
        if name:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    return destination


def test_public_licence_inventory() -> None:
    """Require complete metadata and preserve the actual third-party licence boundaries."""
    result = _audit(ROOT)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["summary"]["compliant"] is True
    assert report["summary"]["files_total"] == report["summary"]["files_with_licensing_info"]
    licences = {
        file["path"]: {entry["value"] for entry in file["spdx_expressions"]} for file in report["files"]
    }
    assert licences["tools/ci_check_shot.py"] == {"AGPL-3.0-or-later"}
    assert licences["source/d3d_legacy/fec_adapter/scoring_kit/lcfs.py"] == {"MIT"}
    assert licences["source/mast_legacy/vendor/challenge_masks/d3d_envelope.npz"] == {"MIT"}
    assert licences["experiments/D28_pressure_reliability_2026-10-09T202147Z/official_data_fixes.py"] == {
        "MIT"
    }
    assert licences["CODE_OF_CONDUCT.md"] == {"CC-BY-4.0"}
    assert licences["docs/figures/fig6_reconstruction.png"] == {"CC-BY-4.0 AND AGPL-3.0-or-later"}


@pytest.mark.parametrize("fault", ["missing_mit", "unprovided_licence", "lost_annotations"])
def test_licensing_faults(projection: Path, fault: str) -> None:
    """Reject a missing licence text or lost/invalid metadata in the real public projection."""
    if fault == "missing_mit":
        (projection / "LICENSES/MIT.txt").unlink()
    elif fault == "unprovided_licence":
        path = projection / "REUSE.toml"
        text = path.read_text().replace(
            'SPDX-License-Identifier = "AGPL-3.0-or-later"',
            'SPDX-License-Identifier = "LicenseRef-Unprovided"',
        )
        path.write_text(text)
    else:
        (projection / "REUSE.toml").unlink()
    result = _audit(projection)
    assert result.returncode != 0
    assert json.loads(result.stdout)["summary"]["compliant"] is False
