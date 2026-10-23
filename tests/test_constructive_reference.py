# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — isolated reference runtime integration
"""Exercise the complete legacy inverse calculation and its export boundary.

These integrations require CONSTRUCTIVE_REFERENCE_PYTHON to name the compatible
reference interpreter with the pinned coverage observer installed. Fault cases
run the genuine inverse calculation before corrupting its exported coil state;
they test refusal of unsafe provider output, not solver accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.constructive_pairs import SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_python() -> Path:
    """Require the separately provisioned runtime for numerical integrations."""
    value = os.environ.get("CONSTRUCTIVE_REFERENCE_PYTHON")
    if value is None:
        pytest.skip(
            "Requires the pinned legacy interpreter; executed in the dedicated "
            "constructive integration job, not the fast diagnostic suite."
        )
    return Path(value)


@pytest.fixture
def reference_binding(tmp_path: Path) -> Path:
    """Bind the real repository sources without substituting a solver fixture."""
    path = tmp_path / "binding.json"
    path.write_text(
        json.dumps(
            {"source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}}
        )
    )
    return path


def run_reference(
    python: Path, binding: Path, output: Path, script: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the production entrypoint with real interpreter and file boundaries."""
    target = [str(script)] if script is not None else ["-m", "tools.constructive_reference"]
    return subprocess.run(
        [
            str(python),
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=tools",
            *target,
            "--binding",
            str(binding),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.mark.integration
def test_reference_cli_exports_actual_inverse(
    reference_python: Path, reference_binding: Path, tmp_path: Path
) -> None:
    """Solve the genuine limiter-aware inverse and read its generated artifacts."""
    output = tmp_path / "reference"
    result = run_reference(reference_python, reference_binding, output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "returned"
    assert report["coil_labels"] == [f"FC{i}" for i in range(1, 19)]
    assert report["coil_turns"] == [1] * 18
    assert report["current_romberg_A"] == pytest.approx(1e6, rel=1e-10)
    assert report["current_rectangle_A"] == pytest.approx(1e6, rel=0.01)
    currents = np.load(output / "currents.npy", allow_pickle=False)
    assert currents.shape == (18,) and np.isfinite(currents).all()
    assert np.max(np.abs(currents)) <= 5e6
    with np.load(output / "maps.npz", allow_pickle=False) as maps:
        assert maps["psi_Wb_per_rad"].shape == (65, 65)
        assert np.isfinite(maps["psi_Wb_per_rad"]).all()
        assert maps["core_mask"].shape == (65, 65)
        assert set(np.unique(maps["core_mask"])) <= {0.0, 1.0}
        area = np.diff(maps["R_m"])[0] * np.diff(maps["Z_m"])[0]
        assert np.sum(maps["current_A_m2"]) * area == pytest.approx(report["current_rectangle_A"])


@pytest.mark.parametrize("bad_current", ["float('nan')", "6e6"])
@pytest.mark.integration
def test_reference_refuses_corrupted_native_output(
    reference_python: Path, reference_binding: Path, tmp_path: Path, bad_current: str
) -> None:
    """Retain failure after a real solve followed by an unsafe coil-state fault."""
    script = tmp_path / "provider_fault.py"
    script.write_text(
        "import freegs.picard\n"
        "from tools.constructive_reference import main\n"
        "original = freegs.picard.solve\n"
        "def solve_then_corrupt(eq, *args, **kwargs):\n"
        "    original(eq, *args, **kwargs)\n"
        f"    eq.tokamak.coils[0][1].current = {bad_current}\n"
        "freegs.picard.solve = solve_then_corrupt\n"
        "raise SystemExit(main())\n"
    )
    output = tmp_path / "refused"
    result = run_reference(reference_python, reference_binding, output, script)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["exception"].startswith("ValueError:")
    assert "bounded" in report["exception"] and "coil currents" in report["exception"]
    assert not (output / "currents.npy").exists()
    assert not (output / "maps.npz").exists()


def test_reference_rejects_wrong_runtime(reference_binding: Path, tmp_path: Path) -> None:
    """Reject the actually incompatible modern numerical runtime."""
    output = tmp_path / "wrong-runtime"
    result = run_reference(Path(sys.executable), reference_binding, output)
    assert result.returncode != 0
    assert "Unqualified reference environment" in result.stderr
    assert not output.exists()


def test_reference_rejects_changed_source(
    reference_python: Path, reference_binding: Path, tmp_path: Path
) -> None:
    """Reject corrupted provenance independently in the correct runtime."""
    output = tmp_path / "changed-source"
    binding = json.loads(reference_binding.read_text())
    binding["source_sha256"][SOURCE_PATHS[0]] = "0" * 64
    reference_binding.write_text(json.dumps(binding))
    result = run_reference(reference_python, reference_binding, output)
    assert result.returncode != 0
    assert "Source changed after binding" in result.stderr
    assert not output.exists()
