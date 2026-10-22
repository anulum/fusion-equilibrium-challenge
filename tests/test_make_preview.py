# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — preview image renderer tests
"""Render the real preview from the published frames and refuse damaged or degenerate inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.image import imread

from tools.make_preview import DATA, OUT, bilinear, font, main, render, surfaces

ROOT = Path(__file__).resolve().parents[1]


def frames(tmp_path: Path, **changes: np.ndarray) -> Path:
    """Copy the published frames with some arrays replaced."""
    with np.load(DATA, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays.update(changes)
    path = tmp_path / "frames.npz"
    np.savez(path, **arrays)
    return path


def test_bilinear_is_exact_for_bilinear_fields() -> None:
    """A field linear in R, Z and R*Z is reproduced exactly; points outside the grid are refused."""
    radius, height = np.linspace(1.0, 2.0, 5), np.linspace(-1.0, 1.0, 7)
    rr, zz = np.meshgrid(radius, height)
    field = 0.5 + 2.0 * rr - 3.0 * zz + 1.5 * rr * zz
    points = np.array([[1.0, -1.0], [2.0, 1.0], [1.37, 0.21], [1.5, 0.0]])
    expected = 0.5 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 1.5 * points[:, 0] * points[:, 1]
    assert np.allclose(bilinear(radius, height, field, points), expected, rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="outside"):
        bilinear(radius, height, field, np.array([[2.1, 0.0]]))


def test_surfaces_of_the_published_frame() -> None:
    """The drawn data are the typical frame: axis at 0, boundary at 1, its published distance."""
    shown = surfaces(DATA)
    with np.load(DATA, allow_pickle=False) as stored:
        assert shown.distance_mm == 1000 * float(stored["p50_distance_m"])
        assert np.array_equal(shown.lcfs, stored["p50_margin_lcfs"])
    iz, ir = shown.axis_index
    assert shown.inside[iz, ir] and shown.psin[iz, ir] == 0
    on_boundary = bilinear(shown.radius, shown.height, shown.psin, shown.lcfs)
    assert abs(float(np.mean(on_boundary)) - 1) < 1e-12
    assert float(shown.psin[shown.inside].max()) < 1.2


def test_font_choice() -> None:
    """Inter is used when an Inter file is offered, DejaVu Sans otherwise."""
    dejavu = font_manager.findfont("DejaVu Sans")
    assert font([dejavu]) == "DejaVu Sans"
    assert font([]) == "DejaVu Sans"
    assert font() in {"Inter", "DejaVu Sans"}


def test_render_is_deterministic(tmp_path: Path) -> None:
    """A real render has the committed size and is byte-identical when repeated on one host."""
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    render(DATA, first)
    render(DATA, second)
    assert first.read_bytes() == second.read_bytes()
    assert imread(first).shape[:2] == imread(OUT).shape[:2] == (640, 1280)


@pytest.mark.parametrize(
    "changes",
    [
        {"p50_margin_lcfs": np.array([[9.0, 0.0], [9.1, 0.0], [9.1, 0.1]])},
        {"p50_margin_lcfs": np.array([[1.5, 0.001], [1.5001, 0.001], [1.5001, 0.0011]])},
        {"p50_margin_psi": np.ones((65, 65), dtype=np.float32)},
    ],
    ids=["boundary-outside-grid", "no-plasma-pixel", "flat-flux"],
)
def test_degenerate_frames_refused(tmp_path: Path, changes: dict[str, np.ndarray]) -> None:
    """Frames that cannot define flux surfaces are refused instead of drawn."""
    with pytest.raises(ValueError):
        surfaces(frames(tmp_path, **changes))


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command writes the image, and fails cleanly on missing or damaged frames."""
    out = tmp_path / "preview.png"
    assert main(["--out", str(out)]) == 0
    assert out.exists() and "boundary distance 4.8 mm" in capsys.readouterr().out
    damaged = tmp_path / "damaged.npz"
    np.savez(damaged, R=np.zeros(3))
    for data in (damaged, tmp_path / "absent.npz", frames(tmp_path, p50_margin_psi=np.ones((65, 65)))):
        assert main(["--data", str(data), "--out", str(out)]) == 1
        assert capsys.readouterr().out == "Unable to read or draw the reconstruction frame.\n"


def test_real_cli(tmp_path: Path) -> None:
    """Exercise the deployed command with its default frame file."""
    out = tmp_path / "preview.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            str(ROOT / "tools/make_preview.py"),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout == f"Wrote {out} (boundary distance 4.8 mm)\n"
