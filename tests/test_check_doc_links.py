# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — published document link consumer tests
"""Validate the real public documents and refuse broken or private destinations."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_doc_links import DOCS, anchors, check, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """Copy the published Markdown and images, excluding the private backlog."""
    destination = tmp_path / "docs"
    destination.mkdir()
    for page in DOCS.glob("*.md"):
        shutil.copy2(page, destination / page.name)
    for name in ("assets", "figures"):
        shutil.copytree(DOCS / name, destination / name)
    return destination


def test_public_documents() -> None:
    """Resolve every currently published relative page, image and heading link."""
    count, errors = check(DOCS)
    assert count == len(list(DOCS.glob("*.md"))) > 0
    assert errors == []
    assert "result" in anchors(DOCS / "index.md")


@pytest.mark.parametrize("fault", ["outside", "private", "missing", "heading", "symlink"])
def test_invalid_destination(site: Path, fault: str) -> None:
    """Refuse broken links and existing private files, including symlink escapes."""
    target = {
        "outside": "../outside.md",
        "private": "internal/review.md",
        "missing": "missing.md",
        "heading": "STUDY.md#heading-that-does-not-exist",
        "symlink": "escaped.md",
    }[fault]
    if fault == "private":
        (site / "internal").mkdir()
        shutil.copy2(DOCS / "STUDY.md", site / "internal/review.md")
    elif fault == "symlink":
        outside = site.parent / "outside.md"
        outside.write_bytes((DOCS / "STUDY.md").read_bytes() + b"\xff")
        (site / "escaped.md").symlink_to(outside)
    with (site / "index.md").open("a") as stream:
        stream.write(f"\n[Invalid destination]({target})\n")
    count, errors = check(site)
    assert count > 0
    assert any(target in error for error in errors)


def test_local_anchors_and_fences(site: Path) -> None:
    """Accept local headings and network links but refuse headings inside code fences."""
    page = site / "index.md"
    with page.open("a") as stream:
        stream.write(
            "\n[Local](#result)\n[Network](//github.com/anulum)\n"
            "```markdown\n# Fenced heading\n```\n[Fenced](#fenced-heading)\n"
        )
    _, errors = check(site)
    assert len(errors) == 1
    assert "#fenced-heading" in errors[0]


def test_empty_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Reject an empty or wrong documentation root instead of reporting vacuous success."""
    assert check(tmp_path) == (0, ["No published Markdown pages found."])
    assert main(["--docs-root", str(tmp_path)]) == 1
    assert capsys.readouterr().out == "No published Markdown pages found.\n"


def test_unreadable_document(site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Return a controlled failure for damaged UTF-8 in an otherwise real document tree."""
    (site / "index.md").write_bytes(b"\xff")
    assert main(["--docs-root", str(site)]) == 1
    assert capsys.readouterr().out == "Unable to read the published documentation.\n"


def test_real_cli() -> None:
    """Exercise the deployed docs command and its default repository-root discovery."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--parallel-mode", str(ROOT / "tools/check_doc_links.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout == f"{len(list(DOCS.glob('*.md')))} pages, every relative link resolves\n"
