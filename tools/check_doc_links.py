# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — published documentation links and anchors
"""Check relative links, images and heading anchors in the published documents.

GitHub Pages publishes docs/ without docs/internal. Relative links must stay in
that public root; repository files outside it use their GitHub URL instead.
Every relative fragment must name a heading of the destination Markdown page.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """Return the identifier used by the existing published documents.

    Parameters
    ----------
    heading : str
        Markdown heading text without its leading hashes.

    Returns
    -------
    str
        Lowercase identifier with punctuation removed and spaces replaced.
    """
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(page: Path) -> set[str]:
    """Read heading identifiers outside the page's fenced code examples.

    Parameters
    ----------
    page : Path
        UTF-8 Markdown document.

    Returns
    -------
    set[str]
        Identifiers of headings outside triple-backtick fences.
    """
    found: set[str] = set()
    fence = False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("#"):
            found.add(slug(line.lstrip("#")))
    return found


def check(docs: Path) -> tuple[int, list[str]]:
    """Check links and images in the published top-level Markdown pages.

    Parameters
    ----------
    docs : Path
        Published documentation root. Its private ``internal`` subtree is never
        accepted as a link destination.

    Returns
    -------
    tuple[int, list[str]]
        Number of pages examined and unresolved or private link descriptions.
    """
    docs = docs.resolve()
    errors: list[str] = []
    pages = sorted(docs.glob("*.md"))
    if not pages:
        return 0, ["No published Markdown pages found."]
    for page in pages:
        if page.resolve().parent != docs:
            errors.append(f"{page.name} leaves the published docs/ folder")
            continue
        text = page.read_text(encoding="utf-8")
        for pattern in (LINK, IMAGE):
            for target in pattern.findall(text):
                if re.match(r"[a-z]+:", target) or target.startswith("//"):
                    continue
                path, _, anchor = target.partition("#")
                resolved = (page.parent / path).resolve() if path else page
                if not resolved.is_relative_to(docs) or resolved.is_relative_to(docs / "internal"):
                    errors.append(f"{page.name}: {target} leaves the published docs/ folder")
                elif not resolved.exists():
                    errors.append(f"{page.name}: {target} does not exist")
                elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                    errors.append(f"{page.name}: {target} names no heading of {resolved.name}")
    return len(pages), errors


def main(argv: Sequence[str] | None = None) -> int:
    """Check the published documentation and return a failing status on errors.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero when every link resolves; one for invalid links or unreadable pages.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    try:
        count, errors = check(args.docs_root)
    except (OSError, UnicodeError):
        print("Unable to read the published documentation.")
        return 1
    print("\n".join(errors) if errors else f"{count} pages, every relative link resolves")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
