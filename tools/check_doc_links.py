# SPDX-License-Identifier: AGPL-3.0-or-later
# Fusion Equilibrium Challenge entry — every relative link of the published site must land on a published page
"""python tools/check_doc_links.py

GitHub Pages publishes docs/ only (without docs/internal). A relative link from a page in docs/ to a file outside
it resolves to a path that does not exist on the site; such files are linked by their GitHub URL instead. The
check also requires every #anchor of a relative link to name a heading of the target page.
"""
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(page: Path) -> set[str]:
    found, fence = set(), False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("#"):
            found.add(slug(line.lstrip("#")))
    return found


def main() -> int:
    errors = []
    pages = [p for p in DOCS.glob("*.md")]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for pattern in (LINK, IMAGE):
            for target in pattern.findall(text):
                if re.match(r"[a-z]+:", target) or target.startswith("//"):
                    continue
                path, _, anchor = target.partition("#")
                resolved = (page.parent / path).resolve() if path else page
                if not resolved.is_relative_to(DOCS) or resolved.is_relative_to(DOCS / "internal"):
                    errors.append(f"{page.name}: {target} leaves the published docs/ folder")
                elif not resolved.exists():
                    errors.append(f"{page.name}: {target} does not exist")
                elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                    errors.append(f"{page.name}: {target} names no heading of {resolved.name}")
    print("\n".join(errors) if errors else f"{len(pages)} pages, every relative link resolves")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
