# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — publication boundary of the public projection
"""Refuse private custody and process-only records in the public projection.

The projection is what Git would publish: tracked files plus untracked files
that are not ignored. A directory, such as a freshly exported experiments/
tree, can be checked instead. Three kinds of file are refused:

* private custody: paths under docs/internal, private or a coordination
  directory, and coordination records named as handovers, session logs,
  message inboxes or bus records;
* message wrappers: text containing at least two quoted keys of a
  message-bus envelope or delivery receipt, whatever the file is called;
* process-only assessments: file names that mark a review as a peer review,
  a review request, delivery, reply or acknowledgement, or a review of one
  agent's work, and any delivery record that is not code. A program named
  for a delivery step (such as an upload script) is source, not a record.

A scientific file is not refused because its name or its directory contains
the word review; the name must also mark it as process.
"""

from __future__ import annotations

import argparse
import re
import shutil

# subprocess runs only the resolved git binary with a fixed argument list and no shell.
import subprocess  # nosec B404
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIRS = (("docs", "internal"), ("private",), (".coordination",), ("agentic_group_workspace",))
CUSTODY_NAME = re.compile(r"(?i)(handover|session_log|full_session_log|synapse|inbox|waiter)")
ENVELOPE_KEYS = (
    "client_msg_id",
    "msg_id",
    "message_id",
    "message_seq",
    "hub_id",
    "matched_recipients",
    "recipients",
    "recipient_wake_capabilities",
    "dead_lettered",
    "canonical_receipt",
)
ENVELOPE = re.compile(r'"(' + "|".join(ENVELOPE_KEYS) + r')"\s*:')
PROCESS_MARKS = {"peer", "request", "delivery", "reply", "ack", "agent", "first", "second"}
CODE_SUFFIXES = {".py", ".sh"}
TEXT_LIMIT = 4_000_000


def tokens(name: str) -> set[str]:
    """Split a file name into lowercase words.

    Parameters
    ----------
    name : str
        File name with or without its suffix.

    Returns
    -------
    set[str]
        Alphabetic words of the name; digits and separators split words.
    """
    return set(re.findall(r"[a-z]+", name.lower()))


def classify(relative: Path, content: bytes | None) -> str | None:
    """Return why one file may not be published, or None when it may.

    Parameters
    ----------
    relative : Path
        Path of the file relative to the projection root.
    content : bytes or None
        File bytes, or None when they were not read (too large).

    Returns
    -------
    str or None
        ``private_custody``, ``message_wrapper`` or ``process_record``; None
        for an admissible file.
    """
    parts = relative.parts
    for private in PRIVATE_DIRS:
        if any(parts[i : i + len(private)] == private for i in range(len(parts) - len(private) + 1)):
            return "private_custody"
    if CUSTODY_NAME.search(relative.name):
        return "private_custody"
    words = tokens(relative.name)
    record = relative.suffix not in CODE_SUFFIXES
    if ("delivery" in words and record) or ("review" in words and words & PROCESS_MARKS):
        return "process_record"
    if content is not None and b"\0" not in content[:8192]:
        found = set(ENVELOPE.findall(content.decode("utf-8", errors="replace")))
        if len(found) >= 2:
            return "message_wrapper"
    return None


def projection(root: Path) -> list[Path]:
    """List what Git would publish from a work tree.

    Parameters
    ----------
    root : Path
        Git work tree.

    Returns
    -------
    list[Path]
        Tracked files and untracked files that are not ignored, relative to
        ``root``, in sorted order.

    Raises
    ------
    subprocess.CalledProcessError
        ``root`` is not a Git work tree.
    FileNotFoundError
        No git executable is on the search path.
    """
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    listed = subprocess.run(  # nosec B603
        [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted({Path(name) for name in listed.split("\0") if name})


def tree(directory: Path) -> list[Path]:
    """List every regular file below a directory.

    Parameters
    ----------
    directory : Path
        Root of an exported tree.

    Returns
    -------
    list[Path]
        Files relative to ``directory`` in sorted order.
    """
    return sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())


def check(root: Path, files: Iterable[Path]) -> list[str]:
    """Classify every listed file and describe the refused ones.

    Parameters
    ----------
    root : Path
        Directory the listed paths are relative to.
    files : Iterable[Path]
        Relative paths to check. A listed path that no longer exists, such as
        a staged deletion, is skipped.

    Returns
    -------
    list[str]
        One ``reason: path`` line per refused file.
    """
    refused: list[str] = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            continue
        content = path.read_bytes() if path.stat().st_size <= TEXT_LIMIT else None
        reason = classify(relative, content)
        if reason:
            refused.append(f"{reason}: {relative.as_posix()}")
    return refused


def main(argv: Sequence[str] | None = None) -> int:
    """Check the public projection or an exported tree.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero when nothing is refused; one when a file is refused or the
        projection cannot be listed or read.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Git work tree to check (default: this one)")
    parser.add_argument("--tree", type=Path, help="check every file below this directory instead")
    args = parser.parse_args(argv)
    try:
        if args.tree is not None:
            root, files = args.tree, tree(args.tree)
        else:
            root, files = args.root, projection(args.root)
        refused = check(root, files)
    except (OSError, subprocess.CalledProcessError):
        print("Unable to list or read the projection.")
        return 1
    print("\n".join(refused) if refused else f"{len(files)} files, none outside the publication boundary")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
