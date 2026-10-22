# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — publication boundary of the public projection
"""Refuse private custody and process-only records in the public projection.

The projection is what the next commit would publish: every index entry (its
staged blob and, when present, the work-tree file at the same path) and every
untracked file that is not ignored. Both versions of a staged path are
checked, so neither a staged record hidden behind a clean work-tree file nor
the reverse passes. A directory, such as a freshly exported experiments/ tree,
can be checked instead. Refused:

* symbolic links, staged or in the work tree or the exported directory, and
  work-tree paths below a linked directory; no link is followed and such a
  path is not read, so no path can resolve outside the root;
* private custody: paths under docs/internal, private or a coordination
  directory, and coordination records named as handovers, session logs,
  message inboxes or bus records;
* message wrappers: text containing at least two quoted keys of a
  message-bus envelope or delivery receipt, whatever the file is called and
  however large it is, also inside every member of a gzip file; a gzip file
  that cannot be decompressed, or decompresses beyond GZIP_LIMIT in total, is
  refused unchecked;
* process-only assessments: file names that mark a review as a peer review,
  a review request, delivery, reply or acknowledgement, or a review of one
  agent's work, and any delivery record that is not code. A program named
  for a delivery step (such as an upload script) is source, not a record.

A scientific file is not refused because its name or its directory contains
the word review; the name must also mark it as process. Binary files (a NUL
byte in the first 8 kB, such as NumPy archives and images) are classified by
their path only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil

# subprocess runs only the resolved git binary with fixed argument lists and no shell.
import subprocess  # nosec B404
import zlib
from collections.abc import Sequence
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
GZIP_LIMIT = 64 * 2**20
SYMLINK_MODE = "120000"

# One file version to classify: its path, its bytes, and whether it is a symbolic link.
Entry = tuple[Path, bytes, bool]


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


def text_of(content: bytes) -> str | None:
    """Return the text to scan for envelope keys.

    Parameters
    ----------
    content : bytes
        File bytes; a gzip member is decompressed first.

    Returns
    -------
    str or None
        Decoded text, or None for binary content.

    Raises
    ------
    ValueError
        A gzip member is corrupt or decompresses beyond ``GZIP_LIMIT``.
    """
    if content[:2] == b"\x1f\x8b":
        members: list[bytes] = []
        remaining = content
        while remaining:
            inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
            budget = GZIP_LIMIT + 1 - sum(map(len, members))
            try:
                members.append(inflate.decompress(remaining, budget))
            except zlib.error as error:
                raise ValueError("corrupt gzip member") from error
            if sum(map(len, members)) > GZIP_LIMIT or inflate.unconsumed_tail or not inflate.eof:
                raise ValueError("gzip member too large or truncated")
            remaining = inflate.unused_data
        content = b"".join(members)
    if b"\0" in content[:8192]:
        return None
    return content.decode("utf-8", errors="replace")


def classify(relative: Path, content: bytes, link: bool = False) -> str | None:
    """Return why one file version may not be published, or None when it may.

    Parameters
    ----------
    relative : Path
        Path of the file relative to the projection root.
    content : bytes
        Complete bytes of this version (empty for a work-tree link, which is
        never read; the link target for a staged link).
    link : bool
        Whether this version is a symbolic link.

    Returns
    -------
    str or None
        ``symlink``, ``private_custody``, ``process_record``,
        ``unchecked_archive`` or ``message_wrapper``; None for an admissible file.
    """
    if link:
        return "symlink"
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
    try:
        text = text_of(content)
    except ValueError:
        return "unchecked_archive"
    if text is not None and len(set(ENVELOPE.findall(text))) >= 2:
        return "message_wrapper"
    return None


def _git(root: Path, *arguments: str, stdin: bytes = b"") -> bytes:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603
        [git, *arguments], cwd=root, input=stdin, check=True, capture_output=True
    ).stdout


def _worktree(root: Path, relative: Path) -> Entry:
    for depth in range(1, len(relative.parts) + 1):
        if (root / Path(*relative.parts[:depth])).is_symlink():
            return relative, b"", True
    return relative, (root / relative).read_bytes(), False


def projection(root: Path) -> list[Entry]:
    """Read every file version the next commit could publish from a work tree.

    Parameters
    ----------
    root : Path
        Git work tree.

    Returns
    -------
    list[Entry]
        For each index entry its staged blob and, when the path exists in the
        work tree, the work-tree version; for each untracked, not ignored file
        its work-tree version. Sorted by path.

    Raises
    ------
    subprocess.CalledProcessError
        ``root`` is not a Git work tree or a blob cannot be read.
    FileNotFoundError
        No git executable is on the search path.
    """
    staged = []
    for record in _git(root, "ls-files", "--stage", "-z").decode().split("\0"):
        if record:
            meta, name = record.split("\t", 1)
            mode, sha, _stage = meta.split()
            staged.append((Path(name), sha, mode == SYMLINK_MODE))
    batch = _git(root, "cat-file", "--batch", stdin="".join(f"{sha}\n" for _, sha, _ in staged).encode())
    entries: list[Entry] = []
    offset = 0
    for relative, _sha, link in staged:
        header_end = batch.index(b"\n", offset)
        header = batch[offset:header_end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise OSError(f"staged object not readable: {relative}")
        size = int(header[2])
        entries.append((relative, batch[header_end + 1 : header_end + 1 + size], link))
        offset = header_end + 2 + size
        path = root / relative
        if path.is_symlink() or path.exists():
            entries.append(_worktree(root, relative))
    for name in _git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"):
        if name:
            entries.append(_worktree(root, Path(name)))
    return sorted(entries, key=lambda entry: entry[0])


def tree(directory: Path) -> list[Entry]:
    """Read every file below an exported directory without following links.

    Parameters
    ----------
    directory : Path
        Root of an exported tree.

    Returns
    -------
    list[Entry]
        Regular files and symbolic links (to files or directories), relative
        to ``directory``, sorted by path.

    Raises
    ------
    FileNotFoundError
        ``directory`` does not exist or is not a directory.
    """
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(str(directory))
    entries: list[Entry] = []
    for folder, directories, files in os.walk(directory):
        base = Path(folder)
        for name in directories + files:
            path = base / name
            if path.is_symlink() or name in files:
                entries.append(_worktree(directory, path.relative_to(directory)))
    return sorted(entries, key=lambda entry: entry[0])


def check(entries: Sequence[Entry]) -> list[str]:
    """Classify every file version and describe the refused paths.

    Parameters
    ----------
    entries : Sequence[Entry]
        File versions from :func:`projection` or :func:`tree`.

    Returns
    -------
    list[str]
        One ``reason: path`` line per refused path, without repetitions.
    """
    refused: list[str] = []
    for relative, content, link in entries:
        reason = classify(relative, content, link)
        line = f"{reason}: {relative.as_posix()}"
        if reason and line not in refused:
            refused.append(line)
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
        entries = tree(args.tree) if args.tree is not None else projection(args.root)
        refused = check(entries)
    except (OSError, subprocess.CalledProcessError):
        print("Unable to list or read the projection.")
        return 1
    paths = len({entry[0] for entry in entries})
    print("\n".join(refused) if refused else f"{paths} files, none outside the publication boundary")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
