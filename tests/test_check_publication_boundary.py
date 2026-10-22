# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — publication boundary tests
"""Check the real public projection and refuse the historical and staged process-record shapes."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_publication_boundary import GZIP_LIMIT, ROOT, check, classify, main, projection, tree

TOOL = "check_publication_boundary.py"
D39 = "experiments/D39_joint_finisher_review_2026-10-11T083412Z"
D42 = "experiments/D42_X_surrogate_official_metric_diagnostic_2026-10-12T113715Z"
# Built with keyword arguments so that this file itself holds no envelope keys in quoted-key form.
RECEIPT = dict(
    canonical_receipt=dict(
        seq=1, payload=dict(client_msg_id="cli-0", delivered=True, matched_recipients=["a"])
    )
)
EVENTS = [dict(seq=1, kind="chat", payload=dict(client_msg_id="cli-0", hub_id="hub", msg_id=1))]
WRAPPER = json.dumps(EVENTS).encode()
SCIENCE = json.dumps(dict(score=0.996359, frames=78875)).encode()
# The six process records found in the public experiment export, by their actual published paths.
D39_AUDIT = "experiments/D39_first_agent_D38_map_objective_source_audit_2026-10-11"
HISTORICAL = {
    f"{D39_AUDIT}/REVIEW_D40_design_and_code_first_agent.md": b"# Review of D40\n\nNo error found.\n",
    f"{D39}/peer_amendment4_review_delivery.json": WRAPPER,
    f"{D39}/peer_amendment4_review_cli.json": b'{"returncode": 3, "stdout": "delivery unknown"}',
    f"{D39}/peer_amendment4_review_reply.txt": b"the second agent review: accepted.\n",
    f"{D42}/terminal_results_peer_review_delivery_v1.json": json.dumps(RECEIPT).encode(),
    f"{D42}/terminal_results_peer_review_request_v1.txt": b"Please review actual S.\n",
}
UNREADABLE = "Unable to list or read the projection.\n"


def git(root: Path, *arguments: str) -> None:
    """Run one git command in a disposable repository."""
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create an empty disposable Git work tree."""
    git(tmp_path, "init", "-q")
    return tmp_path


def test_public_projection() -> None:
    """The current projection, staged blobs and untracked candidates included, holds no refused file."""
    entries = projection(ROOT)
    paths = {entry[0] for entry in entries}
    assert len(paths) > 2000
    assert Path("tools/check_publication_boundary.py") in paths
    assert check(entries) == []


@pytest.mark.parametrize("name", sorted(HISTORICAL))
def test_historical_records_refused(tmp_path: Path, name: str) -> None:
    """Each actual historical process record is refused in an exported tree."""
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(HISTORICAL[name])
    assert check(tree(tmp_path)) == [f"process_record: {name}"]


@pytest.mark.parametrize("payload", [RECEIPT, EVENTS])
def test_message_wrapper_under_any_name(payload: object) -> None:
    """A bus envelope or delivery receipt is refused whatever the file is called."""
    content = json.dumps(payload, indent=2).encode()
    assert classify(Path(f"{D39}/analysis.json"), content) == "message_wrapper"


def test_large_and_compressed_wrappers() -> None:
    """Size does not exempt text, and a wrapper inside a gzip member is found."""
    padded = WRAPPER + b" " * (8 * 2**20)
    assert classify(Path("experiments/x/measurement.json"), padded) == "message_wrapper"
    assert classify(Path("experiments/x/measurement.json.gz"), gzip.compress(padded)) == "message_wrapper"
    assert classify(Path("experiments/x/statistics.json.gz"), gzip.compress(SCIENCE)) is None
    appended = gzip.compress(SCIENCE) + gzip.compress(WRAPPER)
    assert classify(Path("experiments/x/measurements.json.gz"), appended) == "message_wrapper"
    both = gzip.compress(SCIENCE) + gzip.compress(SCIENCE)
    assert classify(Path("experiments/x/measurements.json.gz"), both) is None


@pytest.mark.parametrize(
    "content",
    [
        gzip.compress(b" " * (GZIP_LIMIT + 1)),
        gzip.compress(b" " * (GZIP_LIMIT // 2 + 1)) * 2,
        gzip.compress(WRAPPER)[:-12],
        b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\xff" * 8,
        gzip.compress(SCIENCE) + b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\xff" * 8,
    ],
    ids=["beyond-limit", "members-beyond-limit", "truncated", "corrupt", "corrupt-second-member"],
)
def test_unchecked_archive_refused(content: bytes) -> None:
    """A gzip member that cannot be fully checked is refused, not admitted."""
    assert classify(Path("experiments/x/statistics.json.gz"), content) == "unchecked_archive"


@pytest.mark.parametrize(
    "name",
    [
        "docs/internal/TODO.md",
        "private/tooling/make_public_experiments.py",
        "experiments/x/.coordination/record.md",
        "experiments/agentic_group_workspace/x.py",
        "experiments/x/HANDOVER_2026-10-20T191437Z.md",
        "experiments/x/synapse_inbox_6c83.log",
        "experiments/x/claude_FULL_SESSION_LOG_equilibrium.md",
    ],
)
def test_private_custody_refused(name: str) -> None:
    """Private custody paths and coordination records are refused."""
    assert classify(Path(name), b"text") == "private_custody"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (f"{D39}/analysis.json", b'{"msg_id": 1, "score": 0.99}'),
        (f"{D39}/joint_protocol_peer_ack_32_grid_amendment.json", b'{"amendment": 1}'),
        (f"{D39}/peer_seq141794_X_head_review/X_HEAD_DIAGNOSTICS.md", b"# X head diagnostics\n"),
        ("experiments/D36_production_scalar_submission_2026-10-10T223719Z/run_delivery.py", b"print(1)\n"),
        ("experiments/x/review_of_boundary_metric.md", b"# Boundary metric review\n"),
        ("experiments/x/weights.npz", b"\0" + WRAPPER),
    ],
)
def test_scientific_records_admitted(name: str, content: bytes) -> None:
    """Scientific files pass although their names or directories contain review, peer or delivery."""
    assert classify(Path(name), content) is None


@pytest.mark.parametrize(("staged", "worktree"), [(WRAPPER, SCIENCE), (SCIENCE, WRAPPER), (WRAPPER, None)])
def test_staged_and_worktree_versions(repo: Path, staged: bytes, worktree: bytes | None) -> None:
    """A wrapper in the index or in the work tree is refused whichever version is clean or absent."""
    path = repo / "measurement.json"
    path.write_bytes(staged)
    git(repo, "add", "measurement.json")
    if worktree is None:
        path.unlink()
    else:
        path.write_bytes(worktree)
    assert check(projection(repo)) == ["message_wrapper: measurement.json"]
    assert main(["--root", str(repo)]) == 1


def test_symbolic_links_refused(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Links are refused when staged, untracked or exported, and are never followed."""
    outside = tmp_path_factory.mktemp("outside")
    (outside / "notes.md").write_text("private\n")
    (repo / "staged_link").symlink_to(outside / "notes.md")
    git(repo, "add", "staged_link")
    (repo / "data").symlink_to(outside, target_is_directory=True)
    assert check(projection(repo)) == ["symlink: data", "symlink: staged_link"]
    exported = tmp_path_factory.mktemp("exported")
    (exported / "experiments").symlink_to(outside, target_is_directory=True)
    (exported / "result.json").write_bytes(SCIENCE)
    assert check(tree(exported)) == ["symlink: experiments"]


def test_path_below_linked_directory(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A staged path whose work-tree parent became a link is refused without reading through the link."""
    (repo / "d").mkdir()
    (repo / "d" / "x.json").write_bytes(SCIENCE)
    git(repo, "add", "d/x.json")
    outside = tmp_path_factory.mktemp("outside")
    (outside / "x.json").write_bytes(WRAPPER)
    (repo / "d" / "x.json").unlink()
    (repo / "d").rmdir()
    (repo / "d").symlink_to(outside, target_is_directory=True)
    entries = projection(repo)
    assert (Path("d/x.json"), b"", True) in entries
    assert check(entries) == ["symlink: d", "symlink: d/x.json"]


def test_git_projection(repo: Path) -> None:
    """The projection holds tracked and untracked files but not ignored ones."""
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text("x")
    (repo / "tracked.md").write_text("x")
    git(repo, "add", "tracked.md")
    (repo / "untracked_peer_review_reply.txt").write_text("x")
    entries = projection(repo)
    assert sorted({entry[0] for entry in entries}) == [
        Path(".gitignore"),
        Path("tracked.md"),
        Path("untracked_peer_review_reply.txt"),
    ]
    assert check(entries) == ["process_record: untracked_peer_review_reply.txt"]


def test_missing_staged_object(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An index entry whose blob cannot be read stops the check instead of being skipped."""
    git(repo, "update-index", "--add", "--cacheinfo", f"100644,{'1' * 40},ghost.json")
    assert main(["--root", str(repo)]) == 1
    assert capsys.readouterr().out == UNREADABLE


def test_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Without a git executable on the search path the projection cannot be listed."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert main([]) == 1
    assert capsys.readouterr().out == UNREADABLE


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command passes the real repository and fails refused, unlisted, missing or unreadable input."""
    assert main([]) == 0
    assert "none outside the publication boundary" in capsys.readouterr().out
    exported = tmp_path / "experiments"
    exported.mkdir()
    (exported / "peer_amendment4_review_reply.txt").write_text("accepted\n")
    assert main(["--tree", str(exported)]) == 1
    assert capsys.readouterr().out == "process_record: peer_amendment4_review_reply.txt\n"
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "result.json").write_bytes(SCIENCE)
    assert main(["--tree", str(plain)]) == 0
    assert capsys.readouterr().out == "1 files, none outside the publication boundary\n"
    for argv in (["--root", str(plain)], ["--tree", str(tmp_path / "absent")]):
        assert main(argv) == 1
        assert capsys.readouterr().out == UNREADABLE
    locked = plain / "locked.json"
    locked.write_text("{}")
    locked.chmod(0)
    try:
        assert main(["--tree", str(plain)]) == 1
    finally:
        locked.chmod(0o600)
    assert capsys.readouterr().out == UNREADABLE


def test_real_cli() -> None:
    """Exercise the deployed command and its default repository-root discovery."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--parallel-mode", str(ROOT / "tools" / TOOL)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.endswith(" files, none outside the publication boundary\n")
