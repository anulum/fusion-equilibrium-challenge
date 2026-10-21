# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — publication boundary tests
"""Check the real public projection and refuse the historical process-record shapes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_publication_boundary import ROOT, TEXT_LIMIT, check, classify, main, projection, tree

D39 = "experiments/D39_joint_finisher_review_2026-10-11T083412Z"
TOOL = "check_publication_boundary.py"
D42 = "experiments/D42_X_surrogate_official_metric_diagnostic_2026-10-12T113715Z"
# Built with keyword arguments so that this file itself holds no envelope keys in quoted-key form.
RECEIPT = dict(
    canonical_receipt=dict(
        seq=1, payload=dict(client_msg_id="cli-0", delivered=True, matched_recipients=["a"])
    )
)
EVENTS = [dict(seq=1, kind="chat", payload=dict(client_msg_id="cli-0", hub_id="hub", msg_id=1))]
# The six process records found in the public experiment export, by their actual published paths.
D39_AUDIT = "experiments/D39_first_agent_D38_map_objective_source_audit_2026-10-11"
HISTORICAL = {
    f"{D39_AUDIT}/REVIEW_D40_design_and_code_first_agent.md": b"# Review of D40\n\nNo error found.\n",
    f"{D39}/peer_amendment4_review_delivery.json": json.dumps(EVENTS).encode(),
    f"{D39}/peer_amendment4_review_cli.json": b'{"returncode": 3, "stdout": "delivery unknown"}',
    f"{D39}/peer_amendment4_review_reply.txt": b"the second agent review: accepted.\n",
    f"{D42}/terminal_results_peer_review_delivery_v1.json": json.dumps(RECEIPT).encode(),
    f"{D42}/terminal_results_peer_review_request_v1.txt": b"Please review actual S.\n",
}


def test_public_projection() -> None:
    """The current public projection, untracked candidates included, holds no refused file."""
    files = projection(ROOT)
    assert len(files) > 2000
    assert Path("tools/check_publication_boundary.py") in files
    assert check(ROOT, files) == []


@pytest.mark.parametrize("name", sorted(HISTORICAL))
def test_historical_records_refused(tmp_path: Path, name: str) -> None:
    """Each actual historical process record is refused in an exported tree."""
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(HISTORICAL[name])
    assert check(tmp_path, tree(tmp_path)) == [f"process_record: {name}"]


@pytest.mark.parametrize("payload", [RECEIPT, EVENTS])
def test_message_wrapper_under_any_name(payload: object) -> None:
    """A bus envelope or delivery receipt is refused whatever the file is called."""
    content = json.dumps(payload, indent=2).encode()
    assert classify(Path(f"{D39}/analysis.json"), content) == "message_wrapper"


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
        ("experiments/x/weights.bin", b"\0" + json.dumps(EVENTS).encode()),
        ("experiments/x/large.json", None),
    ],
)
def test_scientific_records_admitted(name: str, content: bytes | None) -> None:
    """Scientific files pass although their names or directories contain review, peer or delivery."""
    assert classify(Path(name), content) is None


def test_large_file_not_read(tmp_path: Path) -> None:
    """A file above the text limit is classified by its path only; a missing listed path is skipped."""
    path = tmp_path / "large.json"
    path.write_bytes(json.dumps(EVENTS).encode() + b" " * TEXT_LIMIT)
    assert check(tmp_path, [Path("large.json"), Path("deleted.json")]) == []


def test_git_projection(tmp_path: Path) -> None:
    """The projection holds tracked and untracked files but not ignored ones."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("x")
    (tmp_path / "tracked.md").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.md"], check=True)
    (tmp_path / "untracked_peer_review_reply.txt").write_text("x")
    expected = [Path(".gitignore"), Path("tracked.md"), Path("untracked_peer_review_reply.txt")]
    assert projection(tmp_path) == expected
    assert main(["--root", str(tmp_path)]) == 1


def test_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Without a git executable on the search path the projection cannot be listed."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert main([]) == 1
    assert capsys.readouterr().out == "Unable to list or read the projection.\n"


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command passes the real repository and fails refused, unlisted or unreadable input."""
    assert main([]) == 0
    assert "none outside the publication boundary" in capsys.readouterr().out
    exported = tmp_path / "experiments"
    exported.mkdir()
    (exported / "peer_amendment4_review_reply.txt").write_text("accepted\n")
    assert main(["--tree", str(exported)]) == 1
    assert capsys.readouterr().out == "process_record: peer_amendment4_review_reply.txt\n"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["--root", str(plain)]) == 1
    assert capsys.readouterr().out == "Unable to list or read the projection.\n"
    locked = plain / "locked.json"
    locked.write_text("{}")
    locked.chmod(0)
    try:
        assert main(["--tree", str(plain)]) == 1
    finally:
        locked.chmod(0o600)
    assert capsys.readouterr().out == "Unable to list or read the projection.\n"


def test_real_cli() -> None:
    """Exercise the deployed command and its default repository-root discovery."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--parallel-mode", str(ROOT / "tools" / TOOL)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.stdout.endswith(" files, none outside the publication boundary\n")
