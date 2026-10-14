# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — restart a long accelerator job whose log has gone silent

"""Run a command, restart it when its log goes silent, stop when it exits cleanly.

A process hung inside the accelerator runtime never exits: the cards show full
utilisation, the log stops, and waiting for an exit code waits for ever. It
happened three times in a day on this workstation, once for four and a half
hours, and on restart the same shot solved in its usual twenty seconds — so the
hang is not in the inputs and a restart is the remedy. What can be observed
from outside is the log, so silence of the log is the signal.

This only makes sense for a command that resumes where it stopped and logs a
line per unit of work, which :mod:`fec_adapter.physics_base` does. No
accelerator runtime is imported here; the watchdog must not be able to hang the
way its child does.

Events are appended to ``<log>.watchdog`` — ``started``, ``STALL``, ``exited``,
``FINISHED``, ``giving up`` — so that a monitor can follow one small file.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WatchSettings:
    """How patient the watchdog is.

    Attributes
    ----------
    stall_seconds:
        Silence of the log after which the command counts as hung. It has to
        exceed the longest honest gap between two log lines, which for the
        physics base is the first shot's compilation.
    max_restarts:
        Restarts after the first start before the watchdog gives up.
    poll_seconds:
        Interval between looks at the log.
    grace_seconds:
        Time a terminated process group gets before it is killed.
    settle_seconds:
        Pause after a stall before restarting, for the runtime to release the cards.
    """

    stall_seconds: float = 300.0
    max_restarts: int = 80
    poll_seconds: float = 15.0
    grace_seconds: float = 60.0
    settle_seconds: float = 20.0


def watch(command: Sequence[str], log: Path, settings: WatchSettings) -> int:
    """Run ``command`` under the watchdog and return 0 if it finished, 1 if given up on.

    The command's output is appended to ``log``. It runs in a process group of
    its own, and a stall terminates the whole group: the hung work is usually
    in a child of whatever was started.
    """
    events = log.with_name(log.name + ".watchdog")

    def note(text: str) -> None:
        with events.open("a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")

    for attempt in range(settings.max_restarts + 1):
        with log.open("ab") as sink:
            child = subprocess.Popen(  # noqa: S603 - the caller's own command, never a shell string
                list(command), stdout=sink, stderr=subprocess.STDOUT, start_new_session=True
            )
        shown = " ".join(" ".join(command).split())[:160]  # one event, one line
        note(f"started pid {child.pid} (attempt {attempt}): {shown}")
        if _stalled(child, log, settings):
            note(f"STALL: log silent {settings.stall_seconds:.0f}s; terminating group {child.pid}")
            _stop_group(child, settings.grace_seconds)
            time.sleep(settings.settle_seconds)
        if child.returncode == 0:
            note("FINISHED with exit code 0")
            return 0
        note(f"exited with code {child.returncode}")
    note("giving up")
    return 1


def _stalled(child: subprocess.Popen[bytes], log: Path, settings: WatchSettings) -> bool:
    """Wait for the child; return whether the log went silent for too long first."""
    size, changed = -1, time.monotonic()
    while child.poll() is None:
        time.sleep(settings.poll_seconds)
        now = log.stat().st_size if log.exists() else 0
        if now != size:
            size, changed = now, time.monotonic()
        elif time.monotonic() - changed > settings.stall_seconds:
            return child.poll() is None
    return False


def _stop_group(child: subprocess.Popen[bytes], grace_seconds: float) -> None:
    """Terminate the child's process group, and kill it if it does not go."""
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main() -> int:
    """Watch the command given after ``--``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=float, default=WatchSettings.stall_seconds)
    parser.add_argument("--max-restarts", type=int, default=WatchSettings.max_restarts)
    parser.add_argument("command", nargs="+", help="the command and its arguments, after --")
    args = parser.parse_args()
    settings = WatchSettings(stall_seconds=args.stall_seconds, max_restarts=args.max_restarts)
    return watch(args.command, args.log, settings)


if __name__ == "__main__":
    raise SystemExit(main())
