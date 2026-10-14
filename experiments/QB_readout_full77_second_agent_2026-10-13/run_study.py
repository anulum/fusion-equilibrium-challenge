# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed full readout study orchestration
"""Fit the full declared ensemble, then evaluate both cohorts exactly once."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--fitted", type=Path, required=True)
    args = parser.parse_args()
    stages = [("fit", "fit_readouts.py", ["--out", str(args.fitted), "--workers", "4"]),
              ("evaluate", "evaluate_readouts.py", ["--fitted", str(args.fitted)])]
    record = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "stages": []}
    for stage, source, additional in stages:
        command = [sys.executable, "-B", str(args.work / source), "--work", str(args.work),
                   "--runs", str(args.runs), *additional]
        result = subprocess.run(command, check=False)
        record["stages"].append({"stage": stage, "returncode": result.returncode})
        if result.returncode:
            record["status"] = "operational_failure_no_admission"
            break
    else:
        record["status"] = "complete_frozen_full77_study_review_required"
    record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (args.work / "terminal_receipt_v1.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    if record["status"] != "complete_frozen_full77_study_review_required":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
