# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Equilibrium Challenge — D71b: bind every shot record and the run receipt to the activation and the exact cohort
"""Fail-closed validation of D71b outputs (TODO FEC-R18, after the second agent's review of 20 October 2026).

A shot record is accepted only if it names the expected activation, the expected stem and frame count, the frozen
map order, and the SHA-256 of its measures file, and if that file's arrays have exactly (frames, maps) shape and the
expected measure names. A run receipt is accepted as a cohort result only if it is a complete run of exactly the
activation's cases, in order, with every shot record valid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAPS = ("reference", "raw_64coords", "kept_45coords", "smoothed", "axis", "contour", "shape", "x_points",
        "reshape", "margin")
MEASURES = ("outside_fixed_over_ref", "cp_max", "cp_min", "cp_saddle", "gs_surface", "gs_sse", "gs_tss",
            "gs_pixels", "gs_rank", "gs_rms_residual", "gs_rms_source", "gs_degenerate", "outside_pixels")


class ReceiptError(ValueError):
    """A D71b record or receipt does not match what the activation requires."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_shot(folder: Path, case: dict[str, object], activation_sha256: str) -> dict[str, object]:
    """Return the shot record of ``folder`` if every binding holds, else raise ReceiptError."""
    stem, frames = str(case["stem"]), int(str(case["frames"]))
    record_path, measures_path = folder / "complete.json", folder / "measures.npz"
    if folder.name != stem or not record_path.exists() or not measures_path.exists():
        raise ReceiptError(f"{stem}: record or measures missing")
    record = json.loads(record_path.read_text())
    summary = record.get("summary", {})
    if record.get("activation_sha256") != activation_sha256:
        raise ReceiptError(f"{stem}: record made under another activation")
    if summary.get("stem") != stem or summary.get("frames") != frames or record.get("maps") != list(MAPS):
        raise ReceiptError(f"{stem}: record stem, frame count or map order differs")
    if digest(measures_path) != summary.get("measures_sha256"):
        raise ReceiptError(f"{stem}: measures bytes differ from the record")
    with np.load(measures_path, allow_pickle=False) as z:
        if [str(m) for m in z["maps"]] != list(MAPS):
            raise ReceiptError(f"{stem}: map order in the measures differs")
        if z["times_ms"].shape != (frames,) or z["ip_A"].shape != (frames,):
            raise ReceiptError(f"{stem}: time axis has the wrong length")
        names = set(z.files) - {"maps", "times_ms", "ip_A"}
        if names != set(MEASURES):
            raise ReceiptError(f"{stem}: measure names differ")
        for name in MEASURES:
            if z[name].shape != (frames, len(MAPS)):
                raise ReceiptError(f"{stem}: {name} has shape {z[name].shape}")
    return record


def validate_run(run: Path, cases: list[dict[str, object]], activation_sha256: str) -> list[dict[str, object]]:
    """Return the shot records of a complete run of exactly ``cases`` under the activation, else raise."""
    receipt = json.loads((run / "complete.json").read_text())
    if receipt.get("status") != "complete":
        raise ReceiptError(f"run status is {receipt.get('status')!r}, not a complete cohort run")
    if receipt.get("activation_sha256") != activation_sha256:
        raise ReceiptError("run receipt made under another activation")
    stems = [str(c["stem"]) for c in cases]
    if [m.get("stem") for m in receipt.get("members", [])] != stems:
        raise ReceiptError("run members are not exactly the activation's cases in order")
    if receipt.get("shots") != len(cases) or receipt.get("frames") != sum(int(str(c["frames"])) for c in cases):
        raise ReceiptError("run shot or frame totals differ from the activation's cases")
    present = sorted(p.name for p in run.iterdir() if p.is_dir())
    if present != sorted(stems):
        raise ReceiptError("run folder holds missing or extra shots")
    records = []
    for case, member in zip(cases, receipt["members"], strict=True):
        record = validate_shot(run / str(case["stem"]), case, activation_sha256)
        if record["summary"]["measures_sha256"] != member.get("measures_sha256"):
            raise ReceiptError(f"{case['stem']}: run receipt and shot record disagree")
        records.append(record)
    return records
