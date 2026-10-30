# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-surface census and refusal tests
"""Use measured D72 support and corrupt its real statistics at the public boundary."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tools.scored_map_bundle import load, load_shot
from tools.scored_map_measure import Measurement, measure
from tools.scored_map_summary import RAW
from tools.scored_surface_summary import Shot, extract, summarise

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "experiments/d72_scored_map_replay/illustration_v1"


@pytest.fixture(scope="module")
def measured() -> tuple[list[Shot], Measurement]:
    """Measure the actual three public frames once, without a model or mocked solver."""
    bundle = load(EXAMPLE, hashlib.sha256((EXAMPLE / "manifest.json").read_bytes()).hexdigest())
    shots = []
    first = None
    for case in bundle.cases:
        data = load_shot(bundle, case)
        result = measure(bundle, data)
        if first is None:
            first = result
        shots.append(
            extract(
                case.stem, result.values[RAW.index("gs_surface")], result.support, data.used, data.converged
            )
        )
    assert first is not None
    return shots, first


def test_measured_support_sensitivity_and_bootstrap(measured: tuple[list[Shot], Measurement]) -> None:
    """Full four-level support leaves the actual common-frame result unchanged."""
    shots, _ = measured
    result = summarise(shots, [s["stem"] for s in shots])
    assert result["frames"] == result["shots"] == 3
    assert result["seed"] == 23023 and result["draws"] == 20000
    assert result["full_support_replaces_primary"] is False
    maps = cast(list[dict[str, object]], result["maps"])
    assert all(m["full"] == 3 and m["used_histogram"] == [0, 0, 0, 0, 3] for m in maps)
    strata = cast(dict[str, dict[str, object]], result["strata"])
    pairs = cast(dict[str, dict[str, object]], strata["all"]["pairs"])
    assert all(p["common"] == p["full"] for p in pairs.values())
    json.dumps(result, allow_nan=False)


def test_explicitly_absent_geometry_keeps_no_support_estimate() -> None:
    """Missing candidate geometry produces unavailable ranges and pairs, not zero residuals."""
    bundle = load(EXAMPLE, hashlib.sha256((EXAMPLE / "manifest.json").read_bytes()).hexdigest())
    data = load_shot(bundle, bundle.cases[0])
    contour = data.contour(0, 0)
    assert contour is not None
    data = replace(
        data,
        contours=contour.copy(),
        offsets=np.array([0, len(contour), len(contour), len(contour)], dtype=np.int64),
        fields=(data.fields[0], data.fields[1], data.fields[1]),
        used=np.zeros(1, dtype=bool),
        converged=np.zeros(1, dtype=bool),
        iterations=np.zeros(1, dtype=np.int64),
        pixels=np.zeros(1, dtype=np.int64),
    )
    result = measure(bundle, data)
    shot = extract(
        data.case.stem, result.values[RAW.index("gs_surface")], result.support, data.used, data.converged
    )
    assert shot["maps"][1]["levels"][0]["condition"]["count"] == 0
    assert shot["strata"]["unused"]["full"]["pairs"]["base->gs"]["change"] is None
    summary = summarise([shot], [shot["stem"]])
    maps = cast(list[dict[str, object]], summary["maps"])
    assert maps[1]["full"] == maps[2]["full"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rank-dtype",
        "empty",
        "ratios",
        "used",
        "converged",
        "levels",
        "full",
        "mismatch",
        "flags",
        "metadata-shape",
        "rank",
        "reason",
        "used-reason",
    ],
)
def test_invalid_measured_metadata_refused(measured: tuple[list[Shot], Measurement], case: str) -> None:
    """Changed output metadata is rejected before it can enter the surface census."""
    _, measured_frame = measured
    ratios = measured_frame.values[RAW.index("gs_surface")].copy()
    support = {k: v.copy() for k, v in measured_frame.support.items()}
    used = converged = np.ones(1, dtype=bool)
    if case == "dtype":
        support["used_levels"] = support["used_levels"].astype(float)
    elif case == "rank-dtype":
        support["rank"] = support["rank"].astype(float)
    elif case == "empty":
        ratios = ratios[:0]
    elif case == "ratios":
        ratios = ratios[:, :2]
    elif case == "used":
        used = used[:0]
    elif case == "converged":
        converged = converged[:0]
    elif case == "levels":
        support["used_levels"] = support["used_levels"][:, :, :2]
    elif case == "full":
        support["full_support"] = support["full_support"][:, :2]
    elif case == "mismatch":
        support["full_support"][:] = False
    elif case == "flags":
        used = np.zeros(1, dtype=bool)
    elif case == "metadata-shape":
        support["condition"] = support["condition"][:, :, :2]
    elif case == "rank":
        support["rank"][0, 0, 0] = 3
    elif case == "reason":
        support["reason"][0, 0, 0] = 5
    else:
        support["reason"][0, 0, 0] = 1
    with pytest.raises(ValueError):
        extract("invalid-original-metadata", ratios, support, used, converged)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "order",
        "duplicate",
        "count",
        "population",
        "strata",
        "levels",
        "full",
        "mass",
        "fit-dimensions",
        "fit-support",
        "unavailable-extrema",
        "range",
        "samples",
        "pair-dimensions",
        "availability",
        "unsupported-pair",
        "pair-availability",
        "full-counts",
        "full-pairs",
    ],
)
def test_changed_statistics_refused(measured: tuple[list[Shot], Measurement], case: str) -> None:
    """Rehashed real aggregates cannot hide invalid dimensions, counts or paired support."""
    shots = copy.deepcopy(measured[0])
    stems = [s["stem"] for s in shots]
    shot = shots[0]
    level = shot["maps"][0]["levels"][0]
    entry = shot["strata"]["all"]["common"]
    if case == "empty":
        shots = []
    elif case == "order":
        shots.reverse()
    elif case == "duplicate":
        stems = [stems[0]] * 3
    elif case == "count":
        shot["frames"] = -1
    elif case == "population":
        shot["maps"].pop()
    elif case == "strata":
        shot["strata"]["unused"]["frames"] += 1
    elif case == "levels":
        shot["maps"][0]["levels"].pop()
    elif case == "full":
        shot["maps"][0]["full"] = 0
    elif case == "mass":
        level["used"] = 0
    elif case == "fit-dimensions":
        level["ranks"].pop()
    elif case == "fit-support":
        level["reasons"][0] = 0
    elif case == "unavailable-extrema":
        level["condition"]["count"] = 0
    elif case == "range":
        level["condition"]["minimum"] = -1
    elif case == "samples":
        level["contour_points"] = {"count": 0, "minimum": None, "maximum": None}
    elif case == "pair-dimensions":
        entry["counts"].pop()
    elif case == "availability":
        entry["means"][0] = None
    elif case == "unsupported-pair":
        entry["pairs"]["base->gs"]["frames"] = 0
    elif case == "pair-availability":
        entry["pairs"]["base->gs"]["change"] = None
    elif case == "full-counts":
        entry["counts"] = [0, 0, 0]
        entry["means"] = [None, None, None]
        for pair in entry["pairs"].values():
            pair.update({"frames": 0, "means": None, "change": None})
    else:
        # Individually admissible marginals can still contradict pairwise subset support.
        shot["frames"] = 2
        shot["strata"]["all"]["frames"] = 2
        shot["strata"]["unused"]["frames"] += 1
        for m in shot["maps"]:
            m["used_histogram"][0] += 1
            for fit_level in m["levels"]:
                fit_level["ranks"][0] += 1
                fit_level["reasons"][1] += 1
        entry["pairs"]["base->gs"].update({"frames": 0, "means": None, "change": None})
    with pytest.raises(ValueError):
        summarise(shots, stems)


@pytest.mark.parametrize("case", ["pin", "schema", "gzip-eof", "deflate", "report", "oversize"])
def test_actual_cli_invalid_inputs(
    measured: tuple[list[Shot], Measurement], tmp_path: Path, case: str
) -> None:
    """Process-level malformed archives return exit2 without tracebacks or private paths."""
    shots, _ = measured
    stems = [s["stem"] for s in shots]
    table = {
        "schema": "scored-surface-shot-statistics.v1",
        "cohort": stems,
        "shots": shots,
    }
    if case == "schema":
        table["schema"] = "invalid"
    path = tmp_path / "data.json.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(" " * (64 * 2**20 + 1) if case == "oversize" else json.dumps(table))
    if case == "gzip-eof":
        path.write_bytes(path.read_bytes()[:-5])
    elif case == "deflate":
        path.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x07\x00")
    report = tmp_path / "report.json"
    report.write_text("{}" if case == "report" else json.dumps(summarise(shots, stems)))
    checksum = "0" * 64 if case == "pin" else hashlib.sha256(path.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--parallel-mode",
            "-m",
            "tools.scored_surface_summary",
            "--table",
            str(path),
            "--table-sha256",
            checksum,
            "--report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout == "Unable to verify the supplied scored-surface support statistics.\n"
    assert str(tmp_path) not in result.stdout
