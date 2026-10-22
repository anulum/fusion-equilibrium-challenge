# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — shot-level surface-support reporting
"""Exercise real measured aggregates and explicit availability edge cases."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tools.surface_support_summary import Shot, resample, summarise

QUALIFIED = ["d3d_shot_0093deb434", "d3d_shot_862555f165", "d3d_shot_fedaca60be"]


def test_empty_shots_drop_and_renormalise_without_redraw() -> None:
    """The undefined draw stays counted, while duplicate supported shots keep their multiplicity."""
    values = np.array([1.0, np.nan, 5.0])
    draws = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 1, 1], [1, 2, 1]])
    result = resample(values, draws)
    finite_means = np.array([1.0, 1.0, 11 / 3, 5.0])
    assert result["mean_change"] == 3.0
    assert result["ci99"] == pytest.approx(np.quantile(finite_means, [0.005, 0.995]).tolist())
    assert result["supported_shots"] == 2
    assert result["draws"] == 5 and result["finite_draws"] == 4 and (result["unavailable_draws"] == 1)
    assert result["sampled_nonempty_histogram"] == [1, 2, 0, 2]


def test_all_empty_support_has_no_estimate_or_interval() -> None:
    """Unavailable measurements cannot become a zero effect or a zero-width interval."""
    result = resample(np.full(2, np.nan), np.array([[0, 0], [0, 1], [1, 1]]))
    assert result["mean_change"] is None
    assert result["ci99"] is None
    assert result["supported_shots"] == result["finite_draws"] == 0
    assert result["draws"] == result["unavailable_draws"] == 3
    assert result["sampled_nonempty_histogram"] == [3, 0, 0]


def test_complete_support_is_the_equal_shot_bootstrap() -> None:
    """Longer shots do not gain extra weight; each supplied value is a shot frame mean."""
    short = np.array([2.0])
    long = np.full(100, 8.0)
    values = np.array([short.mean(), long.mean()])
    draws = np.random.default_rng(25025).integers(0, 2, size=(20000, 2))
    result = resample(values, draws)
    assert result["mean_change"] == 5.0
    assert result["mean_change"] != np.concatenate([short, long]).mean()
    assert result["ci99"] == np.quantile(values[draws].mean(axis=1), [0.005, 0.995]).tolist()
    assert result["finite_draws"] == 20000 and result["unavailable_draws"] == 0
    assert result["sampled_nonempty_histogram"] == [0, 0, 20000]


@pytest.mark.parametrize(
    "case",
    [
        "empty-values",
        "2d-values",
        "infinite",
        "1d-draws",
        "no-draws",
        "short-draws",
        "float-draws",
        "negative",
        "outside",
    ],
)
def test_invalid_resampling_contract_refused(case: str) -> None:
    """Reject malformed resampling plans instead of changing their unit or cohort."""
    values = np.array([1.0, 2.0])
    draws = np.array([[0, 1], [1, 1]])
    if case == "empty-values":
        values = np.array([], dtype=float)
    elif case == "2d-values":
        values = values[:, None]
    elif case == "infinite":
        values[0] = np.inf
    elif case == "1d-draws":
        draws = draws[0]
    elif case == "no-draws":
        draws = np.empty((0, 2), dtype=np.int64)
    elif case == "short-draws":
        draws = draws[:, :1]
    elif case == "float-draws":
        with pytest.raises(ValueError, match="Invalid"):
            resample(values, draws.astype(float))
        return
    elif case == "negative":
        draws[0, 0] = -1
    else:
        draws[0, 0] = 2
    with pytest.raises(ValueError, match="Invalid"):
        resample(values, draws)


@pytest.fixture
def measured_shots() -> list[Shot]:
    """Three source-bound qualification records, in the independently fixed cohort order."""
    data_path = Path(__file__).resolve().parents[1] / "experiments/d71b_surface_support/per_shot_v1.json.gz"
    with gzip.open(data_path, "rt") as stream:
        data = json.load(stream)
    selected = [data["shots"][index] for index in (0, 179, 357)]
    assert [shot["stem"] for shot in selected] == QUALIFIED
    return cast(list[Shot], selected)


def test_real_qualification_report(measured_shots: list[Shot]) -> None:
    """Three measured shots retain their counts and equal-shot historical point estimates."""
    shots = measured_shots
    result = summarise(shots, QUALIFIED)
    assert result["shots"] == 3 and result["frames"] == result["reference_valid"] == 723
    assert result["seed"] == 25025 and result["draws"] == 20000
    assert result["reference_valid_frames_removed_by_full_support"] == 0
    common = cast(dict[str, object], result["common"])
    full = cast(dict[str, object], result["full"])
    assert common == full
    assert common["frames"] == 723 and common["shots"] == 3
    change = cast(dict[str, object], common["change"])
    values = [shot["historical"]["paired_change"] for shot in shots]
    assert all(x is not None for x in values)
    expected = float(np.mean(np.asarray(values, dtype=float)))
    assert change["mean_change"] == expected
    assert change["sampled_nonempty_histogram"] == [0, 0, 0, 20000]
    historical = cast(dict[str, dict[str, object]], result["historical"])
    assert historical["smoothed->margin"]["mean_change"] == expected
    assert historical["reference"]["shot_mean"] == 0
    maps = cast(list[dict[str, object]], result["maps"])
    assert len(maps) == 10
    assert maps[0]["used_level_histogram"] == [0, 0, 0, 0, 723]
    assert result["frame_quantiles_reproduced"] is False
    assert result["private_maps_remeasured"] is False
    json.dumps(result, allow_nan=False)


def test_unavailable_measurement_remains_unavailable(measured_shots: list[Shot]) -> None:
    """A fully unavailable support counterexample cannot become a zero diagnostic."""
    shot = copy.deepcopy(measured_shots[0])
    n = shot["frames"]
    shot["reference_valid"] = 0
    for entry in shot["maps"]:
        entry.update(
            {"measured": 0, "missing_reference": n, "finite_ratio": 0, "used_level_histogram": [0] * 5}
        )
        for level in entry["levels"]:
            level.update(
                {
                    "used": 0,
                    "rank_deficient": 0,
                    "condition_unavailable": 0,
                    "condition_range": None,
                    "rank_histogram": [0] * 3,
                    "zero_source": 0,
                }
            )
            level["reasons"] = {key: n if key == "missing_reference" else 0 for key in level["reasons"]}
            for points in (level["vertices"], level["finite_points"]):
                points.update({"count": 0, "total": 0, "minimum": None, "maximum": None})
    shot["historical"] = {
        "counts": [0] * 3,
        "excess_means": [None] * 3,
        "paired_frames": 0,
        "paired_change": None,
    }
    shot["common"] = {"frames": 0, "raw_means": None, "excess_means": None, "change": None}
    shot["full"] = copy.deepcopy(shot["common"])
    result = summarise([shot], [shot["stem"]])
    assert result["frames"] == n and result["reference_valid"] == 0
    common = cast(dict[str, object], result["common"])
    assert common == result["full"]
    assert common["frames"] == common["shots"] == 0
    assert common["raw_means"] is common["excess_means"] is None
    change = cast(dict[str, object], common["change"])
    assert change["mean_change"] is change["ci99"] is None
    assert change["finite_draws"] == 0 and change["unavailable_draws"] == 20000
    maps = cast(list[dict[str, object]], result["maps"])
    assert maps[0]["measured"] == maps[0]["finite_ratio"] == 0
    assert maps[0]["missing_reference"] == n
    json.dumps(result, allow_nan=False)


def test_removed_frames_use_common_reference_valid_support(measured_shots: list[Shot]) -> None:
    """A controlled subset count excludes only frames that were common-reference-valid."""
    shot = copy.deepcopy(measured_shots[0])
    shot["full"]["frames"] -= 1
    result = summarise([shot], [shot["stem"]])
    assert result["reference_valid_frames_removed_by_full_support"] == 1


@pytest.mark.parametrize("case", ["reordered", "duplicated"])
def test_wrong_cohort_order_refused(measured_shots: list[Shot], case: str) -> None:
    """The same measurements cannot silently change their bootstrap identity or appear twice."""
    shots = list(reversed(measured_shots)) if case == "reordered" else [measured_shots[0]] * 2
    expected = QUALIFIED if case == "reordered" else [QUALIFIED[0]] * 2
    with pytest.raises(ValueError, match="order"):
        summarise(shots, expected)


def test_empty_cohort_refused() -> None:
    """An empty export cannot generate a report."""
    with pytest.raises(ValueError, match="No shots"):
        summarise([], [])
