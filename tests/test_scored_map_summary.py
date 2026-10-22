# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-map shot statistics tests
"""Exercise measured aggregates and explicit whole-shot availability controls."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.scored_map_summary import METRICS, STRATA, Shot, bootstrap, summarise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/d72_scored_map_physics/per_shot_v1.json.gz"
SELECTED = ("d3d_shot_0093deb434", "d3d_shot_862555f165", "d3d_shot_fedaca60be")


@pytest.fixture
def measured() -> list[Shot]:
    """Select three actual measured shots without presenting them as a full cohort."""
    with gzip.open(DATA, "rt") as stream:
        records = json.load(stream)
    indexed = {s["stem"]: s for s in records["shots"]}
    shots = cast(list[Shot], [indexed[stem] for stem in SELECTED])
    assert sum(s["frames"] for s in shots) == 723
    return shots


def test_measured_summary_keeps_matched_means_and_support(measured: list[Shot]) -> None:
    """The actual aggregate preserves its finite pairs, stratum counts and exterior parity."""
    result = summarise(measured, SELECTED)
    assert result["shots"] == 3 and result["frames"] == result["reference_frames"] == 723
    assert result["seed"] == 23023 and result["draws"] == 20000
    strata = cast(dict[str, dict[str, object]], result["strata"])
    assert strata["all"]["frames"] == 723
    metrics = cast(dict[str, dict[str, object]], strata["all"]["measures"])
    assert set(metrics) == set(METRICS)
    surface = cast(dict[str, dict[str, object]], metrics["gs_surface"]["pairs"])["base->gs"]
    entries = [s["statistics"]["all"]["gs_surface"]["pairs"]["base->gs"] for s in measured]
    assert surface["frames"] == sum(e["frames"] for e in entries)
    assert surface["mean_change"] == float(np.mean([e["change"] for e in entries]))
    assert surface["matched_means"] == np.mean([e["means"] for e in entries], axis=0).tolist()
    exterior = cast(dict[str, dict[str, object]], metrics["exterior_abs_current_over_refinside"]["pairs"])[
        "base->gs"
    ]
    assert exterior["mean_change"] == 0 and exterior["ci99"] == [0, 0]
    assert result["physical_force_balance_certified"] is False
    json.dumps(result, allow_nan=False)


def test_unavailable_aggregates_have_no_estimate(measured: list[Shot]) -> None:
    """Unavailable diagnostic support stays null rather than becoming a zero effect."""
    shots = copy.deepcopy(measured)
    for shot in shots:
        for stratum in STRATA:
            for metric in METRICS:
                entry = shot["statistics"][stratum][metric]
                entry["counts"], entry["means"] = [0, 0, 0], [None, None, None]
                for pair in entry["pairs"].values():
                    pair.update({"frames": 0, "means": None, "change": None})
    result = summarise(shots, SELECTED)
    strata = cast(dict[str, dict[str, object]], result["strata"])
    metrics = cast(dict[str, dict[str, object]], strata["all"]["measures"])
    surface = cast(dict[str, dict[str, object]], metrics["gs_surface"]["pairs"])["base->gs"]
    assert surface["mean_change"] is surface["ci99"] is surface["matched_means"] is None
    assert surface["shots"] == surface["finite_draws"] == surface["frames"] == 0
    assert surface["unavailable_draws"] == 20000
    assert metrics["gs_surface"]["means"] == [None, None, None]
    assert result["frames"] == 723


def test_missing_weight_renormalises_without_redrawing() -> None:
    """An empty sampled cohort is counted, while supported multiplicities are retained."""
    weights = np.array([[1, 1, 1], [0, 3, 0], [3, 0, 0]], dtype=np.int64)
    result = bootstrap(np.array([1.0, np.nan, 5.0]), weights)
    assert result["mean_change"] == 3 and result["shots"] == 2
    assert result["finite_draws"] == 2 and result["unavailable_draws"] == 1
    assert result["sampled_nonempty_histogram"] == [1, 0, 1, 1]
    assert np.allclose(cast(list[float], result["ci99"]), [1.01, 2.99])


def test_complete_support_matches_direct_weighting(measured: list[Shot]) -> None:
    """Actual paired shot means use equal original-shot weights, independently recomputed."""
    values = np.array(
        [s["statistics"]["all"]["gs_surface"]["pairs"]["base->gs"]["change"] for s in measured], dtype=float
    )
    weights = np.random.default_rng(23023).multinomial(3, np.full(3, 1 / 3), size=20000)
    result = bootstrap(values, weights)
    expected = weights @ values / 3
    assert result["mean_change"] == float(values.mean())
    assert result["ci99"] == np.quantile(expected, [0.005, 0.995]).tolist()
    assert result["finite_draws"] == 20000 and result["unavailable_draws"] == 0


@pytest.mark.parametrize("case", ["empty", "reordered", "duplicated"])
def test_changed_shot_identity_refused(measured: list[Shot], case: str) -> None:
    """Original identities cannot be lost, repeated or silently reordered."""
    selected = (
        [] if case == "empty" else list(reversed(measured)) if case == "reordered" else [measured[0]] * 3
    )
    with pytest.raises(ValueError, match="order"):
        summarise(selected, SELECTED)


@pytest.mark.parametrize(
    "case",
    ["shape", "empty", "inf", "negative", "mass", "floating", "no_values", "two_dimensional", "scalar"],
)
def test_invalid_weights_refused(case: str) -> None:
    """Malformed cohort dimensions, weights and infinite values fail explicitly."""
    values = np.array([1.0, 2.0])
    weights = np.array([[1, 1]], dtype=np.int64)
    if case == "shape":
        weights = np.ones((2, 1), dtype=np.int64)
    elif case == "empty":
        weights = np.empty((0, 2), dtype=np.int64)
    elif case == "inf":
        values[0] = np.inf
    elif case == "negative":
        weights[0] = [-1, 3]
    elif case == "mass":
        weights[0] = [0, 1]
    elif case == "floating":
        weights = cast(NDArray[np.int64], np.ones((1, 2), dtype=float))
    elif case == "no_values":
        values = np.array([], dtype=float)
    elif case == "two_dimensional":
        values = values[:, None]
    else:
        values = np.array(1.0)
    with pytest.raises(ValueError, match="weights"):
        bootstrap(values, weights)
