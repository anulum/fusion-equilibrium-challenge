# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — reproduce D72 pooled scores and shot bootstrap
"""Verify the frozen D72 gate from public per-shot sufficient statistics.

Reference moments and residual sums are resampled together. This checks score
arithmetic, not the private maps, their physical diagnostics or source authenticity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

QUANTITIES = ("psi", "q95", "betaN", "R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
COUNTERS = (
    *(f"cons_frames_{n}" for n in QUANTITIES[3:]),
    *(f"cons_fails_{n}" for n in QUANTITIES[3:]),
    "lcfs_frames",
    "lcfs_fails",
    "ref_lcfs_dropped",
    "psi_pixels",
    "psi_nonfinite",
    "missing",
)
STAGES = ("base", "gs")
GATE_SHA256 = "41381c826e894168c61a23f333f9ce67af7ef89ce68ed7f03399ec359835f985"
COHORT_SHA256 = "913f7d8c5ba2aedf295cf4e91eeaaa142dd8541c320e03db11fa2652f2c2951d"


class Shot(TypedDict):
    """Hash-bound shot identity, frame support and convergence counts."""

    stem: str
    frames: int
    record_sha256: str
    projection_sha256: str
    used: int
    converged: int


class Statistics(TypedDict):
    """Per-shot reference moments, candidate residuals and diagnostics counters."""

    stems: list[str]
    reference_sum: list[list[float]]
    reference_sumsq: list[list[float]]
    reference_n: list[list[float]]
    residual: dict[str, list[list[float]]]
    distance: dict[str, list[float | None]]
    counters: dict[str, list[list[int]]]
    used: list[int]
    converged: list[int]


class Score(TypedDict):
    """Published pooled components and extraction failure counts for one arm."""

    S: float
    psi: float
    qb: float
    D: float
    C: float
    cons_each: list[float]
    lcfs_fails: int
    cons_fails: int


class Change(TypedDict):
    """Whole-shot multinomial bootstrap comparison in the frozen gate."""

    delta_S: float
    CI99: list[float]
    CI95: list[float]
    P_positive: float
    PASS: bool


class Gate(TypedDict):
    """Frozen headline results reproduced by the sufficient statistics."""

    shots: int
    bootstrap: str
    base: Score
    gs: Score
    gs_minus_base: Change


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cohort(manifest_path: Path, data: Path, gate: Path) -> list[Shot]:
    decoded: object = json.loads(manifest_path.read_text())
    if not isinstance(decoded, dict):
        raise ValueError("Expected a statistics manifest object")
    manifest = cast(dict[str, object], decoded)
    fixed: dict[str, object] = {
        "schema": "d72-pooled-statistics.v1",
        "shots": 358,
        "frames": 78875,
        "stages": list(STAGES),
        "quantities": list(QUANTITIES),
        "counters": list(COUNTERS),
        "draws": 20000,
        "seed": 72020,
        "activation_sha256": "5b156162e3add75896f89e2f44e9b3f98b2151fdd2f61f7e8fe33024b5c8eb0c",
        "run_receipt_sha256": "eef3a6bd6a459033847111f67f35276976a3092aabe63da5f2cd8cca457d02ee",
        "statistics_sha256": "9f70f4a4947f37cca0c7b315ca27973845b5f4acfe79dba6ac6a77865edf6845",
        "gate_sha256": GATE_SHA256,
        "gate_source_sha256": "726d348ff9675817868663b10d9e54948b2cdf6dc74445175e7af76192a9f2c4",
        "cohort_sha256": COHORT_SHA256,
    }
    if set(manifest) != set(fixed) | {"cohort", "export_source_sha256", "data_sha256"}:
        raise ValueError("Unexpected or missing manifest fields")
    if any(manifest[k] != v for k, v in fixed.items()):
        raise ValueError("Manifest differs from the frozen D72 cohort or method")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["export_source_sha256"])):
        raise ValueError("Invalid extraction-source identity")
    if _sha256(data) != manifest["data_sha256"] or _sha256(gate) != GATE_SHA256:
        raise ValueError("Statistics or frozen gate bytes differ")
    encoded = json.dumps(manifest["cohort"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != COHORT_SHA256:
        raise ValueError("Shot order, support or original record identities differ")
    return cast(list[Shot], manifest["cohort"])


def _array(value: object, shape: tuple[int, ...], *, nonnegative: bool = False) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid shape or non-finite statistics")
    if nonnegative and np.any(result < 0):
        raise ValueError("Negative count, square or residual sum")
    return result


def _load(data_path: Path, cohort: list[Shot]) -> Statistics:
    with gzip.open(data_path, "rt", encoding="utf-8") as stream:
        data = cast(Statistics, json.load(stream))
    expected = {
        "stems",
        "reference_sum",
        "reference_sumsq",
        "reference_n",
        "residual",
        "distance",
        "counters",
        "used",
        "converged",
    }
    if set(data) != expected or data["stems"] != [s["stem"] for s in cohort]:
        raise ValueError("Missing, duplicated or reordered shot statistics")
    for name in ("used", "converged"):
        if data[name] != [s[name] for s in cohort]:
            raise ValueError("Per-shot convergence support differs")
    _array(data["reference_sum"], (358, 10))
    _array(data["reference_sumsq"], (358, 10), nonnegative=True)
    counts = _array(data["reference_n"], (358, 10), nonnegative=True)
    frames = np.asarray([s["frames"] for s in cohort], dtype=float)
    if (
        np.any(counts != np.floor(counts))
        or np.any(counts[:, 0] != frames * 65 * 65)
        or np.any(counts[:, 1:] > frames[:, None])
    ):
        raise ValueError("Invalid reference sample counts")
    for values in (data["residual"], data["distance"], data["counters"]):
        if set(values) != set(STAGES):
            raise ValueError("Unexpected candidate arms")
    for arm in STAGES:
        _array(data["residual"][arm], (358, 10), nonnegative=True)
        distance = np.asarray(data["distance"][arm], dtype=float)
        if (
            distance.shape != (358,)
            or np.isinf(distance).any()
            or any(v is not None and not np.isfinite(v) for v in data["distance"][arm])
            or np.any(distance < 0)
        ):
            raise ValueError("Invalid boundary distances")
        counters = _array(data["counters"][arm], (358, 20), nonnegative=True)
        limits = np.column_stack(
            (np.repeat(frames[:, None], 17, axis=1), frames * 65 * 65, frames * 65 * 65, np.ones(358))
        )
        if np.any(counters != np.floor(counters)) or np.any(counters > limits):
            raise ValueError("Invalid extraction counters")
    return data


def _scores(weights: NDArray[np.float64], data: Statistics, arm: str) -> NDArray[np.float64]:
    denominators = []
    residuals = []
    # Preserve the frozen scorer's separate contiguous psi/scalar/consistency operands.
    for group in (0, slice(1, 3), slice(3, 10)):
        totals, squares, counts = (
            weights @ np.ascontiguousarray(np.asarray(value, dtype=float)[:, group])
            for value in (data["reference_sum"], data["reference_sumsq"], data["reference_n"])
        )
        denominators.append(squares - totals**2 / np.maximum(counts, 1.0))
        residuals.append(
            weights @ np.ascontiguousarray(np.asarray(data["residual"][arm], dtype=float)[:, group])
        )
    denominator = np.column_stack(denominators)
    residual = np.column_stack(residuals)
    r2 = np.where(denominator > 0, 1 - residual / np.maximum(denominator, 1e-300), np.nan)
    flux = np.where(denominator[:, 0] > 0, r2[:, 0], 0.0)
    scalar = np.where(denominator[:, 1:3] > 0, r2[:, 1:3], 0.0).mean(axis=1)
    consistency = np.nanmean(np.maximum(0, r2[:, 3:]), axis=1)
    distances = np.asarray(data["distance"][arm], dtype=float)
    valid = weights @ np.isfinite(distances).astype(float)
    boundary = np.where(valid > 0, (weights @ np.nan_to_num(distances)) / np.maximum(valid, 1.0), 1.0)
    score = (
        0.55 * np.maximum(0, flux)
        + 0.15 * np.maximum(0, scalar)
        + 0.10 * (1 - np.minimum(1, boundary))
        + 0.20 * consistency
    )
    return np.column_stack((score, flux, scalar, boundary, consistency, r2[:, 3:]))


def _difference(actual: object, expected: object) -> float:
    difference = np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
    if not np.isfinite(difference).all() or np.any(difference > 1e-13):
        raise ValueError("Recomputed D72 gate differs")
    return float(difference.max())


def verify(data_path: Path, manifest: Path, gate: Path) -> dict[str, object]:
    """Reconstruct the frozen D72 pooled components and whole-shot intervals.

    Parameters
    ----------
    data_path : Path
        Gzip JSON of per-shot reference moments, residual sums and counters.
    manifest : Path
        Versioned ordered cohort, convergence support and source identities.
    gate : Path
        Original, hash-bound D72 gate report.

    Returns
    -------
    dict[str, object]
        Verification bounds, convergence totals and reconstructed score change.

    Raises
    ------
    ValueError
        Inputs violate the identity, support or score arithmetic contract.
    OSError
        An input cannot be read.
    """
    cohort = _cohort(manifest, data_path, gate)
    data = _load(data_path, cohort)
    stored = cast(Gate, json.loads(gate.read_text()))
    weights = np.random.default_rng(72020).multinomial(358, np.full(358, 1 / 358), size=20000).astype(float)
    bootstrap = {}
    exact = {}
    worst = 0.0
    failures = {}
    reconstructed: dict[str, Score] = {}
    for arm in STAGES:
        calculated = _scores(np.ones((1, 358)), data, arm)[0]
        recorded = {"base": stored["base"], "gs": stored["gs"]}[arm]
        worst = max(
            worst,
            _difference(
                calculated,
                [recorded["S"], recorded["psi"], recorded["qb"], recorded["D"], recorded["C"]]
                + recorded["cons_each"],
            ),
        )
        counters = np.asarray(data["counters"][arm], dtype=np.int64)
        fails = (int(counters[:, 15].sum()), int(counters[:, 7:14].sum()))
        if fails != (recorded["lcfs_fails"], recorded["cons_fails"]):
            raise ValueError("Extraction failure totals differ")
        reconstructed[arm] = {
            "S": float(calculated[0]),
            "psi": float(calculated[1]),
            "qb": float(calculated[2]),
            "D": float(calculated[3]),
            "C": float(calculated[4]),
            "cons_each": [float(value) for value in calculated[5:]],
            "lcfs_fails": fails[0],
            "cons_fails": fails[1],
        }
        failures[arm] = fails
        exact[arm] = float(calculated[0])
        bootstrap[arm] = _scores(weights, data, arm)[:, 0]
    delta = exact["gs"] - exact["base"]
    draws = bootstrap["gs"] - bootstrap["base"]
    ci99 = np.quantile(draws, [0.005, 0.995]).tolist()
    change = stored["gs_minus_base"]
    worst = max(
        worst,
        _difference(delta, change["delta_S"]),
        _difference(ci99, change["CI99"]),
        _difference(np.quantile(draws, [0.025, 0.975]), change["CI95"]),
        _difference(float((draws > 0).mean()), change["P_positive"]),
    )
    passed = bool(
        np.all(
            [
                delta >= 1e-5,
                ci99[0] > 0,
                failures["gs"][0] <= failures["base"][0],
                failures["gs"][1] <= failures["base"][1],
            ]
        )
    )
    worst = max(worst, _difference(float(passed), float(change["PASS"])))
    return {
        "status": "verified",
        "shots": 358,
        "frames": 78875,
        "maximum_absolute_difference": worst,
        "base": reconstructed["base"],
        "gs": reconstructed["gs"],
        "delta_S": delta,
        "CI99": ci99,
        "PASS": passed,
        "used_frames": sum(data["used"]),
        "nonconverged_frames": sum(data["used"]) - sum(data["converged"]),
        "shots_with_nonconvergence": sum(s["used"] != s["converged"] for s in cohort),
        "maps_recomputed": False,
        "physical_improvement_verified": False,
        "original_measurements_authenticated": False,
    }


def main() -> int:
    """Run the public verifier with a caller-safe refusal message.

    Returns
    -------
    int
        Zero for verified inputs, two for invalid or unavailable inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.data, args.manifest, args.gate)
    except (ValueError, OSError, TypeError, KeyError, OverflowError, EOFError, zlib.error):
        print(
            json.dumps({"status": "refused", "message": "D72 inputs are missing, invalid or inconsistent."})
        )
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
