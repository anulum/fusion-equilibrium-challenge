# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — official-method audit baseline reproduction
"""Reproduce the starter's coil/current PCA–Ridge method on declared shot sets.

Adapted from Sophelio's MIT starter at 823f6eef1f527e1b5566a8fc1e952da6e2fadf43.
See NOTICE.md for the retained permission notice and the sampling differences.
This diagnostic baseline does not modify or load the submitted model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # Arrow has no distributed typing stubs.
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore[import-untyped]  # SciPy does not ship stubs.
from sklearn.decomposition import PCA  # type: ignore[import-untyped]  # sklearn has no distributed stubs.
from sklearn.linear_model import RidgeCV  # type: ignore[import-untyped]  # sklearn has no distributed stubs.
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]  # sklearn has no stubs.

from tools.equilibrium_audit import Parser

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
SIGNALS = ("ECOILA", "bcoil", *(f"F{i}{side}" for i in range(1, 10) for side in "AB"), "plasma_current")
REFUSAL = "Baseline fit refused: check the shot manifest, data and output directory."


@dataclass(frozen=True)
class Shot:
    """Finite aligned rows, preserving their exact released timestamps."""

    features: F32
    maps: F32
    times: F64
    original_frames: int
    sha256: str


def checksum(path: Path) -> str:
    """Return a streaming SHA-256 without retaining a shot's file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def align(row: Mapping[str, object]) -> tuple[F32, F32, F64]:
    """Align the starter's 21 input signals and retain its finite-row filter.

    Parameters
    ----------
    row : Mapping of str to object
        One released DIII-D Parquet row. Times are milliseconds and flux is Wb/rad.

    Returns
    -------
    features, maps, times : ndarray
        Finite float32 inputs/maps and their corresponding float64 EFIT times.

    Raises
    ------
    ValueError
        The released row is empty, misaligned or has an unsupported map shape.
    """
    times = np.asarray(row["efit_times"], dtype=np.float64)
    maps = np.stack([np.asarray(x, dtype=np.float32) for x in cast(Sequence[object], row["efit_psirz"])])
    if maps.ndim == 2 and maps.shape[1] == 65 * 65:
        maps = maps.reshape(-1, 65, 65)
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    ip = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or maps.shape != (len(times), 65, 65)
        or ip.ndim != 1
        or len(ip) < 2
        or ip_times.shape != ip.shape
        or shared.ndim != 1
        or len(shared) < 2
        or not np.isfinite(ip_times).all()
        or not np.isfinite(shared).all()
        or not np.isfinite(ip).any()
    ):
        raise ValueError("Unsupported released DIII-D row")
    threshold = 0.10 * np.nanpercentile(ip, 99.5)
    breakdown = float(ip_times[int(np.argmax(ip > threshold))])
    if not -500 <= breakdown <= 1000:
        ip_times = float(shared[0]) + (ip_times - ip_times[0])
    columns = []
    for signal in SIGNALS:
        key = "magnetics_" + signal
        if key not in row:
            columns.append(np.zeros(len(times), dtype=np.float32))
            continue
        clock = ip_times if signal == "plasma_current" else shared
        values = np.asarray(row[key], dtype=np.float32)
        if clock.shape != values.shape or np.any(np.diff(clock) <= 0):
            raise ValueError("Unsupported magnetic acquisition clock")
        interpolate = interp1d(clock, values, kind="linear", fill_value="extrapolate", bounds_error=False)
        columns.append(np.asarray(interpolate(times), dtype=np.float32))
    features = np.column_stack(columns).astype(np.float32)
    valid = np.isfinite(features).all(axis=1) & np.isfinite(maps.reshape(len(maps), -1)).all(axis=1)
    return features[valid], maps[valid], times[valid]


def read_shot(path: Path) -> Shot:
    """Read one trusted released Parquet shot and record finite-row provenance.

    Parameters
    ----------
    path : Path
        Original one-row DIII-D training file; no dataset download is performed.

    Returns
    -------
    Shot
        Aligned data, original frame count and exact file checksum.

    Raises
    ------
    ValueError
        The file does not contain exactly one shot.
    """
    required = {"efit_times", "efit_psirz", "magnetics_time", "magnetics_plasma_current_times"}
    columns = sorted(required | {"magnetics_" + signal for signal in SIGNALS})
    available = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[key for key in columns if key in available], use_threads=False)
    rows = cast(list[dict[str, object]], table.to_pylist())
    if len(rows) != 1:
        raise ValueError("Expected exactly one released shot")
    x, y, times = align(rows[0])
    return Shot(x, y, times, len(cast(Sequence[object], rows[0]["efit_times"])), checksum(path))


def predict(x: F32, y: F32, queries: F32, components: int = 50) -> tuple[F32, dict[str, object]]:
    """Fit the declared starter estimator and predict independent input rows.

    Parameters
    ----------
    x, y : ndarray
        Finite training inputs (n, 21) and maps (n, 65, 65).
    queries : ndarray
        Finite independent input rows (m, 21), with no target used by the fit.
    components : int
        PCA component count; the published baseline fixes 50. Smaller values
        support bounded analytic tests, not outcome-based model selection.

    Returns
    -------
    maps, fit : ndarray and dict
        Float32 maps and the actual regularisation, solver and variance metadata.

    Raises
    ------
    ValueError
        Dimensions, finite values or the component count are invalid.
    FloatingPointError
        PCA variance or inference exceeds the representable numerical range.
    """
    if (
        x.ndim != 2
        or x.shape != (len(y), 21)
        or y.shape != (len(y), 65, 65)
        or queries.ndim != 2
        or queries.shape[1] != 21
        or not len(queries)
        or not 1 <= components <= min(len(y), 65 * 65)
        or not all(np.isfinite(a).all() for a in (x, y, queries))
    ):
        raise ValueError("Invalid independent baseline arrays")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        pca = PCA(n_components=components, random_state=42)
        flat = y.reshape(len(y), -1).astype(np.float64)
        pca.fit(flat)
        targets = pca.transform(flat).astype(np.float32)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        ridge.fit(scaled, targets)
        coefficients = np.asarray(ridge.predict(scaler.transform(queries)), dtype=np.float32)
        maps = np.asarray(pca.inverse_transform(coefficients.astype(np.float64)), dtype=np.float32)
    return maps.reshape(-1, 65, 65), {
        "selected_alpha": float(ridge.alpha_),
        "components": components,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "random_state": 42,
        "frames": len(y),
    }


def reproduce(data: Path, manifest: Path, out: Path) -> dict[str, object]:
    """Reproduce a fixed shot-level fit without serialising a competitive model.

    Parameters
    ----------
    data : Path
        Directory of the original DIII-D training Parquet files.
    manifest : Path
        JSON with training_shots, excluded_shots and examples. Each example
        declares stem and time_ms. training_manifest and evaluation_manifest
        bind the exact original file hashes. No train/evaluation overlap is permitted.
    out : Path
        New output directory for four map illustrations and a fit receipt.

    Returns
    -------
    dict of str to object
        Training provenance, fit metadata, frame identity and payload checksum.

    Raises
    ------
    ValueError
        Shot identifiers, exclusions, requested times or their support differ.
    OSError
        Input files are unavailable or the output directory already exists.
    """
    plan = cast(dict[str, object], json.loads(manifest.read_text()))
    training = cast(list[str], plan["training_shots"])
    excluded = cast(list[str], plan["excluded_shots"])
    examples = cast(dict[str, dict[str, object]], plan["examples"])
    evaluation = [str(item["stem"]) for item in examples.values()]
    ids = [*training, *excluded, *evaluation]
    if (
        not training
        or len(training) > 1024
        or len(set(training)) != len(training)
        or not examples
        or any(re.fullmatch(r"d3d_shot_[0-9a-f]{10}", item) is None for item in ids)
        or set(training) & set(excluded)
        or not set(evaluation) <= set(excluded)
        or any(re.fullmatch(r"[a-z][a-z0-9_]{0,20}", key) is None for key in examples)
    ):
        raise ValueError("Invalid disjoint shot manifest")
    if out.exists():
        raise FileExistsError("Baseline output must be new")
    shots = [read_shot(data / (stem + ".parquet")) for stem in training]
    expected = cast(list[dict[str, object]], plan["training_manifest"])
    if [(item["stem"], item["sha256"]) for item in expected] != [
        (stem, shot.sha256) for stem, shot in zip(training, shots, strict=True)
    ]:
        raise ValueError("Original training file identity differs")
    x, y = np.concatenate([shot.features for shot in shots]), np.concatenate([shot.maps for shot in shots])
    provenance = [
        {
            "stem": stem,
            "sha256": shot.sha256,
            "input_frames": shot.original_frames,
            "finite_frames": len(shot.times),
        }
        for stem, shot in zip(training, shots, strict=True)
    ]
    del shots
    cases = {key: read_shot(data / (str(item["stem"]) + ".parquet")) for key, item in examples.items()}
    expected_cases = cast(dict[str, dict[str, object]], plan["evaluation_manifest"])
    if set(expected_cases) != set(cases) or any(
        shot.sha256 != expected_cases[key]["sha256"] for key, shot in cases.items()
    ):
        raise ValueError("Original evaluation file identity differs")
    queries = []
    for key, item in examples.items():
        shot = cases[key]
        indices = np.flatnonzero(shot.times == float(cast(float, item["time_ms"])))
        if len(indices) != 1 or len(shot.times) != shot.original_frames:
            raise ValueError("Evaluation time must have complete finite support")
        queries.append(shot.features[int(indices[0])])
    maps, fit = predict(x, y, np.stack(queries))
    out.mkdir()
    payload = out / "baseline_frames.npz"
    np.savez_compressed(payload, **{key + "_psi": maps[i] for i, key in enumerate(examples)})
    receipt = {
        "schema": "official-ridge-audit.v1",
        "manifest_sha256": checksum(manifest),
        "source_sha256": checksum(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pyarrow")},
        },
        "evaluation_shots": {
            key: {"sha256": shot.sha256, "input_frames": shot.original_frames} for key, shot in cases.items()
        },
        "training_shots": provenance,
        "examples": examples,
        "fit": fit,
        "payload_sha256": checksum(payload),
        "external_independent_replication": False,
        "competition_score_computed": False,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run an explicit offline baseline fit, with a fixed caller-safe refusal."""
    parser = Parser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        report = reproduce(args.data, args.manifest, args.out)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    except Exception:
        print(json.dumps({"status": "refused", "message": REFUSAL}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
