# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — independent-method baseline contracts
"""Exercise actual alignment, fitting, Parquet lineage and public-command refusal."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]  # Arrow has no distributed typing stubs.
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # Arrow has no distributed stubs.
import pytest

from tools.official_ridge import SIGNALS, align, checksum, main, predict, read_shot, reproduce


def _row(n: int = 80) -> dict[str, object]:
    t = np.arange(n, dtype=float) * 10
    rr, zz = np.meshgrid(np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65))
    maps = -((rr**2 - 1.7**2) ** 2) / 8 - 0.9 * zz**2
    fields = maps + 0.1 * np.sin(t[:, None, None] / 80) * rr**2
    fields += 0.05 * np.cos(t[:, None, None] / 90) * zz**2
    row: dict[str, object] = {
        "efit_times": t.tolist(),
        "efit_psirz": fields.tolist(),
        "magnetics_time": t.tolist(),
        "magnetics_plasma_current_times": t.tolist(),
    }
    for i, signal in enumerate(SIGNALS):
        row["magnetics_" + signal] = (np.sin(t / (80 + i)) + 0.2 * t).tolist()
    return row


def _files(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    for stem in ("d3d_shot_0000000001", "d3d_shot_0000000002"):
        pq.write_table(pa.Table.from_pylist([_row()]), data / (stem + ".parquet"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_shots": ["d3d_shot_0000000001"],
                "excluded_shots": ["d3d_shot_0000000002"],
                "examples": {"p50": {"stem": "d3d_shot_0000000002", "time_ms": 40.0}},
                "training_manifest": [
                    {"stem": "d3d_shot_0000000001", "sha256": checksum(data / "d3d_shot_0000000001.parquet")}
                ],
                "evaluation_manifest": {"p50": {"sha256": checksum(data / "d3d_shot_0000000002.parquet")}},
            }
        )
    )
    return data, manifest


def test_alignment_and_ip_erratum() -> None:
    """The Ip fix re-origins only the clock that misses the breakdown window."""
    row = _row(8)
    x, y, times = align(row)
    assert x.shape == (8, 21) and y.shape == (8, 65, 65)
    assert x.dtype == y.dtype == np.float32 and times.dtype == np.float64
    assert np.array_equal(x[:, 0], np.asarray(row["magnetics_ECOILA"], dtype=np.float32))
    shifted = dict(row, magnetics_plasma_current_times=(times + 3000).tolist())
    assert np.array_equal(align(shifted)[0], x)
    flattened = dict(row, efit_psirz=y.reshape(8, -1).tolist())
    assert np.array_equal(align(flattened)[1], y)
    missing = dict(row)
    del missing["magnetics_F2A"]
    assert np.array_equal(align(missing)[0][:, SIGNALS.index("F2A")], np.zeros(8))
    query = dict(row, efit_times=(times + 5).tolist())
    expected = (x[:-1, 0] + x[1:, 0]) / 2
    assert np.allclose(align(query)[0][:-1, 0], expected, rtol=1e-6)


@pytest.mark.parametrize("fault", ["clock", "clock_order", "maps", "empty", "ip_nan", "ip_clock", "times"])
def test_invalid_acquisition(fault: str) -> None:
    """Invalid clock/schema support is refused before fitting, rather than repaired."""
    row = _row(8)
    if fault == "clock":
        row["magnetics_ECOILA"] = [1.0]
    elif fault == "clock_order":
        row["magnetics_time"] = [0.0] * 8
    elif fault == "maps":
        row["efit_psirz"] = np.zeros((8, 3, 3)).tolist()
    elif fault == "empty":
        row["efit_times"] = []
    elif fault == "ip_nan":
        row["magnetics_plasma_current"] = [float("nan")] * 8
    elif fault == "ip_clock":
        row["magnetics_plasma_current_times"] = [0.0]
    else:
        row["efit_times"] = [float("inf")] * 8
    with pytest.raises(ValueError):
        align(row)


def test_finite_filter_preserves_timestamp_identity() -> None:
    """Dropping one invalid target also drops that row's time and all input columns."""
    row = _row(8)
    maps = np.asarray(row["efit_psirz"], dtype=float)
    maps[3, 20, 20] = np.nan
    row["efit_psirz"] = maps.tolist()
    x, y, times = align(row)
    assert len(x) == len(y) == len(times) == 7
    assert times.tolist() == [0, 10, 20, 40, 50, 60, 70]


def test_ridge_recovers_known_linear_maps() -> None:
    """An independent affine-map oracle is recovered without using query targets."""
    rng = np.random.default_rng(2358)
    x = rng.normal(size=(80, 21)).astype(np.float32)
    queries = rng.normal(size=(4, 21)).astype(np.float32)
    rr, zz = np.meshgrid(np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65))

    def fields(
        inputs: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        return np.asarray(2 + inputs[:, 0, None, None] * rr + inputs[:, 1, None, None] * zz, dtype=np.float32)

    y = fields(x)
    fitted, receipt = predict(x, y, queries, 2)
    assert receipt["selected_alpha"] == 0.001 and receipt["frames"] == 80
    assert receipt["components"] == 2 and receipt["random_state"] == 42
    assert np.max(np.abs(fitted - fields(queries))) < 0.0003
    repeat, metadata = predict(x, y, queries, 2)
    assert np.array_equal(repeat, fitted) and metadata == receipt


@pytest.mark.parametrize("fault", ["x", "y", "query", "empty", "components", "nonfinite"])
def test_invalid_fit_arrays(fault: str) -> None:
    """The fit refuses dimension, finite-data and PCA-support errors."""
    x = np.ones((4, 21), dtype=np.float32)
    y = np.ones((4, 65, 65), dtype=np.float32)
    q = np.ones((2, 21), dtype=np.float32)
    components = 2
    if fault == "x":
        x = x[:, :20]
    elif fault == "y":
        y = y[:, :2, :2]
    elif fault == "query":
        q = q[:, :20]
    elif fault == "empty":
        q = q[:0]
    elif fault == "components":
        components = 5
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        predict(x, y, q, components)


def test_real_parquet_and_public_fit_cli(tmp_path: Path) -> None:
    """Fit a real excluded-shot Parquet pair through the installed command path."""
    data, manifest = _files(tmp_path)
    source = data / "d3d_shot_0000000001.parquet"
    shot = read_shot(source)
    assert shot.original_frames == len(shot.times) == 80
    assert shot.sha256 == hashlib.sha256(source.read_bytes()).hexdigest() == checksum(source)
    out = tmp_path / "fit"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "tools.official_ridge",
            "--data",
            str(data),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    assert result.stderr == ""
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((out / "receipt.json").read_text())
    assert receipt["training_shots"][0]["stem"] == "d3d_shot_0000000001"
    assert receipt["fit"]["frames"] == 80 and receipt["fit"]["components"] == 50
    assert receipt["payload_sha256"] == checksum(out / "baseline_frames.npz")
    with np.load(out / "baseline_frames.npz", allow_pickle=False) as stored:
        assert stored.files == ["p50_psi"] and stored["p50_psi"].shape == (65, 65)
    with pytest.raises(FileExistsError):
        reproduce(data, manifest, out)
    assert main(["--out", str(out)]) == 2


@pytest.mark.parametrize(
    "fault", ["overlap", "duplicate", "path", "example", "unexcluded", "empty", "count", "time", "filtered"]
)
def test_manifest_and_evaluation_refusal(tmp_path: Path, fault: str) -> None:
    """Reject leakage, missing examples, unsafe identities and unsupported query times."""
    data, manifest = _files(tmp_path)
    plan = json.loads(manifest.read_text())
    if fault == "overlap":
        plan["excluded_shots"].append(plan["training_shots"][0])
    elif fault == "duplicate":
        plan["training_shots"] *= 2
    elif fault == "path":
        plan["training_shots"][0] = "../d3d_shot_0000000001"
    elif fault == "example":
        plan["examples"]["../escape"] = plan["examples"].pop("p50")
    elif fault == "unexcluded":
        plan["excluded_shots"] = []
    elif fault == "empty":
        plan["examples"] = {}
    elif fault == "count":
        plan["training_shots"] *= 1025
    elif fault == "time":
        plan["examples"]["p50"]["time_ms"] = 35.0
    else:
        row = _row()
        fields = np.asarray(row["efit_psirz"], dtype=float)
        fields[0] = np.nan
        row["efit_psirz"] = fields.tolist()
        pq.write_table(pa.Table.from_pylist([row]), data / "d3d_shot_0000000002.parquet")
        plan["evaluation_manifest"]["p50"]["sha256"] = checksum(data / "d3d_shot_0000000002.parquet")
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        reproduce(data, manifest, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_parquet_shot_count_and_missing_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A multirow file and unreadable CLI input cannot produce a fit receipt."""
    path = tmp_path / "two.parquet"
    pq.write_table(pa.Table.from_pylist([_row(3), _row(3)]), path)
    with pytest.raises(ValueError, match="exactly one"):
        read_shot(path)
    assert (
        main(
            ["--data", str(tmp_path), "--manifest", str(tmp_path / "absent"), "--out", str(tmp_path / "out")]
        )
        == 2
    )
    reply = json.loads(capsys.readouterr().out)
    assert reply["status"] == "refused" and str(tmp_path) not in reply["message"]


def test_actual_numerical_range_refusal() -> None:
    """Finite but extreme queries cannot silently generate infinite prediction maps."""
    rng = np.random.default_rng(456)
    x = rng.normal(size=(60, 21)).astype(np.float32)
    y = np.broadcast_to((1000 * x[:, 0])[:, None, None], (60, 65, 65)).copy()
    queries = np.full((1, 21), 3e38, dtype=np.float32)
    with pytest.raises(FloatingPointError):
        predict(x, y, queries, 2)
    with pytest.raises(FloatingPointError):
        predict(x, np.ones_like(y), x[:1], 2)


@pytest.mark.parametrize("kind", ["training", "evaluation", "missing_evaluation"])
def test_observed_raw_hashes_must_match_manifest(tmp_path: Path, kind: str) -> None:
    """A familiar filename does not admit altered raw training or evaluation data."""
    data, manifest = _files(tmp_path)
    plan = json.loads(manifest.read_text())
    if kind == "training":
        plan["training_manifest"][0]["sha256"] = "0" * 64
    elif kind == "evaluation":
        plan["evaluation_manifest"]["p50"]["sha256"] = "0" * 64
    else:
        plan["evaluation_manifest"] = {}
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="identity differs"):
        reproduce(data, manifest, tmp_path / "unwritten")
    assert not (tmp_path / "unwritten").exists()
