# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — current-tail scalar error measurement
"""Measure the entire declared development cohort without changing predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def digest(path: Path) -> str:
    """Return the SHA-256 of a source or receipt file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure() -> None:
    """Write all-frame error strata and provenance for the frozen protocol."""
    directory = Path(__file__).resolve().parent
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = Path(protocol["input"])
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    shot = arrays["shot"].astype(np.int64)
    stems = arrays["stems"].astype(str)
    current = arrays["ip"].astype(np.float64)
    times = arrays["time"].astype(np.float64)
    assert len(stems) == protocol["expected_shots"]
    assert len(current) == protocol["expected_frames"]
    assert np.isfinite(current).all() and np.isfinite(times).all()
    assert np.array_equal(np.unique(shot), np.arange(len(stems)))
    position = np.empty(len(shot), dtype=np.float64)
    for index in range(len(stems)):
        pick = np.flatnonzero(shot == index)
        assert np.all(np.diff(pick) == 1)
        assert np.all(np.diff(times[pick]) > 0)
        position[pick] = np.arange(len(pick)) / max(1, len(pick) - 1)
    report: dict[str, object] = {
        "status": "measurement_complete_no_fit_or_admission",
        "shots": len(stems), "frames": len(current),
        "source_sha256": digest(source),
        "protocol_sha256": digest(protocol_path),
        "measurement_source_sha256": digest(Path(__file__)),
        "cohort_exposure": protocol["cohort"],
        "all_frames_retained": True,
    }
    edges = protocol["current_bins_A"]
    for name in ("q95", "betaN"):
        truth = arrays[f"{name}_truth"].astype(np.float64)
        prediction = (0.75 * arrays[f"{name}_map"].astype(np.float64)
                      + 0.25 * arrays[f"{name}_joint"].astype(np.float64))
        assert prediction.shape == truth.shape == current.shape
        assert np.isfinite(prediction).all()
        defined = np.isfinite(truth)
        error = prediction - truth
        square = error * error
        total = float(square[defined].sum())
        assert total > 0

        def stratum(mask: NDArray[np.bool_]) -> dict[str, object]:
            use = mask & defined
            count = int(use.sum())
            return {
                "frames": count,
                "frame_fraction": float(count / defined.sum()),
                "sse_fraction": float(square[use].sum() / total),
                "bias": float(error[use].mean()) if count else None,
                "rmse": float(np.sqrt(square[use].mean())) if count else None,
            }

        bins = [{"abs_current_A": [lo, hi],
                 **stratum((np.abs(current) >= lo) & (np.abs(current) < hi))}
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
        assert sum(row["frames"] for row in bins) == int(defined.sum())
        per_shot = np.bincount(shot[defined], weights=square[defined], minlength=len(stems))
        order = np.argsort(np.where(defined, square, -np.inf))[::-1][:20]
        block: dict[str, object] = {
            "defined_frames": int(defined.sum()), "sse": total,
            "r2_pooled": float(1 - total / np.sum((truth[defined] - truth[defined].mean()) ** 2)),
            "by_abs_current": bins,
            "historical_floor_clipped": stratum(np.abs(current) < 50000),
            "negative_current": stratum(current < 0),
            "last5percent_record": stratum(position >= 0.95),
            "last5percent_and_below_floor": stratum((position >= 0.95) & (np.abs(current) < 50000)),
            "largest_shot_error_fraction": float(per_shot.max() / total),
            "largest_shot_error_stem": str(stems[np.argmax(per_shot)]),
            "truth_quantiles_0_1_50_99_100_percent": np.quantile(truth[defined], [0, .01, .5, .99, 1]).tolist(),
            "truth_times_signed_Ip_MA_quantiles": np.quantile((truth * current / 1e6)[defined], [0, .01, .5, .99, 1]).tolist(),
            "top20_frames": [{
                "stem": str(stems[shot[i]]), "time_ms": float(times[i]),
                "record_fraction": float(position[i]), "current_A": float(current[i]),
                "truth": float(truth[i]), "prediction": float(prediction[i]),
                "error": float(error[i]), "sse_fraction": float(square[i] / total),
            } for i in order],
        }
        report[name] = block
    with (directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: {key: value for key, value in report[name].items()
                            if key != "top20_frames"} for name in ("q95", "betaN")}, indent=2))


if __name__ == "__main__":
    measure()
