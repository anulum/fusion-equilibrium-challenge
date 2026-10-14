# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the prediction archive the scorer reads

"""Collect the predictions of a set of shots and write them as the scorer reads them.

The scorer takes one archive per machine, keyed by the *position* of each shot
in the test stream: ``shot_0000_psirz``, ``shot_0000_q95``, ``shot_0000_betaN``
and so on. Nothing in the file names the shot, so a prediction stored under the
wrong number is scored against another discharge and nothing fails. The
collector therefore never takes a number: a shot's position is the order in
which it was added, and the caller's only duty is to add the shots in stream
order.

Flux maps are stored as single precision. They were stored as half precision
at first, as the organisers' validator recommends for the file's size, and that
cost the shape terms: the boundary the scorer extracts passes through an
X-point, where the flux is flat, and half precision (one part in 2000) moves it
by centimetres. On 32 validation shots of the DIII-D accuracy recipe, the seven
derived scalars read off the same maps stored in half and in single precision:
κ 0.9618 → 0.9723, δ_top 0.9823 → 0.9859, li 0.9748 → 0.9772, Z_axis 0.9895 →
0.9916, R_axis 0.9874 → 0.9888, δ_bot 0.9714 → 0.9725, volume unchanged. The
file doubles in size (2.7 GB for the DIII-D test set), which the validator
accepts: it asks for a floating dtype only. The two scalars are single
precision as before.

Every stored value is finite. A frame the physics could not resolve carries
``nan`` up to this point, and a file with ``nan`` in it is not a submission;
such values are written as zero and counted, so that the count can be reported
next to the score it belongs to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

DIII_D_SPLIT = "diii_d_public_test"
MAST_SPLIT = "mast_public_test"
GRID = 65
"""Cells along either side of the reconstruction grid, on both machines."""

FLUX_DTYPE = np.float32
SCALAR_DTYPE = np.float32
FLUX_DIMENSIONS = 3
"""Frames, rows and columns."""


class PredictionArchive:
    """Predictions of consecutive shots of one machine, in stream order."""

    def __init__(self) -> None:
        """Start with no shot."""
        self._entries: dict[str, NDArray[np.generic]] = {}
        self._shots = 0
        self._frames = 0
        self._replaced = 0

    @property
    def shots(self) -> int:
        """Number of shots added so far."""
        return self._shots

    @property
    def frames(self) -> int:
        """Number of frames added so far, over all shots."""
        return self._frames

    @property
    def replaced(self) -> int:
        """Number of non-finite values that were written as zero, over all shots."""
        return self._replaced

    def add(
        self,
        psi: NDArray[np.floating],
        q95: NDArray[np.floating],
        beta_n: NDArray[np.floating],
    ) -> None:
        """Add the next shot of the stream.

        Parameters
        ----------
        psi:
            Flux maps, shape ``(T, NZ, NR)``.
        q95, beta_n:
            One value per frame, shape ``(T,)``.

        Raises
        ------
        ValueError
            If the three disagree on the number of frames, or if a value is too
            large for its stored precision — that would store an infinity.
        """
        per_frame = (psi.shape[0],)
        if psi.ndim != FLUX_DIMENSIONS or q95.shape != per_frame or beta_n.shape != per_frame:
            raise ValueError(
                f"shot {self._shots}: flux {psi.shape}, q95 {q95.shape} and betaN "
                f"{beta_n.shape} do not describe the same frames"
            )
        stored = []
        for values, dtype in ((psi, FLUX_DTYPE), (q95, SCALAR_DTYPE), (beta_n, SCALAR_DTYPE)):
            missing = ~np.isfinite(values)
            self._replaced += int(missing.sum())
            with np.errstate(over="ignore"):  # an overflow is caught by the check below
                cast = np.where(missing, 0.0, values).astype(dtype)
            if not bool(np.all(np.isfinite(cast))):
                raise ValueError(
                    f"shot {self._shots}: a value of magnitude "
                    f"{float(np.max(np.abs(np.where(missing, 0.0, values)))):.3g} "
                    f"does not fit {np.dtype(dtype).name}"
                )
            stored.append(cast)
        for name, cast in zip(("psirz", "q95", "betaN"), stored, strict=True):
            self._entries[f"shot_{self._shots:04d}_{name}"] = cast
        self._shots += 1
        self._frames += int(psi.shape[0])

    def write(self, path: Path) -> None:
        """Write the archive, compressed and readable without pickling.

        The scorer loads with ``allow_pickle=False``; saying so here as well pins
        the one keyword an entry name must never be allowed to collide with.
        """
        if not self._shots:
            raise ValueError("an archive without a shot is not written")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, allow_pickle=False, **self._entries)


def split_files(data_root: Path, split: str, limit: int | None) -> list[Path]:
    """Return a split's shot files in stream order, optionally only the first ``limit``."""
    files = sorted((data_root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shot files under {data_root / split}")
    return files if limit is None else files[:limit]


def zero_archive(files: list[Path]) -> PredictionArchive:
    """Return zeros of the right shape for every shot, reading only each shot's time base."""
    archive = PredictionArchive()
    for path in files:
        times = pq.read_table(path, columns=["efit_times"]).column("efit_times")[0].as_py()
        frames = len(times)
        archive.add(np.zeros((frames, GRID, GRID)), np.zeros(frames), np.zeros(frames))
    return archive
