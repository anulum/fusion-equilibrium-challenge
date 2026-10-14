# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — how much of each predicted flux coordinate to believe

"""Scale each predicted flux coordinate by how much of it is signal.

The fit's loss is the squared map error, so a coordinate weighs what its
variance weighs, and the trailing ones weigh almost nothing: the network's
answer there is mostly scatter. On the validation part a mean of three networks
had, in coordinates 35 to 63, a squared error of 41.8 against a variance of
15.0 — leaving them at zero was better than predicting them — and the boundary
shape, which the flux term hardly sees, suffered for it (elongation R² 0.930
against 0.941 with those coordinates left out, composite 0.98961 against
0.99076).

For a coordinate ``c`` predicted as ``p``, the scale ``g`` and shift ``b``
with the least squared error of ``g·p + b`` are the regression of ``c`` on
``p``; ``g`` is held between zero and one, since a coordinate should be
believed less, never more or with its sign turned. Where the prediction is
good ``g`` is one to three places; where it is scatter ``g`` falls to zero and
the coordinate goes to its mean. The gains belong to one set of networks
answering together — a mean of many networks scatters less than one network
and earns larger gains — and are fitted on the calibration part, which no
fit's gradient has seen, with the shots reserved for evaluation left out.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fec_adapter.diii_d_accuracy.network import Network, predict_together


@dataclass(frozen=True)
class Gains:
    """A scale and a shift for every flux coordinate, shape ``(K,)`` each."""

    scale: NDArray[np.float64]
    shift: NDArray[np.float64]

    @classmethod
    def fitted(cls, predicted: NDArray[np.floating], true: NDArray[np.floating]) -> Gains:
        """Return the least-squares scale within [0, 1] and the shift that goes with it."""
        if predicted.shape != true.shape or predicted.ndim != 2:  # noqa: PLR2004 - rows, columns
            raise ValueError(f"shapes {predicted.shape} and {true.shape} do not match")
        wanted, given = true.astype(np.float64), predicted.astype(np.float64)
        spread = ((given - given.mean(axis=0)) ** 2).sum(axis=0)
        together = ((given - given.mean(axis=0)) * (wanted - wanted.mean(axis=0))).sum(axis=0)
        scale = np.clip(np.divide(together, spread, out=np.zeros_like(spread), where=spread > 0.0),
                        0.0, 1.0)
        return cls(scale, wanted.mean(axis=0) - scale * given.mean(axis=0))

    def applied(self, coordinates: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the coordinates scaled and shifted."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.scale.size:  # noqa: PLR2004
            raise ValueError(
                f"coordinates of shape {coordinates.shape} do not have {self.scale.size} columns"
            )
        scaled: NDArray[np.float64] = coordinates.astype(np.float64) * self.scale + self.shift
        return scaled

    def save(self, path: Path) -> None:
        """Store the two arrays without pickling."""
        np.savez_compressed(path, allow_pickle=False, scale=self.scale, shift=self.shift)

    @classmethod
    def load(cls, path: Path) -> Gains:
        """Read stored gains."""
        with np.load(path, allow_pickle=False) as stored:
            return cls(stored["scale"].astype(np.float64), stored["shift"].astype(np.float64))


def main() -> int:
    """Fit the gains of networks answering together and write them with a short report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True, nargs="+")
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the gains file to write (.npz)")
    args = parser.parse_args()
    # Here and not at the top: the build reads this module and has no use for the fitting library.
    from fec_adapter.diii_d_accuracy.fit_network import Table  # noqa: PLC0415

    networks = [Network.load(path) for path in args.network]
    reserved = frozenset(json.loads(args.reserved.read_text())["stems"])
    table = Table.load(args.tables / "table_calibration.npz", networks[0].coordinates, reserved)
    predicted, _ = predict_together(networks, {table.feature_layout: table.features})
    gains = Gains.fitted(predicted, table.coordinates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gains.save(args.out)
    true = table.coordinates.astype(np.float64)
    report = {
        "networks": [str(path) for path in args.network], "frames": int(true.shape[0]),
        "squared_error_before": float(((predicted - true) ** 2).sum()),
        "squared_error_after": float(((gains.applied(predicted) - true) ** 2).sum()),
        "scale": [round(float(value), 4) for value in gains.scale],
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
