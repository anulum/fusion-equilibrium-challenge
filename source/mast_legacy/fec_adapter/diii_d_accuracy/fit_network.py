# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fit the accuracy entry's network on the frame tables

"""Fit the network on the training table and quote it on the validation table.

The three parts of the split have three jobs here and do not trade them:

training
    The weights are fitted on it.
calibration
    Decides *which* weights are kept: after every pass the pooled flux R² on
    the calibration table is computed, and the best pass is the one exported.
    That is a choice made with labels, so it may not be made on the part the
    result is quoted on.
validation
    Read once, after the choice, for the numbers in the report.

The loss is written in the units of the score. Every flux coordinate is divided
by one common number — the root of the summed coordinate variances — so that
the squared error over the coordinates is the pooled squared error over the
map, as a fraction of the maps' variance: the first term *is* ``1 - R²`` of the
flux up to what the basis leaves out. Each label is divided by its own standard
deviation, so its term is that label's ``1 - R²``; labels missing on a frame
are left out of that frame's sum.

``fine_emphasis`` tilts that first term. At zero every coordinate counts by its
squared error, which is the flux R² and nothing else — and then the leading
directions, which carry almost all the variance, are all the optimiser attends
to. The quantities the Consistency term derives from a map (where the axis
sits, the internal inductance) live in directions of small variance. With
emphasis ``g`` a coordinate of variance ``v`` is weighted by ``v**-g``, rescaled
so that a model that is equally good, relatively, in every direction has the
same loss as before: ``g = 1`` would weigh every direction alike, ``g = 0.5``
is half way. Whatever the emphasis, the score that chooses the weights and the
one that is quoted default to the plain flux R². A calibration-bound
``selection_score`` can instead select on the complete finished prediction score.
The emphasis was tried at 0.5 on the full
tables and lost everywhere on the validation part (flux R² 0.99870 against
0.99978, q95 0.929 against 0.995, li 0.770 against 0.930): the tilt shrinks the
labels' share of the loss and spends the layers on directions that are mostly
scatter. It stays an option and is not used.

``dropout`` switches hidden units off at random while fitting. The fit reaches
its training shots far more closely than unseen ones (li: R² 0.988 against
0.930), which is what it is for. It acts only during fitting; the stored arrays
are the network as it evaluates, and a test holds the two together.

``edge_emphasis`` counts map errors in the band the plasma boundary runs through that many
times more (:mod:`.edge_band`); only the accelerator fit implements it, and this one refuses it.

``gradient_limit`` caps the length of a step's whole gradient. Without it one fit in
eight on the accelerators lost a pass to a jump of the loss (by half or more between
two passes) and never came back to its siblings: seed 5 ended at flux R² 0.99849
against 0.99982, and both the 120-pass and the batch-2048 variants ended near 0.9995.
Zero leaves the gradient as it is.

The flux R² of a model is computed from the table alone: with orthonormal
directions its squared error is the error in the coordinates plus what the
basis leaves out, and the table carries both sums shot by shot, so the R² is
exact for whichever shots a fit keeps.

The fitted network leaves as plain arrays (:mod:`.network`); nothing here is
needed to predict with it.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from fec_adapter.diii_d_accuracy.layout import RECORD_LAYOUT, FeatureLayout, layout_of
from fec_adapter.diii_d_accuracy.network import INPUT_CLIP, Network

SCALE_FLOOR = 1.0e-6
"""Below this spread a feature is a constant and is divided by one instead."""


@dataclass(frozen=True)
class FitSettings:
    """Everything that decides a fit besides the data."""

    coordinates: int = 64
    width: int = 512
    depth: int = 4
    epochs: int = 60
    batch: int = 4096
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    label_weight: float = 0.05
    label_weights: tuple[tuple[str, float], ...] = ()
    """Per-label replacements of :attr:`label_weight`, as ``(label name, weight)`` pairs; a
    label not named keeps :attr:`label_weight`. A name the table does not carry is an error."""
    fine_emphasis: float = 0.0
    dropout: float = 0.0
    gradient_limit: float = 0.0
    edge_emphasis: float = 0.0
    shot_balance: float = 0.0
    """Strength, 0 to 1, of weighting each row's label error by its shot's share of frames.

    The scorer averages the boundary distance over shots, so a two-frame record weighs as
    much as a three-hundred-frame one, while the loss averages rows. At strength ``s`` a
    row's label error is weighted ``(1 - s) + s · n̄ / n_shot`` (``n̄`` the mean number of
    frames of a shot), which averages to one over the rows; the flux coordinates are not
    reweighted. Implemented by the accelerator fit only."""
    seed: int = 0
    threads: int = 12


@dataclass(frozen=True)
class Table:
    """One part's table, cut to the coordinates a fit uses."""

    features: NDArray[np.float32]
    coordinates: NDArray[np.float32]
    labels: NDArray[np.float32]
    label_names: tuple[str, ...]
    flux_sums: dict[str, float]
    """The two sums that turn a coordinate error into a pooled flux R², for the kept shots."""
    feature_layout: FeatureLayout = RECORD_LAYOUT
    """What the feature columns are measured against; a table that names none is record-relative."""
    shot_of_row: NDArray[np.int64] | None = None
    """The shot each kept row belongs to, as an index into the table's shots."""

    @classmethod
    def load(
        cls, path: Path, coordinates: int, without: frozenset[str] = frozenset()
    ) -> Table:
        """Read a table, keep its leading ``coordinates``, and leave out the shots named.

        ``without`` is how shots reserved for evaluation stay unseen: their rows
        never reach the optimiser, whichever part of the split they fall in.
        """
        with np.load(path, allow_pickle=False) as stored:
            if stored["coordinates"].shape[1] < coordinates:
                raise ValueError(
                    f"{path.name} holds {stored['coordinates'].shape[1]} coordinates, "
                    f"{coordinates} were asked for"
                )
            kept_shots = np.asarray([str(stem) not in without for stem in stored["stems"]])
            keep = kept_shots[stored["shot_of_row"]]
            cells = float(stored["shot_cells"][kept_shots].sum())
            mean = float(stored["shot_map_sum"][kept_shots].sum()) / cells
            dropped = stored["coordinates"][keep, coordinates:].astype(np.float64)
            sums = {
                "flux_squares_about_the_mean":
                    float(stored["shot_squares_total"][kept_shots].sum()) - cells * mean**2,
                # The stored coordinates a fit does not use are left out of its maps as
                # well, so their squares count with the remainder of the basis.
                "flux_squares_left_out_by_the_basis":
                    float(stored["shot_squares_left_out"][kept_shots].sum())
                    + float(np.sum(dropped**2)),
            }
            return cls(
                stored["features"][keep], stored["coordinates"][keep, :coordinates],
                stored["labels"][keep], tuple(str(name) for name in stored["label_names"]),
                sums,
                layout_of(str(stored["feature_layout"])) if "feature_layout" in stored.files
                else RECORD_LAYOUT,
                np.asarray(stored["shot_of_row"][keep], dtype=np.int64),
            )


class _Model(nn.Module):
    """The linear path plus the multilayer path, on standardised and clipped inputs."""

    def __init__(self, inputs: int, outputs: int, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        sizes = [inputs, *([width] * depth)]
        layers: list[nn.Module] = []
        for before, after in itertools.pairwise(sizes):
            layers += [nn.Linear(before, after), nn.SiLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers, nn.Linear(sizes[-1], outputs))
        self.linear = nn.Linear(inputs, outputs, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the raw, still scaled, outputs."""
        result: torch.Tensor = self.layers(inputs) + self.linear(inputs)
        return result


def label_loss_weights(
    label_names: tuple[str, ...], settings: FitSettings
) -> NDArray[np.float32]:
    """Return each label's weight in the loss, in table order.

    Parameters
    ----------
    label_names:
        The table's label names, in column order.
    settings:
        The fit settings; :attr:`FitSettings.label_weights` replaces
        :attr:`FitSettings.label_weight`
        for the labels it names.

    Raises
    ------
    ValueError
        If a replacement names a label the table does not carry, names one twice, or gives a
        weight that is not finite and non-negative.
    """
    replacements: dict[str, float] = {}
    for name, weight in settings.label_weights:
        if name in replacements:
            raise ValueError(f"label weight for {name!r} given twice")
        if name not in label_names:
            raise ValueError(f"label {name!r} is not among the table's labels {label_names}")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"label weight for {name!r} must be finite and non-negative, not {weight}"
            )
        replacements[name] = float(weight)
    return np.asarray([replacements.get(name, settings.label_weight) for name in label_names],
                      dtype=np.float32)


def flux_r2(squared_coordinate_error: float, part_report: dict[str, float]) -> float:
    """Return the pooled flux R² from the coordinate error and the table report's two sums."""
    left_out = part_report["flux_squares_left_out_by_the_basis"]
    return 1.0 - (squared_coordinate_error + left_out) / part_report["flux_squares_about_the_mean"]


def label_r2(truth: NDArray[np.float32], predicted: NDArray[np.float64]) -> float:
    """Return the pooled R² of one label over the frames that carry it."""
    present = np.isfinite(truth)
    if not present.any():
        return float("nan")
    residual = np.sum((truth[present] - predicted[present]) ** 2)
    return float(1.0 - residual / np.sum((truth[present] - truth[present].mean()) ** 2))


@dataclass(frozen=True)
class _Scaling:
    """How inputs and outputs are brought to the size the optimiser works at."""

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    output_mean: NDArray[np.float64]
    output_scale: NDArray[np.float64]

    @classmethod
    def of(cls, training: Table) -> _Scaling:
        """Take every mean and spread from the training table."""
        spread = training.features.std(axis=0, dtype=np.float64)
        common = float(np.sqrt(training.coordinates.var(axis=0, dtype=np.float64).sum()))
        labels = training.labels.astype(np.float64)
        label_spread = np.nanstd(labels, axis=0)
        return cls(
            feature_mean=training.features.mean(axis=0, dtype=np.float64),
            feature_scale=np.where(spread > SCALE_FLOOR, spread, 1.0),
            output_mean=np.concatenate(
                [training.coordinates.mean(axis=0, dtype=np.float64), np.nanmean(labels, axis=0)]
            ),
            output_scale=np.concatenate([
                np.full(training.coordinates.shape[1], common),
                np.where(label_spread > SCALE_FLOOR, label_spread, 1.0),
            ]),
        )

    def inputs(self, features: NDArray[np.float32]) -> torch.Tensor:
        """Standardise and clip features exactly as :class:`Network` does."""
        scaled = (features.astype(np.float64) - self.feature_mean) / self.feature_scale
        return torch.from_numpy(np.clip(scaled, -INPUT_CLIP, INPUT_CLIP).astype(np.float32))

    def targets(self, table: Table) -> torch.Tensor:
        """Return coordinates and labels in the scaled units of the loss."""
        both = np.concatenate([table.coordinates, table.labels], axis=1).astype(np.float64)
        return torch.from_numpy(((both - self.output_mean) / self.output_scale).astype(np.float32))


def _one_pass(
    model: _Model,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    stepping: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler],
    order: torch.Tensor,
    batch: int,
    gradient_limit: float = 0.0,
) -> float:
    """Run one pass over the training rows in the given order and return the mean loss."""
    inputs, targets, weights = data
    optimiser, schedule = stepping
    model.train()
    running, batches = 0.0, 0
    for start in range(0, inputs.shape[0], batch):
        rows = order[start : start + batch]
        target = targets[rows]
        present = torch.isfinite(target)
        error = torch.where(present, model(inputs[rows]) - torch.nan_to_num(target), 0.0)
        loss = (error**2 * weights).sum(dim=1).mean()
        optimiser.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)  # the method form carries no type annotations
        if gradient_limit > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_limit)
        optimiser.step()
        schedule.step()
        running, batches = running + float(loss.detach()), batches + 1
    return running / batches


def require_one_layout(training: Table, calibration: Table) -> None:
    """Refuse a fit whose two tables measure their features against different things.

    Raises
    ------
    ValueError
        If the tables' feature layouts differ.
    """
    if training.feature_layout != calibration.feature_layout:
        raise ValueError(
            f"training table is {training.feature_layout}-relative, "
            f"calibration table {calibration.feature_layout}-relative"
        )


def _export(
    model: _Model, scaling: _Scaling, settings: FitSettings, training: Table
) -> Network:
    """Copy one evaluated checkpoint into the arrays used by prediction."""
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    return Network(
        feature_mean=scaling.feature_mean.copy(),
        feature_scale=scaling.feature_scale.copy(),
        linear=model.linear.weight.detach().double().numpy(),
        weights=tuple(layer.weight.detach().double().numpy() for layer in linears),
        biases=tuple(layer.bias.detach().double().numpy() for layer in linears),
        output_mean=scaling.output_mean.copy(),
        output_scale=scaling.output_scale.copy(),
        coordinates=settings.coordinates,
        label_names=training.label_names,
        feature_layout=training.feature_layout,
    )


def fit(
    training: Table, calibration: Table, settings: FitSettings,
    selection_score: Callable[[Network], float] | None = None,
) -> tuple[Network, list[dict[str, float]]]:
    """Fit on training rows and keep the best calibration checkpoint.

    Parameters
    ----------
    training, calibration:
        Disjoint training and checkpoint-selection tables.
    settings:
        Network and optimiser settings.
    selection_score:
        Optional larger-is-better score of the exported prediction network. Bind it
        to calibration shots and the complete deployment finisher, including any
        ensemble, and return the final composite score. Validation labels must
        remain unseen. The default selects by pooled calibration flux R².

    Returns
    -------
    Network
        The selected checkpoint as independent prediction arrays.
    list of dict
        Per-epoch loss, flux R², selection score and elapsed seconds.
    """
    require_one_layout(training, calibration)
    if settings.edge_emphasis > 0.0:
        raise ValueError("edge_emphasis is implemented by the accelerator fit only")
    if settings.shot_balance > 0.0:
        raise ValueError("shot_balance is implemented by the accelerator fit only")
    torch.manual_seed(settings.seed)
    torch.set_num_threads(settings.threads)
    generator = torch.Generator().manual_seed(settings.seed)
    scaling = _Scaling.of(training)
    inputs, targets = scaling.inputs(training.features), scaling.targets(training)
    held_inputs = scaling.inputs(calibration.features)
    labels = len(training.label_names)
    variance = training.coordinates.var(axis=0, dtype=np.float64)
    tilt = variance ** -settings.fine_emphasis
    tilt *= variance.sum() / float(np.sum(tilt * variance))
    weights = torch.cat([
        torch.from_numpy(tilt.astype(np.float32)),
        torch.from_numpy(label_loss_weights(tuple(training.label_names), settings)),
    ])
    model = _Model(
        inputs.shape[1], settings.coordinates + labels, settings.width, settings.depth,
        settings.dropout,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    steps = settings.epochs * ((inputs.shape[0] + settings.batch - 1) // settings.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=settings.learning_rate, total_steps=steps, pct_start=0.1
    )
    kept = slice(0, settings.coordinates)
    history: list[dict[str, float]] = []
    best_r2, best_state = -np.inf, None
    for epoch in range(settings.epochs):
        started = time.monotonic()
        order = torch.randperm(inputs.shape[0], generator=generator)
        loss = _one_pass(
            model, (inputs, targets, weights), (optimiser, schedule), order, settings.batch,
            settings.gradient_limit,
        )
        model.eval()
        with torch.no_grad():
            held = model(held_inputs)[:, kept].double().numpy()
        held = held * scaling.output_scale[kept] + scaling.output_mean[kept]
        squared = float(np.sum((held - calibration.coordinates.astype(np.float64)) ** 2))
        r2 = flux_r2(squared, calibration.flux_sums)
        selected = r2 if selection_score is None else float(
            selection_score(_export(model, scaling, settings, training))
        )
        history.append({
            "epoch": epoch + 1, "loss": loss, "calibration_flux_r2": r2,
            "calibration_selection_score": selected,
            "seconds": time.monotonic() - started,
        })
        print(json.dumps(history[-1]), flush=True)
        if np.isfinite(selected) and selected > best_r2:
            best_r2 = selected
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no pass produced a finite calibration score")
    model.load_state_dict(best_state)
    network = _export(model, scaling, settings, training)
    return network, history


def quote(network: Network, table: Table) -> dict[str, float]:
    """Return the pooled flux R² and every label's R² of a network on one table."""
    coordinates, labels = network.predict(table.features)
    squared = float(np.sum((coordinates - table.coordinates.astype(np.float64)) ** 2))
    result = {"flux_r2": flux_r2(squared, table.flux_sums)}
    for index, name in enumerate(table.label_names):
        result[f"{name}_r2"] = label_r2(table.labels[:, index], labels[name])
    return result


def _label_weight_pairs(text: str) -> tuple[tuple[str, float], ...]:
    """Read ``--label-weights`` as literal ``(name, weight)`` pairs, e.g. ``(("li", 0.2),)``."""
    value = ast.literal_eval(text)
    return tuple((str(name), float(weight)) for name, weight in value)


def main() -> int:
    """Fit from a directory of tables and write the network and its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, default=None,
                        help="JSON with a 'stems' list of shots no fit may see")
    defaults = FitSettings()
    for name, value in asdict(defaults).items():
        kind = _label_weight_pairs if isinstance(value, tuple) else type(value)
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind, default=value)
    args = parser.parse_args()
    settings = FitSettings(**{name: getattr(args, name) for name in asdict(defaults)})

    reserved = frozenset(
        [] if args.reserved is None else json.loads(args.reserved.read_text())["stems"]
    )
    parts = {
        part: Table.load(
            args.tables / f"table_{part}.npz", settings.coordinates,
            reserved if part != "validation" else frozenset(),
        )
        for part in ("training", "calibration", "validation")
    }
    print(json.dumps({"settings": asdict(settings), "reserved_shots": len(reserved), "frames": {
        part: int(table.features.shape[0]) for part, table in parts.items()
    }}), flush=True)
    network, history = fit(parts["training"], parts["calibration"], settings)
    args.out.mkdir(parents=True, exist_ok=True)
    network.save(args.out / "network.npz")
    summary = {
        "settings": asdict(settings),
        "basis_ceiling": {part: flux_r2(0.0, table.flux_sums) for part, table in parts.items()},
        "calibration": quote(network, parts["calibration"]),
        "validation": quote(network, parts["validation"]),
        "history": history,
    }
    (args.out / "fit_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("basis_ceiling", "calibration", "validation")}),
          flush=True)
    print("FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
