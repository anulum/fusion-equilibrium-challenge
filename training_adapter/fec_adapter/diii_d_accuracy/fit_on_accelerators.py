# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — several networks fitted at once, one per accelerator

"""Fit several networks at the same time, one on each accelerator of the workstation.

:mod:`fec_adapter.diii_d_accuracy.fit_network` fits one network on a host
processor in about an hour. The workstation's accelerators already run this
project's solver through JAX, so the same fit is written here in JAX and needs
nothing installed: one process drives every free card, one thread and one
network per card, which is how the cards are used concurrently without a
second accelerator process (two of those do not coexist on this machine).

The fit is the same fit: the same table, scaling, form of network, loss in
score units, decoupled weight decay, and the one-cycle schedule of the rate
and of the first momentum with the library's defaults (start at a twenty-fifth
of the peak, peak after a tenth of the steps, cosine down to a ten-thousandth
of the start; momentum 0.95 → 0.85 → 0.95). Weights start uniform within
``±1/√fan_in``, as the library starts a linear layer. Two things differ and are
stated rather than hidden: a pass drops the last, incomplete batch so that a
pass is one compiled scan, and the random numbers come from another generator,
so a seed here and the same seed there are different draws of the same
procedure. What is kept is decided as there — the pass with the best flux R² on
the calibration part — and the result is the same plain-array
:class:`~fec_adapter.diii_d_accuracy.network.Network`, scored by the same
``quote``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fec_adapter.diii_d_accuracy.fit_network import (
    FitSettings,
    Table,
    _Scaling,
    flux_r2,
    label_loss_weights,
    quote,
    require_one_layout,
)
from fec_adapter.diii_d_accuracy.network import Network

PEAK_AFTER = 0.1
START_DIVISOR = 25.0
END_DIVISOR = 1.0e4
MOMENTUM_HIGH, MOMENTUM_LOW = 0.95, 0.85
SECOND_MOMENTUM = 0.999
EPSILON = 1.0e-8

TRIMMED_SHARE = 1000
"""One calibration frame in this many is left out of the score that chooses the pass to keep."""

CHUNK_ROWS = 100_000
"""Rows scaled at a time on the host, so that no whole-table double-precision copy is made."""

Parameters = dict[str, Any]

_TRANSFER = threading.Lock()

_COMPILING = threading.Lock()
"""Held while a thread builds its programs: five threads tuning matrix products at once
raced in the compiler's autotuner and the process died of a segmentation fault."""


def on_device(array: np.ndarray, device: Any) -> jnp.ndarray:
    """Copy a host table to one card a chunk of rows at a time, one card at a time.

    A single copy of the 1.9 GB training table from ordinary host memory fails on this
    runtime with an illegal-address error, and five threads copying at once is how it was
    met. Chunks of :data:`CHUNK_ROWS` rows under a lock have no such trouble; the pieces
    are joined on the card.
    """
    with _TRANSFER:
        pieces = [
            jax.device_put(array[start : start + CHUNK_ROWS], device)
            for start in range(0, array.shape[0], CHUNK_ROWS)
        ]
        joined: jnp.ndarray = jnp.concatenate(pieces, axis=0)
        joined.block_until_ready()
    return joined


@dataclass(frozen=True)
class Prepared:
    """The scaled tables of a run, made once on the host and shared by every fit of it.

    The scaling depends on the training table alone, not on a variant's settings. Five
    threads each scaling 1.3 million rows of 370 columns in double precision asked for
    more memory than the workstation has; one single-precision copy is 2.3 GB.
    """

    scaling: _Scaling
    inputs: np.ndarray
    targets: np.ndarray
    held_inputs: np.ndarray

    @classmethod
    def of(cls, training: Table, calibration: Table) -> Prepared:
        """Scale both tables exactly as the host fit does, a chunk of rows at a time."""
        require_one_layout(training, calibration)
        scaling = _Scaling.of(training)

        def scaled_inputs(table: Table) -> np.ndarray:
            out = np.empty(table.features.shape, dtype=np.float32)
            for start in range(0, out.shape[0], CHUNK_ROWS):
                rows = slice(start, start + CHUNK_ROWS)
                part = replace(table, features=table.features[rows])
                out[rows] = scaling.inputs(part.features).numpy()
            return out

        return cls(
            scaling, scaled_inputs(training), scaling.targets(training).numpy(),
            scaled_inputs(calibration),
        )


def joined(first: Table, second: Table) -> Table:
    """Return one table holding the rows of both, for a fit on two parts of the split.

    The fits keep the last pass or nearly so (35 of the 41 networks of recipe P1 kept
    pass 59 or 60 of 60), so the calibration part chooses almost nothing and its frames
    serve better as training data; the validation part stays unseen. The flux sums are
    the first table's: they only report that part's basis ceiling.
    """
    require_one_layout(first, second)
    if first.label_names != second.label_names:
        raise ValueError("the two tables do not carry the same labels")
    if first.coordinates.shape[1] != second.coordinates.shape[1]:
        raise ValueError("the two tables do not keep the same coordinates")
    shots = None
    if first.shot_of_row is not None and second.shot_of_row is not None:
        offset = int(first.shot_of_row.max()) + 1 if first.shot_of_row.size else 0
        shots = np.concatenate([first.shot_of_row, second.shot_of_row + offset])
    return replace(
        first,
        features=np.concatenate([first.features, second.features]),
        coordinates=np.concatenate([first.coordinates, second.coordinates]),
        labels=np.concatenate([first.labels, second.labels]),
        shot_of_row=shots,
    )


def _cosine(start: float, end: float, fraction: jnp.ndarray) -> jnp.ndarray:
    return end + (start - end) / 2.0 * (1.0 + jnp.cos(jnp.pi * fraction))


def one_cycle(step: jnp.ndarray, steps: int, peak: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the rate and the first momentum of a step of the one-cycle schedule."""
    rise = float(PEAK_AFTER * steps) - 1.0
    fall = float(steps) - 1.0
    position = step.astype(jnp.float32)
    rising = position <= rise
    up = jnp.clip(position / max(rise, 1.0), 0.0, 1.0)
    down = jnp.clip((position - rise) / max(fall - rise, 1.0), 0.0, 1.0)
    start = peak / START_DIVISOR
    rate = jnp.where(rising, _cosine(start, peak, up), _cosine(peak, start / END_DIVISOR, down))
    momentum = jnp.where(
        rising, _cosine(MOMENTUM_HIGH, MOMENTUM_LOW, up), _cosine(MOMENTUM_LOW, MOMENTUM_HIGH, down)
    )
    return rate, momentum


def initial_parameters(
    key: jnp.ndarray, inputs: int, outputs: int, settings: FitSettings
) -> Parameters:
    """Return weights uniform within ``±1/√fan_in``; the linear path has no bias."""
    sizes = [inputs, *([settings.width] * settings.depth), outputs]
    keys = jax.random.split(key, 2 * (len(sizes) - 1) + 1)
    weights, biases = [], []
    for layer, (before, after) in enumerate(itertools.pairwise(sizes)):
        bound = 1.0 / np.sqrt(before)
        shape = (after, before)
        weights.append(jax.random.uniform(keys[2 * layer], shape, jnp.float32, -bound, bound))
        bias_key = keys[2 * layer + 1]
        biases.append(jax.random.uniform(bias_key, shape[:1], jnp.float32, -bound, bound))
    bound = 1.0 / np.sqrt(inputs)
    linear = jax.random.uniform(keys[-1], (outputs, inputs), jnp.float32, -bound, bound)
    return {"weights": weights, "biases": biases, "linear": linear}


def forward(
    parameters: Parameters, inputs: jnp.ndarray, dropout: float, key: jnp.ndarray | None
) -> jnp.ndarray:
    """Return the raw, still scaled, outputs; ``key`` switches dropout on."""
    hidden = inputs
    for index, (weight, bias) in enumerate(
        zip(parameters["weights"][:-1], parameters["biases"][:-1], strict=True)
    ):
        hidden = jax.nn.silu(hidden @ weight.T + bias)
        if key is not None and dropout > 0.0:
            kept = jax.random.bernoulli(jax.random.fold_in(key, index), 1.0 - dropout, hidden.shape)
            hidden = jnp.where(kept, hidden / (1.0 - dropout), 0.0)
    last = hidden @ parameters["weights"][-1].T + parameters["biases"][-1]
    outputs: jnp.ndarray = last + inputs @ parameters["linear"].T
    return outputs


def _loss(
    parameters: Parameters, inputs: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray,
    dropout: float, key: jnp.ndarray, mix: jnp.ndarray,
    rows: tuple[jnp.ndarray, int] | None = None,
) -> jnp.ndarray:
    """Return the weighted mean squared error of a batch.

    ``rows``, when given, is ``(row_weights, coordinates)``: each row's label error (the
    outputs from ``coordinates`` on) is multiplied by its weight (:func:`row_label_weights`);
    the coordinates' error is not. Without it the loss is the plain one, unchanged.
    """
    present = jnp.isfinite(targets)
    predicted = forward(parameters, inputs, dropout, key)
    error = jnp.where(present, predicted - jnp.nan_to_num(targets), 0.0) @ mix
    if rows is None:
        return jnp.mean(jnp.sum(error**2 * weights, axis=1))
    row_weights, coordinates = rows
    squared = error**2 * weights
    flux = jnp.sum(squared[:, :coordinates], axis=1)
    labels = jnp.sum(squared[:, coordinates:], axis=1)
    return jnp.mean(flux + row_weights * labels)


def row_label_weights(training: Table, settings: FitSettings) -> np.ndarray:
    """Return every training row's weight on its label error (:attr:`FitSettings.shot_balance`).

    Raises
    ------
    ValueError
        If a balance is asked for and the table does not say which shot a row belongs to,
        or the strength is outside ``[0, 1]``.
    """
    strength = settings.shot_balance
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"shot_balance {strength} is outside [0, 1]")
    rows = training.features.shape[0]
    if strength == 0.0:
        return np.ones(rows, dtype=np.float32)
    if training.shot_of_row is None:
        raise ValueError("the table does not say which shot each row belongs to")
    _, which, counts = np.unique(training.shot_of_row, return_inverse=True, return_counts=True)
    balanced = (rows / counts.size) / counts[which]
    weights: np.ndarray = ((1.0 - strength) + strength * balanced).astype(np.float32)
    return weights


def _balance_on(training: Table, settings: FitSettings, device: Any) -> jnp.ndarray | None:
    """Return the rows' label weights on ``device``, or ``None`` when no balance is asked for."""
    if settings.shot_balance == 0.0:
        return None
    return on_device(row_label_weights(training, settings), device)


def edge_mix(
    emphasis: float, directions: np.ndarray, band: np.ndarray, outputs: int
) -> np.ndarray:
    """Return the matrix the output errors are multiplied by before they are squared.

    With orthonormal ``directions`` ``(K, P)`` a coordinate error ``e`` is the map error
    ``e·D``, and counting the cells of ``band`` ``(P,)`` ``1 + emphasis·band`` times gives the
    squared error ``e·M·eᵀ`` with ``M = I + emphasis·D·diag(band)·Dᵀ``. The matrix returned
    holds a root of ``M`` for the coordinates and leaves the labels alone; with no emphasis it
    is the identity and the loss is the plain one.
    """
    mix = np.eye(outputs)
    if emphasis > 0.0:
        count = directions.shape[0]
        metric = np.eye(count) + emphasis * (directions * band[np.newaxis, :]) @ directions.T
        mix[:count, :count] = np.linalg.cholesky(metric)
    return mix.astype(np.float32)


def loss_weights(training: Table, settings: FitSettings) -> np.ndarray:
    """Return the weight of every output in the loss, as the host fit sets them."""
    variance = training.coordinates.var(axis=0, dtype=np.float64)
    tilt = variance ** -settings.fine_emphasis
    tilt *= variance.sum() / float(np.sum(tilt * variance))
    labels = label_loss_weights(tuple(training.label_names), settings)
    return np.concatenate([tilt, labels]).astype(np.float32)


def decayed_adam_step(
    state: tuple[Parameters, Parameters, Parameters, jnp.ndarray],
    gradient: Parameters,
    steps: int,
    settings: FitSettings,
) -> tuple[Parameters, Parameters, Parameters, jnp.ndarray]:
    """Advance parameters and both moments by one step of Adam with decoupled weight decay."""
    current, first, second, step = state
    if settings.gradient_limit > 0.0:
        length = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(gradient)))
        shrink = jnp.minimum(1.0, settings.gradient_limit / (length + EPSILON))
        gradient = jax.tree.map(lambda g: g * shrink, gradient)
    rate, momentum = one_cycle(step, steps, settings.learning_rate)
    count = (step + 1).astype(jnp.float32)
    first = jax.tree.map(lambda m, g: momentum * m + (1.0 - momentum) * g, first, gradient)
    second = jax.tree.map(
        lambda v, g: SECOND_MOMENTUM * v + (1.0 - SECOND_MOMENTUM) * g * g, second, gradient
    )
    # The library corrects the first moment with the momentum of the step at hand.
    correction_first = 1.0 - momentum**count
    correction_second = 1.0 - SECOND_MOMENTUM**count
    current = jax.tree.map(
        lambda p, m, v: p * (1.0 - rate * settings.weight_decay)
        - rate * (m / correction_first) / (jnp.sqrt(v / correction_second) + EPSILON),
        current, first, second,
    )
    return current, first, second, step + 1


def _held_scores(error: np.ndarray, flux_sums: dict[str, float]) -> tuple[float, float]:
    """Return the flux R² without the worst frames (:data:`TRIMMED_SHARE`), and with every frame."""
    by_frame = np.sort(np.sum(error**2, axis=1))
    trimmed = max(1, by_frame.size // TRIMMED_SHARE)
    return (flux_r2(float(by_frame[: by_frame.size - trimmed].sum()), flux_sums),
            flux_r2(float(by_frame.sum()), flux_sums))


def _mix_of(
    settings: FitSettings, edge: tuple[np.ndarray, np.ndarray] | None, outputs: int
) -> np.ndarray:
    if edge is None:
        if settings.edge_emphasis > 0.0:
            raise ValueError("edge_emphasis needs the basis directions and the boundary band")
        return np.eye(outputs, dtype=np.float32)
    return edge_mix(settings.edge_emphasis, edge[0][: settings.coordinates], edge[1], outputs)


def _export(
    parameters: Parameters, scaling: _Scaling, settings: FitSettings, training: Table
) -> Network:
    """Copy a device checkpoint to the double precision prediction representation."""
    parameters = jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float64).copy(), parameters)
    return Network(
        feature_mean=scaling.feature_mean.copy(), feature_scale=scaling.feature_scale.copy(),
        linear=parameters["linear"], weights=tuple(parameters["weights"]),
        biases=tuple(parameters["biases"]),
        output_mean=scaling.output_mean.copy(), output_scale=scaling.output_scale.copy(),
        coordinates=settings.coordinates, label_names=training.label_names,
        feature_layout=training.feature_layout,
    )


def fit(
    training: Table, calibration: Table, settings: FitSettings, device: Any,
    prepared: Prepared | None = None, log: Path | None = None,
    edge: tuple[np.ndarray, np.ndarray] | None = None,
    selection_score: Callable[[Network], float] | None = None,
) -> tuple[Network, list[dict[str, float]]]:
    """Fit on a device and keep the best calibration checkpoint.

    Parameters
    ----------
    training, calibration:
        Disjoint training and checkpoint-selection tables.
    settings:
        Network and optimiser settings.
    device:
        JAX device used for fitting.
    prepared:
        Optional pre-scaled tables shared between fits.
    log:
        Optional path for per-epoch JSON records.
    edge:
        Basis directions and boundary band, required for edge emphasis.
    selection_score:
        Optional larger-is-better score of the exported double precision network.
        Bind it to calibration shots and the complete deployment finisher. Every
        frame participates without trimming. Validation labels must remain unseen.
        The default selects by calibration flux R² with the existing trimming.

    Returns
    -------
    Network
        The selected checkpoint as independent prediction arrays.
    list of dict
        Per-epoch loss, flux R², selection score and elapsed seconds.
    """
    prepared = Prepared.of(training, calibration) if prepared is None else prepared
    scaling = prepared.scaling
    labels = len(training.label_names)
    batches = training.features.shape[0] // settings.batch
    steps = settings.epochs * batches
    kept = slice(0, settings.coordinates)

    with jax.default_device(device):
        inputs = on_device(prepared.inputs, device)
        targets = on_device(prepared.targets, device)
        held = on_device(prepared.held_inputs, device)
        weights, row_weights = (jnp.asarray(loss_weights(training, settings)),
                                _balance_on(training, settings, device))
        key = jax.random.key(settings.seed)
        key, start = jax.random.split(key)
        outputs = settings.coordinates + labels
        mix = jnp.asarray(_mix_of(settings, edge, outputs))
        parameters = initial_parameters(start, inputs.shape[1], outputs, settings)
        state = (
            parameters, jax.tree.map(jnp.zeros_like, parameters),
            jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32),
        )

        # The tables are arguments of the compiled pass, not names it closes over: a closed-over
        # array is baked into the program as a constant, two gigabytes of it here.
        @jax.jit
        def one_pass(
            carry: Any, order_key: jnp.ndarray, table: jnp.ndarray, wanted: jnp.ndarray,
            balance: jnp.ndarray | None,
        ) -> tuple[Any, jnp.ndarray]:
            def update(inner: Any, chosen: jnp.ndarray) -> tuple[Any, jnp.ndarray]:
                value, gradient = jax.value_and_grad(_loss)(
                    inner[0], table[chosen], wanted[chosen], weights, settings.dropout,
                    jax.random.fold_in(key, inner[3]), mix,
                    None if balance is None else (balance[chosen], settings.coordinates),
                )
                return decayed_adam_step(inner, gradient, steps, settings), value

            order = jax.random.permutation(order_key, inputs.shape[0])[: batches * settings.batch]
            carry, values = jax.lax.scan(update, carry, order.reshape(batches, settings.batch))
            return carry, jnp.mean(values)

        @jax.jit
        def held_coordinates(current: Parameters, table: jnp.ndarray) -> jnp.ndarray:
            return forward(current, table, 0.0, None)[:, kept]

        history: list[dict[str, float]] = []
        best_r2, best = -np.inf, None
        truth = calibration.coordinates.astype(np.float64)
        # The pass to keep is chosen without the worst thousandth of the calibration frames: on a
        # handful of wild frames the single-precision forward pass on the card overshoots by
        # orders of magnitude (flux R² of minus 170 in one fit) where the stored network, evaluated
        # in double precision, does not, and such a frame must not choose the pass.
        for epoch in range(settings.epochs):
            started = time.monotonic()
            key, order_key = jax.random.split(key)
            if epoch == 0:
                # The first call compiles, so it is taken alone; later passes run side by side.
                with _COMPILING:
                    state, loss = one_pass(state, order_key, inputs, targets, row_weights)
                    predicted = np.asarray(held_coordinates(state[0], held), dtype=np.float64)
            else:
                state, loss = one_pass(state, order_key, inputs, targets, row_weights)
                predicted = np.asarray(held_coordinates(state[0], held), dtype=np.float64)
            predicted = predicted * scaling.output_scale[kept] + scaling.output_mean[kept]
            r2, every = _held_scores(predicted - truth, calibration.flux_sums)
            selected = r2 if selection_score is None else float(
                selection_score(_export(state[0], scaling, settings, training))
            )
            history.append({
                "epoch": epoch + 1, "loss": float(loss), "calibration_flux_r2": r2,
                "calibration_selection_score": selected,
                "calibration_flux_r2_every_frame": every,
                "seconds": time.monotonic() - started,
            })
            if log is not None:
                with log.open("a") as stream:
                    stream.write(json.dumps(history[-1]) + "\n")
                # One line per pass on the common output too: a watchdog reads silence as a hang.
                print(json.dumps({"network": log.parent.name, **history[-1]}), flush=True)
            if np.isfinite(selected) and selected > best_r2:
                best_r2 = selected
                best = jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float64), state[0])
    if best is None:
        raise RuntimeError("no pass produced a finite calibration score")
    network = _export(best, scaling, settings, training)
    return network, history


def _fit_and_write(
    name: str, settings: FitSettings, device: Any, parts: dict[str, Table], prepared: Prepared,
    out_root: Path, edge: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    network, history = fit(
        parts["training"], parts["calibration"], settings, device, prepared, out / "passes.jsonl",
        edge,
    )
    network.save(out / "network.npz")
    summary = {
        "settings": asdict(settings), "fitted_with": f"jax {jax.__version__} on {device}",
        "training_frames": int(parts["training"].features.shape[0]),
        "basis_ceiling": {part: flux_r2(0.0, table.flux_sums) for part, table in parts.items()},
        "calibration": quote(network, parts["calibration"]),
        "validation": quote(network, parts["validation"]),
        "history": history, "seconds": round(time.monotonic() - started),
    }
    (out / "fit_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    line = {"FITTED": name, "seconds": summary["seconds"], "validation": summary["validation"]}
    print(json.dumps(line), flush=True)
    return line


def main() -> int:
    """Fit the named variants, as many at a time as there are accelerators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True,
                        help='JSON object: {"name": {setting: value, ...}, ...}')
    parser.add_argument("--allow-host", action="store_true",
                        help="fit on the host processor when no accelerator is found")
    parser.add_argument("--train-on-calibration", action="store_true",
                        help="fit on the training and calibration parts together")
    args = parser.parse_args()

    variants: dict[str, dict[str, Any]] = json.loads(args.variants.read_text())
    todo = {name: replace(FitSettings(), **changes) for name, changes in variants.items()
            if not (args.out_root / name / "fit_report.json").is_file()}
    done = [name for name in variants if name not in todo]
    print(json.dumps({"to_fit": list(todo), "already_fitted": done}), flush=True)
    if not todo:
        print("FINISHED: nothing left to fit", flush=True)
        return 0
    devices = jax.devices()
    if not args.allow_host and any(device.platform == "cpu" for device in devices):
        # The runtime falls back to the host without an error when the cards are
        # still held by another process; such a run is five times slower and unwanted.
        raise SystemExit(f"no accelerator found (devices: {devices}); pass --allow-host to go on")
    kinds = {settings.coordinates for settings in todo.values()}
    if len(kinds) != 1:
        raise SystemExit("variants of one run must use the same number of coordinates")
    coordinates = kinds.pop()
    reserved = frozenset(json.loads(args.reserved.read_text())["stems"])
    parts = {
        part: Table.load(args.tables / f"table_{part}.npz", coordinates,
                         reserved if part != "validation" else frozenset())
        for part in ("training", "calibration", "validation")
    }
    if args.train_on_calibration:
        parts["training"] = joined(parts["training"], parts["calibration"])
    print(json.dumps({"devices": [str(device) for device in devices], "frames": {
        part: int(table.features.shape[0]) for part, table in parts.items()},
        "trained_on": ["training", "calibration"] if args.train_on_calibration
        else ["training"]}), flush=True)
    prepared = Prepared.of(parts["training"], parts["calibration"])
    print(json.dumps({"prepared_rows": int(prepared.inputs.shape[0])}), flush=True)
    edge = None
    if any(settings.edge_emphasis > 0.0 for settings in todo.values()):
        with np.load(args.tables / "flux_basis.npz", allow_pickle=False) as stored:
            directions = stored["directions"].astype(np.float64)
        edge = (directions, np.load(args.tables / "edge_band.npy").ravel())
    waiting: queue.SimpleQueue[tuple[str, FitSettings]] = queue.SimpleQueue()
    for item in todo.items():
        waiting.put(item)

    def work_through(device: Any) -> int:
        """Fit waiting variants on one card until none is left; a card never holds two."""
        fitted = 0
        while True:
            try:
                name, settings = waiting.get_nowait()
            except queue.Empty:
                return fitted
            _fit_and_write(name, settings, device, parts, prepared, args.out_root, edge)
            fitted += 1

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        fitted = sum(pool.map(work_through, devices))
    print(f"FINISHED: {fitted} networks fitted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
