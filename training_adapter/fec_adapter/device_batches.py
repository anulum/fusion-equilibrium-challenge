# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — one batch of frames solved on every available device

"""Spread one batch of frames over every device and solve the parts at the same time.

The frames of a batch are independent, so nothing has to travel between the
devices: the batch is cut into parts of one fixed length, every device takes the
next part as soon as it has finished the last, and the parts are put back in
their original order. There are no collectives, which matters on a machine whose
cards sit on single-lane risers.

One host thread per device is not a convenience. The solve is a compiled
``while`` loop whose exit condition is read back by the thread that launched it,
so that thread is held until its device has finished. Launched from a single
thread, five cards each reach full utilisation *one after the other* and the
batch takes as long as on one card — measured: 20.5 s against 22.1 s for 640
frames. With one thread per device the same batch takes 4.3 s.

Every part has the same length whatever the size of the batch, because the
compiled program is specialised to the part length: a length that followed the
number of frames in a shot would be compiled again for every shot — measured at
about a minute per setting, 225 s of a 319 s shot. A short last part is filled
by repeating its final frame; the copies are dropped before the result is
returned, and they cannot lengthen the solve because a copy finishes exactly
when its original does.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_batched,
)

from .equilibrium import Geometry


@dataclass(frozen=True)
class FrameBatch:
    """The per-frame inputs of one batch, in frame order.

    Attributes
    ----------
    coil_current:
        Filament currents in amperes, shape ``(B, N_coils)``. The solenoid
        filaments must already be zero: its flux is superposed after the solve,
        as :func:`fec_adapter.equilibrium.solve_frame` records.
    pprime, ffprime:
        Free functions on the normalised-flux knots, shape ``(B, N_knots)``.
    plasma_current:
        Measured plasma current of each frame in amperes, shape ``(B,)``.
    psi_init:
        Start field of each frame, shape ``(B, NZ, NR)``, or ``None`` for the
        solver's cold start from the coil field.
    """

    coil_current: NDArray[np.float64]
    pprime: NDArray[np.float64]
    ffprime: NDArray[np.float64]
    plasma_current: NDArray[np.float64]
    psi_init: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Refuse inputs whose leading dimensions disagree."""
        size = self.coil_current.shape[0]
        lengths = {
            "pprime": self.pprime.shape[0],
            "ffprime": self.ffprime.shape[0],
            "plasma_current": self.plasma_current.shape[0],
        }
        if self.psi_init is not None:
            lengths["psi_init"] = self.psi_init.shape[0]
        wrong = {name: length for name, length in lengths.items() if length != size}
        if size == 0 or wrong:
            raise ValueError(f"batch of {size} frames has inputs of other lengths: {wrong}")

    @property
    def size(self) -> int:
        """Number of frames in the batch."""
        return int(self.coil_current.shape[0])


@dataclass(frozen=True)
class Machine:
    """What every frame of a batch shares: the grid, the filaments and the knots.

    Attributes
    ----------
    geom:
        Grid quantities from :func:`fec_adapter.equilibrium.geometry`.
    coil_r, coil_z:
        Filament positions in metres, shape ``(N_coils,)``.
    psi_n_knots:
        Normalised-flux knots the free functions are sampled on.
    """

    geom: Geometry
    coil_r: NDArray[np.float64]
    coil_z: NDArray[np.float64]
    psi_n_knots: NDArray[np.float64]


@dataclass(frozen=True)
class SolverSettings:
    """Iteration settings of the batched solve.

    Attributes
    ----------
    n_iter:
        Cap on the outer Anderson iterations.
    ip_ramp:
        Iterations over which the plasma current is ramped up. A cold start
        needs the ramp; a warm start must use ``1``, because ramping the current
        from near zero walks the iterate away from the field it was handed.
    part_length:
        Frames solved together on one device. It fixes the compiled program's
        shape, so it is a setting and not a function of the batch.
    """

    n_iter: int = 180
    ip_ramp: int = 30
    part_length: int = 64


def part_slices(size: int, part_length: int) -> list[slice]:
    """Cut ``size`` frames into consecutive parts of ``part_length``; only the last falls short."""
    if size < 1 or part_length < 1:
        raise ValueError(
            f"need at least one frame and a positive part length, got {size} and {part_length}"
        )
    return [slice(start, min(start + part_length, size)) for start in range(0, size, part_length)]


def solve_across_devices(
    machine: Machine,
    batch: FrameBatch,
    settings: SolverSettings,
    devices: Sequence[jax.Device] | None = None,
    fixed_support: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Solve a batch of frames on every device at once.

    Parameters
    ----------
    machine:
        The inputs every frame shares.
    batch:
        The per-frame inputs.
    settings:
        Iteration settings, shared by every part.
    devices:
        Devices to use; every device JAX reports when omitted. Which cards JAX
        may see at all is decided earlier, by ``env.sh``, which hides any card
        that drives a monitor.
    fixed_support:
        Plasma support weights shared by the batch, shape ``(NZ, NR)``, or
        ``None`` for the solver's own support, found from each iterate. With a
        fixed support the current cannot collapse into a sliver, which is what
        the Newton path's first stage relies on as well.

    Returns
    -------
    psi:
        Solver flux of every frame in the order given, shape ``(B, NZ, NR)``,
        without the solenoid.
    iterations:
        Outer iteration count of every frame, shape ``(B,)``. A count equal to
        ``settings.n_iter`` must be read as not converged.
    """
    geom = machine.geom
    pool = list(jax.devices() if devices is None else devices)
    if not pool:
        raise ValueError("no device to solve on")
    parts = part_slices(batch.size, settings.part_length)
    length = settings.part_length
    waiting: SimpleQueue[int] = SimpleQueue()
    for number in range(len(parts)):
        waiting.put(number)
    solved: dict[int, tuple[NDArray[np.float64], NDArray[np.int64]]] = {}

    def solve_part(
        device: jax.Device, rows: slice
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        kept = rows.stop - rows.start

        def on_device(values: NDArray[np.float64] | jnp.ndarray) -> jnp.ndarray:
            return jax.device_put(jnp.asarray(values), device)

        def part_of(values: NDArray[np.float64]) -> jnp.ndarray:
            chunk = values[rows]
            fill = np.repeat(chunk[-1:], length - kept, axis=0)
            return on_device(np.concatenate([chunk, fill]))

        result = solve_predictive_equilibrium_batched(
            part_of(batch.coil_current),
            part_of(batch.pprime),
            part_of(batch.ffprime),
            on_device(geom.r_grid),
            on_device(geom.z_grid),
            on_device(machine.coil_r),
            on_device(machine.coil_z),
            on_device(machine.psi_n_knots),
            part_of(batch.plasma_current),
            on_device(geom.response),
            on_device(geom.wall_idx),
            on_device(geom.source_idx),
            psi_init=None if batch.psi_init is None else part_of(batch.psi_init),
            fixed_support_weights=None if fixed_support is None else on_device(fixed_support),
            n_iter=settings.n_iter,
            ip_ramp=settings.ip_ramp,
            return_iterations=True,
        )
        if not isinstance(result, tuple):
            raise TypeError("the batched solve did not return its iteration counts")
        psi, iterations = result
        return (
            np.asarray(psi, dtype=np.float64)[:kept],
            np.asarray(iterations, dtype=np.int64)[:kept],
        )

    def drain(device: jax.Device) -> None:
        """Solve parts on one device until none is waiting."""
        while True:
            try:
                number = waiting.get_nowait()
            except Empty:
                return
            solved[number] = solve_part(device, parts[number])

    with ThreadPoolExecutor(max_workers=len(pool)) as threads:
        for outcome in [threads.submit(drain, device) for device in pool]:
            outcome.result()
    return (
        np.concatenate([solved[number][0] for number in range(len(parts))]),
        np.concatenate([solved[number][1] for number in range(len(parts))]),
    )
