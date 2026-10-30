# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — scored-field measurement tests
"""Remeasure real D72 maps and exercise the complete public command and refusals."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.scored_map_bundle import load, load_shot
from tools.scored_map_measure import field_error, measure, stencil_classes
from tools.scored_map_summary import RAW

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "experiments/d72_scored_map_replay/illustration_v1"
EXPECTED = EXAMPLE / "expected_statistics.json.gz"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_frame_diagnostics_and_surface_support() -> None:
    """Public remeasurement matches the independently stored original matched-shot frame arrays."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    with np.load(EXAMPLE / "original_frame_diagnostics.npz", allow_pickle=False) as original:
        for i, case in enumerate(bundle.cases):
            result = measure(bundle, load_shot(bundle, case))
            np.testing.assert_allclose(
                result.values, original["values"][:, i : i + 1], rtol=1e-12, atol=1e-13
            )
            np.testing.assert_allclose(
                result.reference_inside,
                original["reference_inside_current"][i : i + 1],
                rtol=1e-12,
                atol=1e-13,
            )
            assert result.support["used_levels"].shape == (1, 3, 4)
            np.testing.assert_array_equal(
                result.support["full_support"], result.support["used_levels"].all(axis=2)
            )
            np.testing.assert_array_equal(
                result.values[RAW.index("exterior_abs_current"), :, 1],
                result.values[RAW.index("exterior_abs_current"), :, 2],
            )


@pytest.mark.parametrize("mode", ["exterior", "unused", "foreign"])
def test_broken_projection_invariants_are_refused(mode: str) -> None:
    """Actual-map corruption cannot pass original domain or case-membership guards."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    data = load_shot(bundle, bundle.cases[0])
    if mode == "foreign":
        data = replace(data, case=replace(data.case, stem="foreign"))
    elif mode == "unused":
        data = replace(data, used=np.zeros(1, dtype=bool))
    else:
        projected = data.fields[2].copy()
        projected[0, 0, 0] += 1
        data = replace(data, fields=(data.fields[0], data.fields[1], projected))
    with pytest.raises(ValueError):
        measure(bundle, data)


def test_absent_candidate_contours_keep_unavailable_surface_support() -> None:
    """Explicitly absent candidate geometry stays unavailable, with all reference data retained."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    data = load_shot(bundle, bundle.cases[0])
    reference = data.contour(0, 0)
    assert reference is not None
    data = replace(
        data,
        contours=reference.copy(),
        offsets=np.array([0, len(reference), len(reference), len(reference)], dtype=np.int64),
        fields=(data.fields[0], data.fields[1], data.fields[1]),
        used=np.zeros(1, dtype=bool),
        converged=np.zeros(1, dtype=bool),
        iterations=np.zeros(1, dtype=np.int64),
        pixels=np.zeros(1, dtype=np.int64),
    )
    result = measure(bundle, data)
    assert np.isnan(result.values[RAW.index("gs_surface"), 0, 1:]).all()
    assert not result.support["full_support"][0, 1:].any()
    assert np.all(result.support["reason"][0, 1:] == 1)
    assert not result.support["used_levels"][0, 1:].any()


def test_actual_stencil_partition_and_insufficient_field_support() -> None:
    """The real reference domain partitions every inner stencil and preserves an unavailable small region."""
    bundle = load(EXAMPLE, _hash(EXAMPLE / "manifest.json"))
    data = load_shot(bundle, bundle.cases[0])
    replaced = np.zeros((65, 65), dtype=bool)
    replaced[20:40, 20:40] = True
    classes = stencil_classes(replaced)
    assert sum(int(mask.sum()) for mask in classes.values()) == 63**2
    assert all(
        not np.any(a & b) for i, a in enumerate(classes.values()) for b in list(classes.values())[i + 1 :]
    )
    phi = data.fields[0][0].astype(float)
    region = np.zeros((65, 65), dtype=bool)
    assert np.isnan(field_error(phi, phi, phi, phi, region, bundle.grid.radius))
