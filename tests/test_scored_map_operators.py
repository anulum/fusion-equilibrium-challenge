# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — frozen numerical boundary tests
"""Use the original published fields and reject changed numerical dependencies."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tools import scored_map_operators
from tools.scored_map_operators import ROOT, SOURCES, load


def test_original_operators_on_published_map_restore_aliases() -> None:
    """Actual imports preserve previously loaded modules and return original operators."""
    previous = {name: sys.modules.get(name) for name in ("physics", "critical", "physics_b")}
    try:
        for name in previous:
            sys.modules.pop(name, None)
        operators = load()
        assert all(name not in sys.modules for name in previous)
        assert operators.source_sha256 == SOURCES
        with np.load(ROOT / "tools/reference/d71_figure_frames.npz", allow_pickle=False) as data:
            radius, height = data["R"], data["Z"]
            phi = data["p50_margin_psi"].astype(float)
            contour = data["p50_margin_lcfs"]
        inside = operators.physics.inside(contour, radius, height, operators.kit["points_in_polygon"])
        assert inside.dtype == bool and inside.any()
        assert not operators.physics.inside(None, radius, height, operators.kit["points_in_polygon"]).any()
        pr, pz, gs = operators.physics.derivatives(phi, radius, height)
        assert pr.shape == pz.shape == gs.shape == (65, 65)
        assert np.isfinite(gs[1:-1, 1:-1]).all()
        density = operators.corrected.density(phi, radius, height, 1.0)
        assert np.isfinite(density).all() and np.all(density >= 0)
        assert operators.physics.dilate(inside, 1).sum() >= inside.sum()
        assert operators.physics.MIN_PIXELS == 30 and operators.physics.EDGE_CELLS == 2
        aliases = {name: ModuleType(name) for name in previous}
        sys.modules.update(aliases)
        again = load()
        assert all(sys.modules[name] is aliases[name] for name in previous)
        assert again.physics.derivatives(phi, radius, height)[2].shape == (65, 65)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.parametrize("name", list(SOURCES))
def test_changed_frozen_source_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Changing any genuine copied numerical source fails before module execution."""
    for relative in SOURCES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    with (tmp_path / name).open("ab") as stream:
        stream.write(b"\n")
    monkeypatch.setattr(scored_map_operators, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="dependency differs"):
        load()
