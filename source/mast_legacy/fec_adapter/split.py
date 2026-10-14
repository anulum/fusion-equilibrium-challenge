# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the fixed shot-level split of the labelled DIII-D set

"""Say which part of the fixed split a labelled shot belongs to, from its name alone.

The split is decided by a hash of the shot's name, so it cannot drift with the
order of a directory listing, with which shots happen to be on disk, or with a
random seed somebody forgot to pin. ``splits.json`` beside the package lists
the result for the released set; a test holds this rule to that list.

What each part is for:

calibration
    Machine constants and the free physics parameters are fitted here.
validation
    Numbers quoted during development come from here.
holdout
    Opened once, for the final estimate, and by nothing before that. Code that
    reads labels refuses these shots (:func:`refuse_holdout`) so that the rule
    does not depend on anyone remembering it.
training
    Everything else; models are fitted here.
"""
from __future__ import annotations

import hashlib
from typing import Final

BUCKETS: Final = (("calibration", 5), ("validation", 10), ("holdout", 15))
"""Upper bounds, in hundredths, of the three small parts; the rest is training."""

HUNDRED: Final = 100


def bucket_of(stem: str) -> str:
    """Return ``calibration``, ``validation``, ``holdout`` or ``training`` for a shot name."""
    position = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % HUNDRED
    for name, bound in BUCKETS:
        if position < bound:
            return name
    return "training"


def refuse_holdout(stem: str) -> None:
    """Raise if the shot belongs to the part that is opened only once."""
    if bucket_of(stem) == "holdout":
        raise PermissionError(f"{stem} belongs to the holdout part; its labels are not read yet")
