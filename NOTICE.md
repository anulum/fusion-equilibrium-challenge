<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Notices

## This entry

© Concepts 1996–2026 Miroslav Šotek. © Code 2020–2026 Miroslav Šotek. Source code is released under
AGPL-3.0-or-later (`LICENSE`); the SPDX identifier at the top of each file governs. A commercial licence is
available from the entrant: protoscience@anulum.li.

## Third-party material kept in this repository

| Material | Where | Origin and licence |
|---|---|---|
| Derivation of the seven map-derived scalars, boundary extraction and axis search | `source/*/fec_adapter/scoring_kit/`, `training_adapter/fec_adapter/scoring_kit/` | Fusion Equilibrium Challenge starter kit, © 2026 Sophelio and the Fusion Equilibrium Challenge authors, MIT; kept verbatim except for relative imports, licence text beside the files |
| Vessel envelope masks | `source/*/vendor/challenge_masks/` | the same starter kit, MIT; licence text beside the files |

The solver modules under `source/*/vendor/scpn_fusion/` are the entrant's own code, copied from SCPN Fusion Core
at a recorded commit; `vendor/PROVENANCE.md` lists the copy and the changes made here.

## Not redistributed

The challenge dataset (`Sophelio/fusion-equilibrium-challenge`, CC BY 4.0) is not part of this repository. No
test-fold prediction is stored here.

## Software the entry runs on

NumPy, SciPy, PyArrow, scikit-image, scikit-learn, JAX, XGBoost and PyTorch (imported by the training code),
each under its own licence, at the versions pinned in `requirements-*.txt`.
