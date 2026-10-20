<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge entry — third-party notices and attribution
-->
# Notices

## This entry

© Concepts 1996–2026 Miroslav Šotek. © Code 2020–2026 Miroslav Šotek. Source code is released under
AGPL-3.0-or-later (`LICENSE`); the SPDX identifier at the top of each file governs. A commercial licence is
available from the entrant: protoscience@anulum.li.

## Third-party material kept in this repository

| Material | Where | Origin and licence |
|---|---|---|
| Derivation of the seven map-derived scalars, boundary extraction and axis search | `source/*/fec_adapter/scoring_kit/`, `training_adapter/fec_adapter/scoring_kit/` | [Fusion Equilibrium Challenge starter kit](https://github.com/Sophelio/fusion-equilibrium-challenge-starter), © 2026 Sophelio and the Fusion Equilibrium Challenge authors, MIT; kept verbatim except for relative imports, licence text beside the files |
| Vessel envelope masks | `source/*/vendor/challenge_masks/` | the same starter kit, MIT; licence text beside the files |
| Helper that fixes two errata of dataset release v1.1.0 (Ip time axis, MAST Thomson channel alignment) | `experiments/D28_pressure_reliability_2026-10-09T202147Z/official_data_fixes.py` | the same starter kit's `data_fixes.py`, MIT; an archival copy kept unchanged, without a header; its licence and copyright are given in `REUSE.toml` |
| Four EFIT reference flux frames and their contours used in the reconstruction figure | `tools/reference/d71_figure_frames.npz`, reference fields and contours | [Sophelio Fusion Equilibrium Challenge dataset](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge), CC BY 4.0; selected DIII-D development frames, with shot identifiers and timestamps retained |
| The same four reference frames, rendered | `docs/figures/fig6_reconstruction.png` (reference panels, and the reference in each difference panel) | the same dataset, CC BY 4.0; adapted: drawn as contours and subtracted from this entry's predictions |
| Code of Conduct | `CODE_OF_CONDUCT.md` | [Contributor Covenant](https://www.contributor-covenant.org/) 2.1, CC BY 4.0; the contact address is this entry's |

The solver modules under `source/*/vendor/scpn_fusion/` are the entrant's own code, copied from SCPN Fusion Core
at a recorded commit; `vendor/PROVENANCE.md` lists the copy and the changes made here.

## Data and prediction excerpts

Shot identifiers of the challenge dataset appear throughout this repository: in the training and evaluation splits
(`source/*/splits.json`, `source/*/reserved_for_evaluation.json`), in the experiment records and in tables derived
from the dataset, such as the per-shot diagnostics `experiments/d71_finisher_physics/per_shot_v1.csv.gz`, its
dependency inventory `dependencies_v1.json.gz` and the D71b report. These tables hold this entry's own measurements
on the dataset's shots, not the dataset's fields; the dataset is attributed to Sophelio and the Fusion Equilibrium
Challenge authors under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


The full challenge datasets and full-fold prediction archives are not included. The small reference excerpt
above is attributed to Sophelio and the Fusion Equilibrium Challenge authors under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); it was selected and packaged for the figure, alongside
predictions from this entry. `tools/reference/d3d_shot_0047b7837b_submitted_986068.npz` contains this entry's
single-shot Development prediction for the CI comparison. It contains predicted fields, not hidden test labels.

## Model weights

The weights are not in this repository. They are published on Hugging Face as
[anulum/fusion-equilibrium-challenge](https://huggingface.co/anulum/fusion-equilibrium-challenge) at the revision
pinned in `WEIGHTS.json`; that model card declares the licence `agpl-3.0` (the Hugging Face licence tag, which has
no "or later" form). They were fitted on the challenge's DIII-D training data, CC BY 4.0, attributed above.

## Software the entry runs on

NumPy, SciPy, PyArrow, scikit-image, scikit-learn, JAX, XGBoost and PyTorch (imported by the training code),
each under its own licence, at the versions pinned in `requirements-*.txt`.

## Text and data mining reservation

The rightsholder, Miroslav Šotek (ANULUM), expressly reserves the use of the
works in this repository for text and data mining, including the collection of
this content for, and its use in, training, fine-tuning or evaluating machine
learning and artificial intelligence models, in accordance with Article 4(3) of
Directive (EU) 2019/790 and the national provisions transposing it.

No permission for such use has been granted beyond what the licence in
`LICENSE` grants. This reservation does not change that licence: uses the
licence permits remain permitted on its terms, including its attribution and
share-alike conditions.

Reservation made on 2026-10-19. Requests for permission: contact the
rightsholder through the contact listed in this repository.
