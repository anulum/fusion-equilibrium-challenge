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
| Three original scored-map reference excerpts and timestamps | `experiments/d72_scored_map_replay/illustration_v1/*/reference.npy`, `times_ms.npy` | Sophelio Fusion Equilibrium Challenge DIII-D training data, CC BY 4.0; original cohort indices 0, 179, 357 at frame zero, represented in scorer float32 precision |
| Original replay contours | `experiments/d72_scored_map_replay/illustration_v1/*/contours.npy` | Extracted reference geometry from the attributed CC BY 4.0 data, alongside this entry's predicted geometry, AGPL-3.0-or-later |
| Original replay grid and envelope | `experiments/d72_scored_map_replay/illustration_v1/{R,Z,envelope}.npy` | Official starter grid and mask, MIT |
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


The full challenge datasets and full-fold prediction archives are not included. The small reference excerpts
above are attributed to Sophelio and the Fusion Equilibrium Challenge authors under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); it was selected and packaged for the figure, alongside
predictions from this entry. `tools/reference/d3d_shot_0047b7837b_submitted_986068.npz` contains this entry's
single-shot Development prediction for the CI comparison. It contains predicted fields, not hidden test labels.

## Model weights

The weights are not in this repository. They are published on Hugging Face as
[anulum/fusion-equilibrium-challenge](https://huggingface.co/anulum/fusion-equilibrium-challenge) at the revision
pinned in `WEIGHTS.json`; that model card declares the licence `agpl-3.0` (the Hugging Face licence tag, which has
no "or later" form). They were fitted on the challenge's DIII-D training data, CC BY 4.0, attributed above.

## Official-method audit baseline

`tools/official_ridge.py` adapts the input alignment, Ip time correction and PCA/Ridge recipe of
[Sophelio's starter](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/tree/823f6eef1f527e1b5566a8fc1e952da6e2fadf43),
particularly `experiments.py` and `data_fixes.py`. The method is independently authored by the organisers;
its reproduction here is performed by this entry, not by an independent outside evaluator.
The training subset, exclusions, random seed, numerical conventions and published predictions are described in
[the calibration record](experiments/audit_calibration/README.md). No MAST equilibrium labels are used.
The four baseline predictions are this entry's generated outputs, fitted on the attributed CC BY 4.0
DIII-D training data; they are not copies of a competitor's model or hidden test labels.

The adapted starter code retains this permission notice (the starter's licence does not license the dataset):

> MIT License
>
> Copyright (c) 2026 Sophelio and the Fusion Equilibrium Challenge authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Software the entry runs on

NumPy, SciPy, PyArrow, scikit-image, scikit-learn, JAX, XGBoost and PyTorch (imported by the training code),
each under its own licence, at the versions pinned in `requirements-*.txt`.

The separate synthetic equilibrium calculations import [FreeGS 0.8.2](https://github.com/freegs-plasma/freegs),
[FreeGS4E 0.15.0](https://github.com/FusionComputingLab/freegs4e) and
[FreeGSNKE 3.1.1](https://github.com/FusionComputingLab/freegsnke). Their inspected
source headers permit LGPL-3.0-or-later. FreeGS and inherited FreeGS4E boundary
code credit Ben Dudson and the University of York; FreeGSNKE credits UKAEA,
UKRI-STFC and its authors. These packages are imported dependencies; their
source is not vendored or relicensed here. Shapely supplies polygon validation
in the same calculation. The two isolated numerical environments are pinned in
`requirements-constructive.txt` and `requirements-constructive-reference.txt`.
These research calculations are separate from the submitted competition solver.

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
