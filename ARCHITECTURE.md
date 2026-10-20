<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge entry — architecture and boundaries
-->
# Architecture

What the repository contains, how the parts depend on each other, what is trusted, and what is kept frozen. The
method itself is described in [docs/METHODS_REPORT.md](docs/METHODS_REPORT.md).

## Parts

| Part | Paths | Role |
|---|---|---|
| Inference entrypoint | `infer.py`, `tree_replica.py`, `deployment.json` | Answers a directory of shot files for one machine and writes the challenge archive and a receipt. `deployment.json` lists model assets and the frozen machine-specific source hashes; its historical fields are explained in DATA_PROVENANCE.md. |
| Weights | `models/` (not tracked; fetched), `WEIGHTS.json`, `WEIGHTS.sha256`, `fetch_weights.py` | `fetch_weights.py` downloads the Hugging Face revision pinned in `WEIGHTS.json` and checks every file against `WEIGHTS.sha256`. |
| DIII-D predictor | `source/d3d_entrypoints/`, `source/d3d_legacy/` | Networks, tree heads and the finishing step for DIII-D. `source/d3d_legacy/vendor/` holds the entrant's solver modules copied from SCPN Fusion Core (`vendor/PROVENANCE.md`) and the organisers' MIT masks. |
| MAST predictor | `source/mast_entrypoints/`, `source/mast_legacy/` | The physics reconstruction for MAST; no model is trained on MAST. |
| Organisers' scoring functions | `source/*/fec_adapter/scoring_kit/`, `training_adapter/fec_adapter/scoring_kit/` | MIT code of the starter kit, kept verbatim except for relative imports, used to read the boundary and the derived scalars the way the scorer does. |
| Training | `train.py`, `training_adapter/`, `training_recipe.json` | From raw training shots to a complete model directory with its own manifest and a copy of the inference runtime. Retains the frozen MAST source. |
| Maintained verifiers | `tools/check_doc_links.py`, `tools/ci_check_shot.py`, `tools/d71b_table.py`, `experiments/d71_finisher_physics/per_shot_table.py`, `tests/` | The code that is linted, typed and tested in CI (see [VALIDATION.md](VALIDATION.md)). |
| Experiment records | `experiments/` | Archival copies of the protocols, scripts and small results of each experiment, with laboratory paths and agent names replaced. Many runners require unredistributed laboratory inputs; the maintained public table verifiers are the explicit exceptions listed above. |
| Documentation | `docs/`, `README.md` and the root `.md` files | GitHub renders the repository documents; GitHub Pages publishes only `docs/`. |

The inference path loads the verified `models/` and imports `source/`, `tree_replica.py` and the pinned packages of
`requirements-inference-cpu.txt`; it does not import `experiments/`, `tools/` or `training_adapter/`. Training uses
its own environment (`requirements-training-*.txt`).

## Inputs and trust

- **Shot files** (Parquet, from the challenge dataset or any directory given as `--split`) are data from outside.
  They are read with PyArrow; the code assumes the challenge's schema and machine conventions and does not try to
  defend against a maliciously crafted file. Run it on data you would open anyway.
- **Model files** are checked against the SHA-256 values of `deployment.json` before they are loaded; NumPy archives
  are read with `allow_pickle=False`, tree models with XGBoost's own loader. The trust anchor is the pair
  `WEIGHTS.sha256` (checked by `fetch_weights.py`) and `deployment.json` (checked inside `infer.py`), both in this
  repository at a known commit.
- **Guards are `assert` statements.** `infer.py` checks the environment, the inputs and every output with 26
  `assert` statements. They are part of the frozen submitted code. Under `python -O` they are removed, so optimised
  Python is not supported: run the entrypoints with ordinary `python`.
- **Fold-dependent MAST fallback.** For a MAST shot in which no frame can be solved, the predictor uses the mean
  current pattern of the solved frames of the supplied fold, and missing q95 or βN take the fold's median. Both come
  from inputs only, so the answer for such a shot depends on which other shots are supplied with it. The receipt
  records the scalar fallback values.

## Lineage of a published answer

source commit (tag) → `WEIGHTS.json` revision and `WEIGHTS.sha256` → `deployment.json` (its SHA-256 is written into
every receipt as `model_manifest_sha256`; `infer.py`'s own SHA-256 as `source_sha256`) → input directory → archive
(its SHA-256 in the receipt) → submission. For the Final submission these identities are listed in the release
`final-submission-2026-10-19`; the Development record is `v1.0.0`.

## What is kept frozen, and why

The Final submission was made from tag `final-submission-2026-10-19`. Its source, `deployment.json`, the weights
revision and the inference arithmetic are not changed afterwards, so that anyone can rebuild exactly what was scored.
Fields of `deployment.json` that describe the day it was frozen are explained in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md) rather than edited. Experiment numerical records remain frozen; publication sanitization is recorded in their export manifests,
and scientific corrections are written beside them. Later work (documentation, verifiers, licensing metadata) lands on `main` without touching
these files.
