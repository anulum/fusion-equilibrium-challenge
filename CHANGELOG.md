<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge entry — changelog
-->
# Changelog

The submitted model, its weights and the inference arithmetic have not changed since the Final submission; every
later change is to documentation, verification tools, experiment records, licensing metadata or CI.

## [Unreleased] - main

### 2026-10-20

- Licensing and governance: `LICENSES/`, `REUSE.toml` (copyright and licence of every file; the organisers' MIT
  code and the CC BY 4.0 dataset excerpts keep their own licences), `GOVERNANCE.md`, `CONTRIBUTORS.md`, issue
  forms and a pull-request template; NOTICE extended (data-fix helper, reconstruction figure, Code of Conduct,
  derived tables, weights).
- Install and training commands use `uv` with fully hashed locks for CPython 3.12 on Linux x86-64.
- Corrected D71 diagnostics (D71b) and the Grad–Shafranov interior test (D72) published with their records
  (`efecd43`); study §4.27, §4.28, ledger rows 2.86 and 2.87.
- Verification tools qualified: the D71 table verifier, the documentation link checker and the submitted-shot
  comparison, with tests on the published files (`6741a82`, `2b49bbd`).
- CI: dependency audits of every lock, including transitive packages and CPU PyTorch; checks that reject omitted
  gates and relaxed thresholds (`bc20d5d`, `52384d5`).
- Wording: reproducibility and validation claims scoped to what was checked (`9a30157`); residual conclusions
  scoped to the experiments tested (`64cbd96`).
- Text and data mining reservation under Art. 4(3) of Directive (EU) 2019/790 (`dbc4ab6`); crawler list in
  `docs/robots.txt` (`014698f`).
- Code archived by Software Heritage (`88e31fd`).

### 2026-10-19

- Finisher diagnostics D71 and the reconstruction figure (`51cc8ad`); corrections to links,
  one install, dated reproduction state, results on the home page (`3fe21ea`).
- Quickstart notebook for one public DIII-D shot, with a comparison against the submitted archive (`d3b04e2`,
  `86a2c61`, `525af63`, `e08f7fa`); "What did not work" page (`54e7aae`).
- CI: weights checksums and a one-shot reproduction against the submitted archive with a tolerance (`666f1a9`,
  `622122b`, `b3bb1aa`).
- Study figures (`95bb6e3`); documentation site on GitHub Pages (`7a1824d`, `2a2d86c`); community files, funding
  and badges (`c4f41b5`, `a4d5b4a`, `5ca34f9`, `933502c`); study §5.9 (`a39fb99`).

## [1.0.0] - 2026-10-19

Record of the entry, archived on Zenodo: version DOI 10.5281/zenodo.23480093, concept DOI
10.5281/zenodo.23480092. Commit `e78c59e`.

## [final-submission-2026-10-19] - 2026-10-19

The code of the Final-phase submission (Codabench 1003274): commit `19d2d34`, weights revision
`398d05cc14b5b0a67b77d9ddfd900e899a94b2da`, Final inputs revision `965040c12b814f33c8e7209f6b1799932247e793`.
Archive SHA-256: DIII-D (879 shots) `8958498fd406bfb259c60fd5a070f594197b6998e60ddd51d56097cfc0f8636a`, MAST
(1 205 shots) `06e0ebd0f0cb3e7fa80ac6f87dda2f0c22aa981b4d96a06cc7da94f1912fc248`. Development submission 986068 (DIII-D S 0.996359,
MAST S 0.714450): on the build host this package reproduces its DIII-D archive bit for bit, and an earlier version
of the package (`package_v2`) its MAST archive.
