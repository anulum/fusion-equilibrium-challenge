<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge — contributions and declared verification scope
-->
# Contributing

This repository records one entry to the Fusion Equilibrium Challenge as it was submitted. Its code, weights and
results are identified by the commits named in the README. Bit-for-bit reproduction is qualified only on the
checked hardware and environment; see the reproduction limits in the README.

What is welcome:

- **Reproduction reports**, successful or not, as issues with the template provided.
- **Corrections** to the documentation: a wrong number, a broken link, an unclear sentence. Open an issue or a pull
  request against `docs/` or the README; every number must cite the protocol, receipt or ledger row it comes from.
- **Discussion** of the method and of the negative results in [Discussions](https://github.com/anulum/fusion-equilibrium-challenge/discussions).

What we will not merge: changes to the model, the weights or the inference arithmetic. Work that builds on the entry
belongs in a fork or in a new repository; please cite it (see [CITATION.cff](CITATION.cff)).

By contributing you agree that your contribution is licensed under AGPL-3.0-or-later and you follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Checking the maintained verification tools

The checked scope comprises the D71, D71b and D72 table verifiers, surface-GS support and its table verifier,
the scored-map diagnostic statistics and their verifier, the preview renderer, the publication-boundary guard,
the document-link checker and the submitted-shot comparison tool, with dedicated tests. These gates do not
qualify every historical experiment script or the full inference and training paths:

```bash
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements-quality.txt
.venv/bin/ruff check experiments/d71_finisher_physics/per_shot_table.py tests/test_d71_table.py
.venv/bin/ruff check tools/check_doc_links.py tools/ci_check_shot.py tools/d71b_table.py tools/d72_table.py tools/surface_gs.py tools/surface_support_summary.py tools/surface_support_table.py tools/scored_map_summary.py tools/scored_map_table.py tools/make_preview.py tools/check_publication_boundary.py tests/test_surface_gs.py tests/test_surface_support_summary.py tests/test_surface_support_table.py tests/test_scored_map_summary.py tests/test_scored_map_table.py tests/test_make_preview.py tests/test_check_publication_boundary.py tests/test_d71b_table.py tests/test_d72_table.py tests/test_check_doc_links.py tests/test_ci_check_shot.py tests/test_licensing.py
.venv/bin/mypy
.venv/bin/python -m coverage run --parallel-mode -m pytest -q tests/test_d71_table.py tests/test_d71b_table.py tests/test_d72_table.py tests/test_surface_gs.py tests/test_surface_support_summary.py tests/test_surface_support_table.py tests/test_scored_map_summary.py tests/test_scored_map_table.py tests/test_make_preview.py tests/test_check_publication_boundary.py tests/test_check_doc_links.py tests/test_ci_check_shot.py tests/test_licensing.py
.venv/bin/python -m coverage combine
.venv/bin/python -m coverage report
```

Tests use the published statistics and reports, including deliberately damaged copies, and require full statement and
branch coverage of the twelve maintained modules. The archive tests exercise the actual submitted prediction,
damage its structure and reference metadata, and check refusal under optimised Python. Documentation tests
reject empty roots and links to missing or private destinations, including symlink escapes.
Licence tests check the actual public projection and reject missing licence texts or lost annotations.
Surface-support tests exercise actual contours and affine fits on analytic fields, including partial support, rank
loss and unavailable sources. The publication guard checks staged blobs as well as working files; its tests use
real Git repositories, symbolic links and compressed records. It refuses known private/process record shapes; it
does not certify that every scientific file is appropriate for publication.
The preview tests render the published frame through the actual CLI, check flux normalisation and output
determinism, and refuse damaged or degenerate frames. Installed fonts affect the rendered pixels; cross-host
image identity is not required. To write a preview without replacing the committed image, use
`python tools/make_preview.py --out /path/to/preview.png`.
`make check` runs the native checks, document links, publication boundary and frozen-source comparison without running tests.
`make test` runs the thirteen maintained test modules; the frozen scientific recipes are not in that test denominator.
Frame percentiles and the physical validity of withdrawn diagnostics
remain outside this arithmetic check. The dependency-audit workflow checks all committed requirements files,
including transitive dependencies, and fails on an advisory or an incomplete audit.
The training environments use OSV for CPU-tagged PyTorch and also check its upstream release version against
PyPI advisories. That additional single-package query does not replace the complete dependency-tree audit or
change the installed version. An unauditable dependency fails the strict audit.

For staged source checks, install `gitleaks` on the executable search path and run
`.venv/bin/pre-commit install`. The hooks use the pinned tools in `.venv`; missing tools fail the check.
