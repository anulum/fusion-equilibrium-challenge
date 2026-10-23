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

This repository records the submitted Fusion Equilibrium Challenge entry and subsequent diagnostic studies. Its code, weights and
results are identified by the commits named in the README. Bit-for-bit reproduction is qualified only on the
checked hardware and environment; see the reproduction limits in the README.

What is welcome:

- **Reproduction reports**, successful or not, as issues with the template provided.
- **Corrections** to the documentation: a wrong number, a broken link, an unclear sentence. Open an issue or a pull
  request against `docs/` or the README; every number must cite the protocol, receipt or ledger row it comes from.
- **Discussion** of the method and of the negative results in [Discussions](https://github.com/anulum/fusion-equilibrium-challenge/discussions).

What we will not merge: changes to the submitted model, weights or inference arithmetic. Work that builds on the entry
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
.venv/bin/ruff check tools/check_doc_links.py tools/ci_check_shot.py tools/d71b_table.py tools/d72_table.py tools/surface_gs.py tools/surface_support_summary.py tools/surface_support_table.py tools/scored_map_summary.py tools/scored_map_table.py tools/audit_map_input.py tools/field_diagnostics.py tools/equilibrium_audit.py tools/official_ridge.py tools/audit_calibration.py tools/make_preview.py tools/check_publication_boundary.py tests/test_surface_gs.py tests/test_surface_support_summary.py tests/test_surface_support_table.py tests/test_scored_map_summary.py tests/test_scored_map_table.py tests/test_audit_map_input.py tests/test_field_diagnostics.py tests/test_equilibrium_audit.py tests/test_official_ridge.py tests/test_audit_calibration.py tests/test_make_preview.py tests/test_check_publication_boundary.py tests/test_d71b_table.py tests/test_d72_table.py tests/test_check_doc_links.py tests/test_ci_check_shot.py tests/test_licensing.py
.venv/bin/mypy
.venv/bin/python -m coverage run --parallel-mode -m pytest -q tests/test_d71_table.py tests/test_d71b_table.py tests/test_d72_table.py tests/test_surface_gs.py tests/test_surface_support_summary.py tests/test_surface_support_table.py tests/test_scored_map_summary.py tests/test_scored_map_table.py tests/test_audit_map_input.py tests/test_field_diagnostics.py tests/test_equilibrium_audit.py tests/test_official_ridge.py tests/test_audit_calibration.py tests/test_make_preview.py tests/test_check_publication_boundary.py tests/test_check_doc_links.py tests/test_ci_check_shot.py tests/test_licensing.py
.venv/bin/python -m coverage combine
.venv/bin/python -m coverage report
```

Tests use the published statistics and reports, including deliberately damaged copies, and require full statement and
branch coverage of the seventeen maintained modules. The archive tests exercise the actual submitted prediction,
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
`make check` runs both maintained scopes' native checks, document links, publication boundary and frozen-source
comparison without running tests. It requires the additional constructive environment described below.
`make test` runs the eighteen diagnostic test modules in the quality environment; the six constructive test
modules have a separate target. The frozen scientific recipes are not in either test denominator.
Frame percentiles and the physical validity of withdrawn diagnostics
remain outside this arithmetic check. The dependency-audit workflow checks all committed requirements files,
including transitive dependencies, and fails on an advisory or an incomplete audit.
The training environments use OSV for CPU-tagged PyTorch and also check its upstream release version against
PyPI advisories. That additional single-package query does not replace the complete dependency-tree audit or
change the installed version. An unauditable dependency fails the strict audit.

For staged source checks, install `gitleaks` on the executable search path and run
`.venv/bin/pre-commit install`. The hooks use the pinned tools in `.venv`; missing tools fail the check.

The external-map audit is exercised through actual NPZ files and its public command; see
[the audit guide](docs/AUDIT_TOOL.md) for units, fixed regions and mathematical definitions.
Analytic controls test discrete operators and applicability, not a force-balance certificate.

Calibration tests check the h² stencil floor against an analytic source, controlled perturbations, unavailable
diagnostics, common-sign invariance, bounded illustration archives and fixed reference-assisted regions.
Baseline tests exercise real Parquet alignment, both Ip timestamp cases, actual PCA–Ridge fitting and the
offline public command, with disjoint-shot and numerical-range refusals. Its small test fixtures are analytic
controls, not independent research baselines. The published baseline's scope and refit command are in
[the calibration guide](docs/AUDIT_CALIBRATION.md).

## Checking the synthetic equilibrium tools

The six `tools/constructive_*.py` modules and six `tests/test_constructive_*.py` modules use a numerical
environment incompatible with the older diagnostic scope's dependency versions. Create it with:

```bash
UV_LINK_MODE=hardlink uv venv --python 3.12.3 .venv-constructive
uv pip install --python .venv-constructive/bin/python --require-hashes -r requirements-constructive.txt
make constructive-quality
```

The commit hooks check their lint, formatting, strict types and source security. Type checking uses
`.venv-constructive`, while the other static tools use the pinned quality environment `.venv`.
Missing environments fail the applicable checks.

`make constructive-test` runs the complete numerical integrations and requires both
`CONSTRUCTIVE_REFERENCE_PYTHON` and `CONSTRUCTIVE_WRONG_VERSION_PYTHON` to identify executable, separately
provisioned interpreters. Their defaults name `.venv-constructive-reference/bin/python` and
`.venv-constructive-wrong-version/bin/python`. The reference interpreter also needs the exact coverage
observer from the quality lock. The deliberate wrong-version interpreter uses the modern lock with the
hash-pinned NumPy probe substituted. The [dedicated CI job](.github/workflows/diagnostics.yml) contains the
complete provisioning recipe. Both interpreter paths are asserted before testing, so this target cannot
report complete coverage by silently skipping the integrations. It retains a separate coverage data file and
requires 100% statement and branch coverage for the six-source scope.

For the small original-map arithmetic check, only the modern environment is needed:

```bash
.venv-constructive/bin/python -m pytest -q tests/test_constructive_pairs.py::test_recorded_coarse_solver_gate
```

[Scientific definitions and reproduction](docs/CONSTRUCTIVE_PAIRS.md) distinguish the failed compatibility
gate, the separate prescribed-source boundary control and software integration checks. The complete numerical
tests run on the hosted runner; a local fast subset is not the full validation result.
