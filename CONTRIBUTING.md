<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
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

## Checking the public diagnostic table

The maintained D71 table verifier has its own checked scope; these commands do not qualify every historical
experiment script or the full inference path:

```bash
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements-quality.txt
.venv/bin/ruff check experiments/d71_finisher_physics/per_shot_table.py tests/test_d71_table.py
.venv/bin/mypy
.venv/bin/python -m coverage run --parallel-mode -m pytest -q tests/test_d71_table.py
.venv/bin/python -m coverage combine
.venv/bin/python -m coverage report
```

Tests use the published CSV and report, including deliberately damaged copies, and require full statement and
branch coverage of `per_shot_table.py`. Frame percentiles and the physical validity of withdrawn diagnostics
remain outside this arithmetic check. The dependency-audit workflow checks all committed requirements files,
including transitive dependencies, and fails on an advisory or an incomplete audit.
The training environments use OSV for CPU-tagged PyTorch and also check its upstream release version against
PyPI advisories. That additional single-package query does not replace the complete dependency-tree audit or
change the installed version. An unauditable dependency fails the strict audit.

For staged source checks, install `gitleaks` on the executable search path and run
`.venv/bin/pre-commit install`. The hooks use the pinned tools in `.venv`; missing tools fail the check.
