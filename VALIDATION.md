<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge entry — what has been validated, and what has not
-->
# Validation

Each claim of this repository rests on one of the checks below. They differ in scope; this page keeps them apart.

## Maintained code under strict gates

Five verification tools are maintained code: `tools/check_doc_links.py`, `tools/ci_check_shot.py`,
`tools/d71b_table.py`, `tools/d72_table.py` and `experiments/d71_finisher_physics/per_shot_table.py`, with their tests in `tests/`. On every push and pull request
the Diagnostics workflow runs Ruff (lint and format), strict mypy and Bandit on them and requires full statement
and branch coverage; the tests use the published files themselves and deliberately damaged copies of them, not
mocks. The same workflow audits every locked dependency of the four `requirements-*.txt` files, transitive ones
included, with `pip-audit --strict`. Each environment retains the resolver's inventory and a separate complete
lock inventory. The latter must match every locked package name and version, including the packages already
installed in the resolver's bootstrap environment. The commands are in
[CONTRIBUTING.md](CONTRIBUTING.md).

The inference and training code, the vendored solver and the experiment records are not under these gates. They
are the code that produced the submission, kept as it was; their quality is shown by the reproduction checks below,
not by lint, type or coverage results. Reproduction evidence is narrower than full native-code acceptance;
those historical qualification gaps are not silently passed.

The figure scripts `tools/make_figures.py`, `tools/make_reconstruction_figure.py` and `tools/make_preview.py`
are not under these gates either. They draw the published figures and the README image from published files; their
output is checked by inspecting the images, not by lint, type or coverage results.

## Reproduction of the submitted predictions

| Check | Where | Scope | Outcome |
|---|---|---|---|
| Whole public DIII-D fold, 874 shots, 193 359 frames | build host (two Intel Xeon E5-2640) | the published package | every map and scalar bit-identical to the submitted Development archive |
| Whole public MAST fold, 1 206 shots, 69 075 frames | build host | an earlier version of the package (`package_v2`) | bit-identical |
| Fresh environment, new directories | build host | a subset of shots | bit-identical on the shots run |
| 22 worker processes against one | build host | the DIII-D fold | bit-identical |
| Final inputs | build host | both Final folds, 879 and 1 205 shots | archives built and checked with the organisers' validator; their SHA-256 values are in the release `final-submission-2026-10-19` |
| One public DIII-D shot on every push (Reproduce workflow) | GitHub-hosted runner, another CPU | weights checksums, one shot | compared with the submitted archive within a tolerance (flux map 1 × 10⁻⁵ of its range, scalars 1 × 10⁻⁴); observed: q95 and βN identical, flux map within 7.5 × 10⁻⁹ |
| One public MAST shot on every push | GitHub-hosted runner | one shot | reconstructed; not compared with an archive |
| Quickstart notebook | Google Colab | one DIII-D shot | runs end to end |

Bit identity is therefore shown on the build host, for the folds named. On other CPUs the last bits of float32 can
differ; the hosted check measures that difference on one shot and is a tolerance check, not a bit-identity proof.
The two hosted input files are pinned to dataset revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb` and checked
against their recorded byte hashes before prediction. Their identities are explicit in the Reproduce workflow.

## Training

`train.py` was run end to end in its limited mode (reduced data and epochs, every model component) on two hosts.
One network of each of the four families was fitted with the full recipe on one NVIDIA GTX 1060 and one AMD
RX 6600 XT ([COMPUTE.md](COMPUTE.md)). The complete production retrain, and the accuracy of an ensemble it would
produce, have not been run; training time on an A100 has not been measured.

## Experiments and scientific claims

[docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md) records the tested configurations and outcomes.
Frozen protocols and decision rules are retained where recorded; this does not retroactively preregister every
exploratory comparison. Reused cohorts are not untouched confirmation data. The finisher diagnostics (study §4.27, D71 and D71b) are numerical diagnostics relative to EFIT, not a validation
of force balance.

## Licensing metadata

Every file's copyright and licence are given by its SPDX header or by `REUSE.toml`; `reuse lint` (REUSE
Specification 3.3) reports the repository compliant.

## Not claimed

- Bit identity on any CPU other than the build host.
- A full production retrain, or its accuracy.
- Retrospective signatures or training-run attestations for the historical releases. The supplementary CI
  provenance described below covers its named source archive and dependency inventories only.
- Accessibility conformance (WCAG) of the documentation site.
- Compliance of the whole repository with any external quality standard beyond the checks named on this page.

## Supplementary source and dependency provenance

The Diagnostics workflow defines a provenance job for successful trusted `main` runs. It packages the exact
Git source object, audits each of the four dependency environments and retains CycloneDX inventories. It uses
GitHub OIDC to attest those subjects, then verifies each digest against the expected repository, commit, main
ref and workflow identity. An altered source archive must be refused. An unsuccessful or skipped job provides
no qualified attestation for that commit. These records are supplementary; they do not backdate a signature of
the old release objects or prove a training run. Runtime locks retain the original direct versions while adding
transitive pins and wheel hashes for CPython 3.12 on Linux x86-64.

To inspect a successful run, first select the commit under review and obtain its Diagnostics run identifier.
Set `FEC_RUN_ID` and `FEC_SOURCE_SHA` to those expected values, then run:

```bash
gh run download "$FEC_RUN_ID" --repo anulum/fusion-equilibrium-challenge \
  --name verified-source-and-sboms --dir verification
gh attestation verify verification/source.tar.gz --bundle verification/attestation.jsonl \
  --repo anulum/fusion-equilibrium-challenge --source-digest "$FEC_SOURCE_SHA" \
  --source-ref refs/heads/main --deny-self-hosted-runners \
  --signer-workflow anulum/fusion-equilibrium-challenge/.github/workflows/diagnostics.yml
```

Verify each `verification/sbom-*.json` in the same way. A valid signature alone does not replace the expected
source/producer policy or the weight checksums. Workflow artifacts have a 30-day retention period; the checked
Git source, locks and weight manifests remain in the repository.
