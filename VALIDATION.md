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

Eighteen verification, diagnostic, rendering and baseline-reproduction tools are maintained code: `tools/check_doc_links.py`, `tools/ci_check_shot.py`,
`tools/d71b_table.py`, `tools/d72_table.py`, `tools/surface_gs.py`, `tools/surface_support_summary.py`, `tools/surface_support_table.py`, `tools/scored_map_summary.py`, `tools/scored_map_table.py`, `tools/audit_map_input.py`, `tools/field_diagnostics.py`, `tools/equilibrium_audit.py`, `tools/competition_map.py`, `tools/official_ridge.py`, `tools/audit_calibration.py`, `tools/make_preview.py`, `tools/check_publication_boundary.py` and `experiments/d71_finisher_physics/per_shot_table.py`, with their tests in `tests/`. On every push and pull request
the Diagnostics workflow runs Ruff (lint and format), strict mypy and Bandit on them and requires full statement
and branch coverage; the tests use the published files themselves and deliberately damaged copies of them, not
mocks. The same workflow audits every locked dependency of the six root `requirements-*.txt` files, transitive ones
included, with `pip-audit --strict`. Each environment retains the resolver's inventory and a separate complete
lock inventory. The latter must match every locked package name and version, including the packages already
installed in the resolver's bootstrap environment. The commands are in
[CONTRIBUTING.md](CONTRIBUTING.md).

The surface-GS helper exposes per-level support, rank and conditioning without changing the historical contour
weights or fit inclusion. Analytic tests exercise complete, partial, rank-deficient and unavailable cases. Its
ratio was also compared with the frozen D71b implementation on 12 published illustration maps and 300 synthetic
perturbed elliptical maps, with exact agreement on the checked host. A separate census of all 358 historical
D71b shots (78,875 frames at ten map stages) found complete four-surface support throughout, with all fits of
rank two. The published per-shot aggregates and checksum-bound command reproduce its support counts,
equal-shot means and 20,000-draw intervals. This verifies the table's arithmetic, without authenticating the
measurements, remeasuring private maps or reproducing frame quantiles. The census uses the historical float64
stage snapshots; it is not an audit of the scored float32 D72 projection maps.
See the [support diagnostic and reproduction command](experiments/d71b_surface_support/README.md).

The scored-map statistics and verifier separately reproduce diagnostic means, common-frame comparisons and
20,000-draw intervals for the actual float32 D72 projection comparison. The full 358-shot, 78,875-frame
measurement retains the repeatedly examined development cohort. The public command verifies typed support,
availability, caller-pinned identities and aggregate arithmetic; its tests use the complete release and
changed copies. It does not regenerate maps, authenticate private measurements, reproduce frame quantiles
or certify force balance. [Definitions and measured results](experiments/d72_scored_map_physics/README.md)
keep own-map GS domains, common exterior, stencil classes and convergence subsets separate.

The external-map audit accepts explicitly declared units and orientation, fixed comparison masks and optional
references/current measurements. Its maintained input, field and CLI modules are covered by analytic discrete
operator controls, actual malformed archives, convention/gauge comparisons and a published DIII-D illustration
through the real command. Rank, surface support, zero denominators and unavailable domains are reported rather
than certified. Those operator tests alone do not evaluate an external predictor or establish independent physical truth. [Input contract and numerical definitions](docs/AUDIT_TOOL.md) describe the scope.

The competition adapter selects one frame from the actual prediction-archive format into the same audit
contract. A process-level test selects a frame from the published submission 986068 archive, runs the real
audit command and compares its diagnostics with the original frame's direct audit. Dedicated tests cover
storage layout, exact caller-supplied timestamps, provenance and malformed inputs. Coordinates, units and
orientation remain explicit; this adapter does not reproduce pooled competition scoring or validate the
unselected predictions.

The calibration adds a known continuous/discrete source comparison on three resolutions and fixed
perturbation, unavailable and common-sign controls. Four checksum-bound public illustrations compare the
entry and the organisers' independently authored PCA–Ridge method on identical reference-assisted regions.
This method was fitted by the entry on 512 other released DIII-D shots, excluding the entire 358-shot cohort;
it is not independent external reproduction. Its maintained offline fitter has real Parquet, alignment,
fit, provenance and public-command tests. The submitted training path remains frozen and outside these gates.
[Calibration definitions and reproduction](docs/AUDIT_CALIBRATION.md) state the population and physical limits.

The publication guard runs in commit hooks, Diagnostics and immediately before the CI source archive. It reads
the staged blobs and working files, rejects symbolic links and recognised private/process record shapes, and
checks all members of gzip files within a fixed decompression limit. Tests cover actual Git staging, exported
trees and refusal cases. This is a bounded publication check, not a general confidentiality guarantee.

The submitted inference and training code, the vendored solver and the historical experiment runners are not under these gates. They
are the code that produced the submission, kept as it was; their quality is shown by the reproduction checks below,
not by lint, type or coverage results. Reproduction evidence is narrower than full native-code acceptance;
those historical qualification gaps are not silently passed.

The preview renderer reads the published reconstruction frame. Tests check its flux normalisation, boundary
data and distance, render determinism and image dimensions through the actual CLI, and refusal of missing,
damaged or degenerate frames. Its pixels depend on installed fonts. The other figure scripts
`tools/make_figures.py` and `tools/make_reconstruction_figure.py` remain outside these gates; their outputs are
checked by inspecting the images.

## Synthetic equilibrium calculations

Seven additional tools are maintained in a separate numerical environment:
`tools/constructive_geometry.py`, `tools/constructive_model.py`, `tools/constructive_observations.py`,
`tools/constructive_pairs.py`, `tools/constructive_reference.py`, `tools/constructive_boundary.py`
and `tools/constructive_plot.py`.
The dedicated Diagnostics integration job checks these files and their seven test modules with Ruff,
strict mypy and Bandit, then requires 100% statement and branch coverage for this separate scope.
This does not extend the eighteen-tool coverage result to the submitted models, third-party solvers
or all historical research code.

The modern forward runtime and the older inverse runtime are isolated, hash-pinned environments.
The legacy test interpreter additionally receives the exact coverage observer from the quality lock;
that observer is not added to the original calculation environment. A separate actual NumPy 2.4.6
installation exercises refusal by the forward entrypoint, whose qualified version is 2.3.5.
The dependencies permit 2.4.6, so its dependency check is expected to pass; this probe tests the
entrypoint's declared-version refusal rather than a dependency conflict or general functional incompatibility.
The two production locks and the wrong-version probe lock are audited with blocking strict checks.
The original submitted-inference and training locks remain unchanged.

Tests run complete external inverse/forward calculations, reload their generated maps and currents,
and check provenance and numerical-runtime refusals. Fault cases execute the genuine solver before
corrupting one returned current value; they check the export guard, not solver accuracy.
Analytic tests check signed current/pressure normalization, discrete/continuous source behavior,
contour availability and observation tolerances. Nine public boundary-operator integrations compare
three grids and three engines with an independent continuous-source Green integral.
The small original coarse-map bundle also has a focused checksum/arithmetic test.
Its maintained figure command exports PNG, SVG and exact plotted quantities. Dedicated tests recompute
the unchanged whole-grid gauge/span and reference-core/edge measures, render repeatable actual images,
and refuse substituted originals or an existing output directory. The figure does not change the failed gate.

The following skips are explicit in a local environment that has not provisioned both interpreters:

| Missing setting | Checks unavailable locally | Where executed |
|---|---|---|
| `CONSTRUCTIVE_REFERENCE_PYTHON` | Complete inverse/export, two post-solve current faults, changed-source refusal in the correct runtime, three legacy boundary grids | Dedicated constructive integration job |
| `CONSTRUCTIVE_WRONG_VERSION_PYTHON` | Actual unqualified-NumPy refusal | Dedicated constructive integration job |

The dedicated job provisions and asserts both interpreter paths before running the tests; missing
provisioning fails the job. A fast subset or named local skip is not reported as complete integration
coverage. Timed-out verification attempts remain distinct from completed checks and physical failures.

The scientific search itself is inconclusive: the fixed coarse solver-pair compatibility gate failed.
Passing software controls does not admit an ambiguity witness or prove a real-DIII-D information limit.
[Model, numerical gates, stored results and reproduction](docs/CONSTRUCTIVE_PAIRS.md) state these limits.

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
Git source object, audits each of the six dependency environments and retains CycloneDX inventories. It uses
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
