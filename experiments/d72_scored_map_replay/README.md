<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge — original scored-field replay
-->
# Remeasure the scored fields

This tool measures existing fields, without loading the trained ensemble. It uses the corrected historical
operators and the same regions as [the D72 physical comparison](../d72_scored_map_physics/README.md).
It does not rerun the competition scorer, fit a model or certify force balance.

The included example contains **three frames**: frame index zero of original cohort indices 0, 179 and 357,
with identifiers `d3d_shot_0093deb434`, `d3d_shot_862555f165` and `d3d_shot_fedaca60be`.
Selection depends on identifiers and frame order, not diagnostic values. Their parent shots contain
213, 293 and 217 frames. This example exercises the measurement path; it does not reproduce a
358-shot or 78,875-frame population claim.

## Run the example

From the repository root, install `requirements-quality.txt` with Python 3.12, as in
[the audit guide](../../docs/AUDIT_TOOL.md#run-an-example). Choose an output directory that does not exist:

```sh
python -m tools.scored_map_replay \
  --bundle experiments/d72_scored_map_replay/illustration_v1 \
  --manifest-sha256 1e9829c8442ca552665bf9ea1191fa456be292b5fbaf1be2c48536bb2c7bea17 \
  --expected-statistics experiments/d72_scored_map_replay/illustration_v1/expected_statistics.json.gz \
  --expected-statistics-sha256 01e8941eb61d8a4c8e6c970f2d4e371613d987c891fb8011b2f57cd92827bb18 \
  --out scored-map-example
```

The command defaults to serial measurement; `--workers N` uses up to twelve isolated processes and
retains the original shot order. Select N within your machine's available resource budget.

The command reads the actual float32 arrays, computes all 21 raw diagnostics, derives the normalised
statistics and compares them to the independently retained original frame measurements. Counts and
availability must match exactly. Numerical comparisons retain the published absolute tolerance 10⁻¹³
plus relative tolerance 10⁻¹². Small differences across numerical environments are possible.
`original_frame_diagnostics.npz` retains these original frame values for direct numerical controls.

Outputs include:

- per-shot NPZ files with measured frame values, the reference-current denominator, used surfaces,
  point counts, rank, condition, singular values and unavailable reasons;
- `per_shot.json.gz` and `report.json`, preserving actual pairwise common finite frames and whole-shot
  bootstrap weights;
- `surface_per_shot.json.gz` and `surface_report.json`, adding four-level support and sensitivity to
  requiring common full support, including convergence strata;
- `complete.json`, with scope, hashes of generated artifacts and actual measuring sources, numerical-library versions
  and comparison discrepancies.

The bootstrap on three frames in three shots is an illustration of the arithmetic, not a population
uncertainty estimate. Full-support sensitivity supplements the all-valid primary measurement; it never
replaces it. An invalid input, failed comparison or existing output directory returns exit 2 with a
fixed refusal message. A partial output directory is not a successful measurement.

## Input contract

`manifest.json` declares ordered shots, exact file hashes, explicit sign and the grid. Arrays are
standalone NPY files read without pickle, with read-only memory mapping. Grid coordinates `R.npy` and
`Z.npy` are increasing, uniform float64 metre coordinates of length 65; `envelope.npy` is the original
65×65 Boolean scorer envelope. Array orientation is `[frame, Z, R]`.

Each shot directory contains:

| Files | Dtype and shape | Meaning |
|---|---|---|
| `reference.npy`, `base.npy`, `gs.npy` | float32, `(frames, 65, 65)` | Released EFIT, original finished prediction, original projected prediction |
| `contours.npy`, `offsets.npy` | float64 `(points, 2)`, int64 `(3×frames+1,)` | Original ragged `(R,Z)` contours in frame-major reference/base/gs order; a zero-length candidate contour means absent geometry |
| `times_ms.npy`, `source_indices.npy` | float64 and int64, `(frames,)` | Original times and source-frame identities |
| `used.npy`, `converged.npy` | Boolean, `(frames,)` | Original replacement and pre-float32 stopping flags |
| `iterations.npy`, `pixels.npy` | int64, `(frames,)` | Original iteration and replacement counts |
| `scalars.npy` | float64, `(frames, 2)` | Unchanged submitted float32 q95 and betaN values promoted exactly to float64; not used to infer physics diagnostics |

Flux values are unchanged released `psirz`, declared as `released-psirz-unscaled`; no factor of 2π or
absolute flux-unit conversion is inferred. The explicit `axis_sign` orients `φ = axis_sign × ψ` with an
axis maximum. These diagnostic ratios preserve the original convention. The general
[external-map audit](../../docs/AUDIT_TOOL.md) has a separate explicit absolute-unit contract.

File hashes bind NPY bytes; `arrays_sha256` additionally binds the selected arrays in C-order value bytes.
The original full-shot D70/D72 array identities remain separate. For whole shots the selected base and
projection identities must equal their original full-shot identities. For the three-frame illustration
they identify the selected slices, not the complete parent arrays. Hashes identify supplied data and
lineage; they do not independently authenticate the original model run.

Reference contours and candidate contours remain explicit inputs. The tool does not silently replace
missing geometry, extract a new boundary or change domains. The shared exterior uses the original
three-contour union. Surface fits retain the original marching-squares vertices, including the closing
point, and do not reject by rank or condition. Detected critical points describe an interpolant.

## Qualification and larger populations

On the build host, the public numerical kernel was qualified against all original measurements of
these same three shots: 723 frames, with exact agreement for all raw diagnostics and the reference-current
denominator. Recomputed subset statistics and intervals also agreed exactly. These are already-examined
development shots, not untouched confirmation or independent external replication.

`qualification_manifest_v1.json` records the actual whole-shot input identities and frame counts.
Its arrays are not included with the small illustration. Substituting this metadata for the illustration
manifest does not create a complete whole-shot package. Only a corresponding full input package can
remeasure the full cohort. The original complete aggregate table remains independently reproducible with
[its existing verifier](../d72_scored_map_physics/README.md#reproduce-the-published-diagnostic-table).

## Attribution

The three released reference maps and timestamps originate in the Sophelio Fusion Equilibrium Challenge
DIII-D training data, CC BY 4.0. They are an excerpt, promoted to the original scorer's float32 dtype.
The contours include extracted reference geometry and this entry's predicted geometry. Grid and envelope
originate in the MIT-licensed official starter. Predictions and measurement code are this entry's work.
See [NOTICE](../../NOTICE.md) and the per-path licensing metadata in `REUSE.toml`.
