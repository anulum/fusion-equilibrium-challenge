<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge — scored-map physical diagnostics
-->
# Physical diagnostics on the scored projection maps

This supplement measures the existing [interior projection](../d72_gs_projection/README.md) on the exact
float32 arrays supplied to the scorer. It compares EFIT (`reference`), the finished held-out evaluation
predictor of Study §4.26 (`base`) and its projection (`gs`). The predictor's 69 map, 8 joint and 8 static
networks were fitted without these 358 development shots; the shots had previously been used in model
comparisons. This is a diagnostic remeasurement of an existing negative experiment. The submitted model,
weights, scalar predictions and original score gate are unchanged.

## Whole-cohort result

All 358 shots and 78,875 frames remain included. Values below are equal-shot means of diagnostic ratios,
multiplied by 100 for display. Changes and their conditional 99% whole-shot bootstrap intervals are in
**percentage points**. These are raw ratios, not excesses above the reference and not percentages of
plasma or current violating equilibrium. Every displayed pair has 358 supported shots and 78,875 common
finite frames; the machine-readable report retains support separately for every metric and stratum.

| Diagnostic ratio | EFIT reference | Base | Projection | Base → projection (99% interval) |
|---|---:|---:|---:|---|
| Flux-surface GS-form residual | 0.20244% | 5.34075% | 0.07149% | −5.26925 points (−5.76319 to −4.84019) |
| Polynomial GS-form RMS residual/source | 0.34739% | 6.28677% | 0.03625% | −6.25052 points (−6.74111 to −5.82095) |
| Poloidal-field error on the reference interior | 0% | 2.97213% | 2.83469% | −0.13744 points (−0.18180 to −0.10523) |
| Poloidal-field error on the eroded envelope | 0% | 2.61383% | 2.50243% | −0.11140 points (−0.15673 to −0.08193) |
| Absolute current proxy on interface-crossing stencils/reference-inside sum | 3.05668% | 3.53878% | 4.20469% | +0.66591 points (+0.59120 to +0.79647) |
| Absolute current proxy on the shared exterior/reference-inside sum | 0.08345% | 1.35007% | 1.35007% | 0 points (0 to 0) |

The projection substantially reduces the tested GS-form fit residuals and modestly reduces poloidal-field
error against EFIT, while increasing the interface-crossing current proxy. The original benchmark change
remains −157 µS (99% interval −239 to −93), with no candidate submitted. This measures a specific tradeoff
between these diagnostics and the benchmark, not a general improvement in physical reconstruction.

## Definitions and limits

Base and projected maps are converted to the scorer's float32 dtype before diagnostic measurement, then
promoted to float64 for arithmetic. EFIT reference arrays are unchanged. The corrected D71b operators
are used, with `φ = axis_sign × ψ` so that the magnetic axis is a maximum. These arrays differ from the
historical float64 stage snapshots in D71b.

The flux-surface diagnostic fits `a R² + b` to `−Δ*φ` at requested normalised flux levels 0.2, 0.4, 0.6 and
0.8. It can return a finite estimate with fewer than all four usable levels. The separate historical
surface-support census does not establish four-level support on these float32 maps. Both GS tests use
each map's own extracted interior and flux; their spatial domains can move. Common finite frames do not
mean common surfaces. Cubic source functions were fitted to the base map itself: lower fit residuals
test that structural property, without independently establishing correct source profiles or force balance.
The report also includes polynomial SSE, TSS, rank, pixel counts, RMS terms and degeneracy flags.

For each frame the shared exterior is outside the one-cell-dilated union of the **three** contours,
restricted to the eroded scorer envelope. All maps use the same exterior and reference-inside current
denominator. This is a different region from D71b's ten-map union; the exterior ratios cannot be directly
equated. Poloidal-field errors compare first derivatives divided by R against EFIT on its interior or
the eroded envelope, reported as the square root of squared vector-field error divided by reference field energy.

Five-point derivative stencils are classified using the base contour's replaced interior: fully inside,
crossing the replacement interface, or wholly outside. Grid-edge nodes are excluded. Each class sums
`abs(Δ*φ)/R`; its normalised version divides by the reference-inside sum. Class membership is fixed
between maps. Stencil classes span all non-edge grid nodes and differ from the eroded shared-exterior region.
Exterior derivative equality is checked separately. An increased crossing-stencil proxy
does not by itself demonstrate a physical current sheet. Critical-point counts describe the interpolant
and have no universal better/worse ordering. None of the intervals for the overall base-to-projection
changes in detected maxima, minima or saddles excludes zero; that does not establish unchanged topology.

## Convergence subsets

The original float64 Picard stopping flag, recorded **before** float32 conversion, partitions 75,440
converged frames in 355 supported shots and 3,435 nonconverged frames in 185 shots. No frame was unused.
A shot may contribute to both subsets. These subsets explain the measurement and never replace the
whole-cohort result. A convergence flag is not a physical certificate of the quantised output.

| Base → projection, percentage points | Converged (99% interval) | Nonconverged (99% interval) |
|---|---|---|
| Flux-surface GS-form residual | −4.87482 (−5.24708 to −4.54506) | −13.45544 (−15.12374 to −11.82454) |
| Polynomial RMS/source | −5.90291 (−6.29867 to −5.55030) | −13.41867 (−14.98170 to −11.92421) |
| Poloidal-field error, reference interior | −0.10863 (−0.13018 to −0.08972) | −0.48140 (−0.65287 to −0.34232) |
| Crossing-stencil current proxy/reference-inside sum | +0.59959 (+0.55957 to +0.64249) | +1.55376 (+1.25956 to +1.90870) |

A separate analysis of retained frame-level scorer residuals attributes 100.515 µS, approximately 64% of
the net score loss, to axis-height consistency. It reproduces the original residual sums and aligns all 78,875 times
and projection flags with this measurement. It does not independently regenerate maps or establish a
causal mechanism. In particular, pinning the axis would not necessarily recover that score. The score
decomposition uses pooled residuals; these physical diagnostic means give equal weight to supported shots.

## Reproduce the published diagnostic table

From the repository root, install `requirements-quality.txt` in Python 3.12, then run:

```sh
python -m tools.scored_map_table \
  --table experiments/d72_scored_map_physics/per_shot_v1.json.gz \
  --manifest experiments/d72_scored_map_physics/manifest_v1.json \
  --manifest-sha256 4312cad1b96dd30484305849603f66bfbf475f986fe8d388238116c2df16fea6 \
  --report experiments/d72_scored_map_physics/report_v1.json
```

The compressed table records the original ordered shot identities, per-map finite-frame means and
counts, actual common-frame paired means and changes, and convergence-stratum support. Each supported
shot receives equal weight. The consumer reproduces all diagnostic means and all 20,000 original-cohort
multinomial draws with seed 23023. Unavailable shot estimates are dropped and their weights renormalised;
entirely unavailable draws are counted rather than redrawn. The empty `unused` stratum has no estimate.
Intervals describe this repeatedly examined cohort, not untouched confirmation or simultaneous familywise
coverage. Independent map means are never substituted for common-frame paired changes.

The caller-pinned manifest binds cohort order, counts, method identities and table/report bytes. Verification
uses absolute tolerance 10⁻¹³ plus relative tolerance 10⁻¹²; counts and availability remain exact. This
reproduces aggregate arithmetic and intervals. It does not remeasure private maps, reproduce frame
quantiles, authenticate original private measurements or certify force balance. Neutral source labels
and SHA-256 hashes retain provenance without operational filesystem paths. The private measurement/export
controllers and raw maps are not redistributed; their identities do not independently authenticate the run.
