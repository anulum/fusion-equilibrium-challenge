<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge — surface-GS fit support
-->
# Surface-GS fit support

[`tools/surface_gs.py`](../../tools/surface_gs.py) exposes the support of the
[D71b surface statistic](../d71b_physics_amendment/README.md). A finite historical ratio can use fewer than the
four requested levels. This helper records that distinction without modifying the frozen experiment.

Install the pinned quality environment as in [CONTRIBUTING.md](../../CONTRIBUTING.md). From the repository root:

```python
import json
import numpy as np
from tools.surface_gs import measure

radius = np.linspace(0.84, 2.54, 65)
height = np.linspace(-1.6, 1.6, 65)
r, z = np.meshgrid(radius, height)
phi = -(r**2 - 1.7**2)**2 / 8 - 0.9*z**2
laplacian_star = -r**2 - 1.8
result = measure(phi, laplacian_star, radius, height,
                 phi >= -0.3, float(phi.max()), -0.3)
print(json.dumps(result, indent=2, allow_nan=False))
```

This analytic Solov'ev field has a source affine in R². The measured ratio includes contour interpolation error;
it need not equal zero on the sampled grid. Coordinates are R and Z in metres; arrays have shape `(len(Z), len(R))`.
The flux must be oriented with the magnetic axis a maximum. Supply its Delta-star in consistent units, the
pixel mask inside its own boundary, the axis value and the mean boundary value. Missing source stencil values
may be NaN. The helper does not extract these inputs from arbitrary files or infer their conventions.

The `surface-gs-support.v1` result contains:

- Requested fractional flux levels 0.2, 0.4, 0.6 and 0.8, and the subset used.
- Per-level selected contour vertices, finite source samples, fit rank, singular values and condition number.
- Inclusion and degeneracy flags, reasons for unavailable support, and residual/source energies.
- The historical root residual/source-energy ratio, or null when no usable ratio exists.

The longest closed contour with more than 12 vertices is selected at each level. A fit requires at least 12
finite samples. Marching-squares vertices, including the repeated closing point, retain their original weights.
Rank-deficient fits remain included, as in D71b; rank and conditioning are diagnostic metadata, not new filters.
An unrepresentable condition number is null. `full_support` means all four fits were included; it does not mean
that their rank, conditioning or source strength is adequate. A zero source can therefore have full contour
support and an unavailable ratio. No arclength resampling is performed.

The ratio agrees exactly with the frozen implementation on the checked host for 12 published illustration maps
and 300 synthetic perturbed elliptical maps. Dedicated analytic tests cover deformation, partial contours,
missing or zero sources and rank loss. This statistic is a diagnostic of the GS form, not a certification of
physical force balance. Original D71b results and source are unchanged.

## Complete cohort census

The census covers the original 358 shots and 78,875 frames at all ten D71b map stages. All four requested surfaces
were used on every frame in every stage. All 3,155,000 individual fits had rank two and positive source energy;
none had an unavailable condition number. Restricting the comparison to complete surface support removes no
frames. Partial support, rank loss and zero sources therefore do not explain the historical cohort result.
Finite condition numbers alone do not establish adequate conditioning or physical validity.

| Surface-GS ratio, equal weight per shot | Reference | Before finishing (`smoothed`) | After finishing (`margin`) |
|---|---:|---:|---:|
| Raw residual/source ratio | 0.2024% | 1.5346% | 5.3407% |
| Excess over the reference on common frames | 0 | 1.3322 percentage points | 5.1383 percentage points |

The paired increase from before to after finishing is **3.8061 percentage points**, with a 99% whole-shot
bootstrap interval of **[3.4380, 4.2001] percentage points**. The common-frame and complete-support comparisons
are identical. These are changes in diagnostic ratios, not percentages of plasma or current outside equilibrium.

The inputs are the historical D71b stage snapshots, evaluated in float64. This census does not measure the
scored float32 D72 projection maps. Its report preserves the original finite-reference selection and separately
shows historical per-map support and common support for the reference, `smoothed` and `margin` maps. Surface
rank is recorded rather than used as a new exclusion. The published aggregates include per-level vertex and
finite-sample counts, rank histograms and condition ranges for each shot and map.

## Recompute the published report

The compressed [per-shot statistics](per_shot_v1.json.gz), [manifest](manifest_v1.json) and
[report](report_v1.json) support an independent arithmetic check without the private maps. From the repository
root, in the pinned quality environment:

```sh
python -m tools.surface_support_table \
  --table experiments/d71b_surface_support/per_shot_v1.json.gz \
  --manifest experiments/d71b_surface_support/manifest_v1.json \
  --manifest-sha256 7afd91aea533d4b79e7db595b657fe4c8c5f0c0cef712090912c76c11fdcdd54 \
  --report experiments/d71b_surface_support/report_v1.json
```

The command checks the caller-supplied manifest checksum, ordered cohort, file checksums, counts, means and
intervals. It refuses missing, reordered or inconsistent records. It reproduces 20,000 draws of the original
358-shot cohort using seed 25025. Unsupported shots are omitted within each draw and the remaining weights
renormalised; undefined draws are counted rather than replaced. All shots support this cohort comparison.
The seed differs from the original D71b report; these intervals supplement its frozen results.

The checksum binds the supplied files to this published manifest. The arithmetic check does not authenticate
the original measurements, remeasure private maps or reproduce frame-level quantiles. The support-conditioned
comparison is a sensitivity diagnostic and does not estimate an unconditional effect on other cohorts.
