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
missing or zero sources and rank loss. The full 358-shot support census has not been run, so these checks do not
establish how partial support affects the published cohort result. This statistic is a diagnostic of the GS form,
not a certification of physical force balance. Original D71b results and source are unchanged.
