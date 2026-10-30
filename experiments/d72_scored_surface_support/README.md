<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge — scored float32 surface support
-->
# Surface support on the scored maps

The [scored-field replay](../d72_scored_map_replay/README.md) measures surface support on the exact
float32 reference, base and projected fields. This is separate from the historical float64
[ten-map census](../d71b_surface_support/README.md).

For each requested normalised level 0.2, 0.4, 0.6 and 0.8, it records inclusion, contour and finite point
counts, rank, singular values, condition and an explicit unavailable reason. Full support means all
four levels entered the original fit. It does not require full rank, good conditioning or a nonzero
source, and does not establish force balance. The original GS ratio is checked against the unchanged
historical implementation before any support statistics are reported.

The all-valid common-frame comparison remains primary. A separate sensitivity calculation requires
both members of each pair to have all four included surfaces and a finite ratio. Its means use exactly
those common frames. Reference/base, reference/projection and base/projection retain independent
supports. Original convergence strata are reported separately and never used to select the main result.
Bootstrap intervals use 20,000 whole-shot multinomial draws with seed 23023; unavailable shot means are
removed and the fixed weights renormalised, with unavailable draws counted rather than redrawn.

Run the replay example to obtain `surface_per_shot.json.gz` and `surface_report.json`. The statistics
contain the information needed to reconstruct the support counts, fit extrema, paired means and
intervals. They do not reconstruct pooled frame quantiles or remeasure fields themselves. The
`tools.scored_surface_summary` command requires the compressed table's independently supplied SHA256
and compares its recomputed report to the supplied report. Invalid input returns exit 2.

The public example contains only three predetermined frames in three already-examined shots.
Its support and bootstrap output is illustrative. It is not the full scored-map support census.
