---
title: Physics audit
layout: default
nav_order: 2
---

# A higher score does not certify force balance

On 358 DIII-D development shots, the map-finishing step raised the benchmark score by 1981 µS while increasing
several measured field, current and Grad–Shafranov (GS) form diagnostics. A subsequent interior projection
reduced the tested GS-form residuals but lowered the score by 157 µS and increased an interface-current proxy.
Here µS means 10⁻⁶ of a benchmark score point, not an electrical unit.

These are measurements of specific numerical properties, not a declaration that either reconstruction satisfies
force balance. The public tools let other researchers inspect their own maps and recompute our published
tables without the trained ensemble.

## What the finishing step changed

The evaluated networks were fitted without these 358 shots, containing 78,875 frames. The shots had already
been used in model comparisons; they are not untouched confirmation data. The corrected measurements compare
the reference and nine map stages using their actual common finite frames. Means give equal weight to shots.

| Corrected diagnostic, minus its EFIT reference value | Before finishing | After finishing | Change, percentage points (99% interval) |
|---|---:|---:|---|
| Exterior current-density ratio on a common region | 0.70% | 1.23% | +0.53 (+0.48 to +0.58) |
| Flux-surface GS-form residual/source | 1.33% | 5.14% | +3.81 (+3.44 to +4.21) |
| Polynomial GS-form RMS residual/source | 2.43% | 5.94% | +3.51 (+3.15 to +3.91) |

These percentages are diagnostic ratios and reference-subtracted ratios. They are not percentages of plasma
or current outside equilibrium. The exterior region is fixed across all ten maps for each frame; GS fits use
each map's own interior or surfaces. Most of the surface-residual increase occurs in the contour-editing stage.
The later support census found all four requested surfaces usable on these historical float64 maps. That does
not establish adequate conditioning or support on other maps.

See [the corrected study](STUDY.md#427-what-the-finishing-step-does-to-the-physics-of-the-map),
[measurement definitions and report](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/d71b_physics_amendment)
and [surface-support census](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/d71b_surface_support).
The [original experiment](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/d71_finisher_physics)
retains its historical results with the faulty critical-point counts withdrawn. Detected critical points of an
interpolant are not guaranteed magnetic axes or boundary X-points.

## What the interior projection changed

The projection retained all 78,875 frames, including 3,435 that did not meet its stopping criterion within
40 iterations. Its original score change was −157 µS, with a 99% whole-shot interval of −239 to −93 µS.
It failed the admission rule and was not submitted.

The following remeasurement uses the exact float32 maps supplied to the scorer. These are **raw** diagnostic
ratios, not the reference-subtracted values above. Regions also differ from the ten-map stage comparison.

| Diagnostic ratio | EFIT reference | Finished map | Projected map |
|---|---:|---:|---:|
| Flux-surface GS-form residual/source | 0.20244% | 5.34075% | 0.07149% |
| Poloidal-field error on the reference interior | 0% | 2.97213% | 2.83469% |
| Interface-crossing current proxy/reference-inside sum | 3.05668% | 3.53878% | 4.20469% |

Lower GS-form fit residuals test the fitted source structure. They do not independently establish correct
pressure/current profiles. The crossing-stencil proxy does not itself demonstrate a physical current sheet.
An analysis of scorer residuals attributes approximately 64% of the score loss to axis-height consistency;
this does not establish that pinning the axis would recover that score.

[Projection and pooled score](STUDY.md#428-a-gradshafranov-interior-for-the-finished-map),
[scored-map diagnostics and convergence subsets](STUDY.md#429-physical-diagnostics-on-the-scored-projection-maps)
and [full diagnostic table](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/d72_scored_map_physics)
retain the intervals, support and definitions.

## Recompute the published tables

From the repository root, use the hash-pinned environment in [the audit guide](AUDIT_TOOL.md#run-an-example).
The three commands below check the public inputs and reproduce aggregate arithmetic and whole-shot intervals:

```bash
.venv/bin/python tools/d71b_table.py \
  --table experiments/d71b_physics_amendment/per_shot_v1.csv.gz \
  --manifest experiments/d71b_physics_amendment/per_shot_manifest_v1.json \
  --report experiments/d71b_physics_amendment/report_v5.json

.venv/bin/python tools/d72_table.py \
  --data experiments/d72_gs_projection/pooled_statistics_v1.json.gz \
  --manifest experiments/d72_gs_projection/pooled_manifest_v1.json \
  --gate experiments/d72_gs_projection/gate_v1.json

.venv/bin/python -m tools.scored_map_table \
  --table experiments/d72_scored_map_physics/per_shot_v1.json.gz \
  --manifest experiments/d72_scored_map_physics/manifest_v1.json \
  --manifest-sha256 4312cad1b96dd30484305849603f66bfbf475f986fe8d388238116c2df16fea6 \
  --report experiments/d72_scored_map_physics/report_v1.json
```

The stage statistics retain paired means on common frames; separate map means cannot replace them. Pooled
score reproduction resamples reference moments and residual sums, rather than averaging per-shot scores.
The intervals are conditional on this repeatedly examined cohort, not independent confirmation or simultaneous
coverage for all comparisons.

These commands do not regenerate private maps, independently authenticate the original measurements or
reconstruct frame percentiles. Full stage/map regeneration still needs the private evaluation models and caches.
See [validation scope](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/VALIDATION.md).

## Audit another model

[Audit your maps](AUDIT_TOOL.md) accepts NPZ flux maps with explicit coordinate units, flux convention and
axis orientation. A reference and measured current are optional. Reports include support, rank, conditioning,
degeneracy and unavailable reasons. A versioned fixed comparison region can be reused across models so that
adding a model does not revise earlier numbers. The command does not infer an axis or boundary, compute the
competition score, or certify force balance.

[Calibrate the audit](AUDIT_CALIBRATION.md) runs eight analytic controls and twelve illustrative maps: EFIT,
this entry and the organisers' independently authored PCA–Ridge method on four frames. The baseline fit excludes
the diagnostic cohort. Those frames were selected by this entry's boundary errors, so this is a reference-assisted
illustration, not a population comparison or an outside replication. The analytic controls expose discretisation
error, controlled perturbations and unavailable diagnostics before experimental maps are interpreted.
