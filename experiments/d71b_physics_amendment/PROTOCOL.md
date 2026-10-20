# D71b — corrected and added physical measures after the second external review (frozen before any D71b result)

Entrant's word, 2026-10-20: "už nič nepushneme kým to po sebe neskontroluješ" with the review of commit 88e31fd.
Measurement only, the same 358 shots and nine regenerated stages as D71 (D70 sources and activation unchanged,
final stage checked against D70's field checksums). Self-reviewed by the first agent; not independent.

## What the review found and what was checked before this protocol
- D71's critical-point counts are unreliable: one smooth maximum between nodes counted 4 times; an axis-aligned
  saddle between nodes missed; a saddle on a node counted as 4 cells. Reproduced exactly (test of 20 October).
  D71's topology rows are withdrawn; D71b replaces them.
- D71's GS fraction is undefined for a nearly constant source (returns ~1 for an exact field). Census on the D71
  frames: 0 of 78 875 reference frames above 0.5, 7 frames at any stage; excluding them changes the finisher's
  shot-mean change from +1.318 to +1.314 points. Its name was too strong: it is a normalized polynomial GS-form
  residual. D71b adds its parts and a degenerate flag.
- D71's outside-current region moves with each map's own boundary. D71b adds a region outside both boundaries.

## Measures (physics_b.py; critical.py; D71's physics.py imported unchanged)
1. Critical points: roots of the bilinearly interpolated central-difference gradient in each cell (closed form),
   classified by the Hessian at the root, merged within half a cell; inside the scorer's envelope eroded by two
   cells. Per frame, against the reference: more maxima, fewer maxima, more saddles, fewer saddles, more minima
   (each a 0/1 frame flag). Tests (test_critical.py, run before freezing, all pass): the review's four fields,
   minima, rotations by 10, 30, 45, 73 degrees, shifts, grids of 33, 65 and 129, two maxima with a saddle, Solov'ev.
2. outside_fixed_excess: sum |Delta* phi| / R over the eroded envelope outside both LCFS (dilated by one cell),
   over the same sum inside the reference LCFS; minus the reference's value.
3. gs_rms_norm_excess: RMS residual of the cubic polynomial GS-form fit over the RMS source, minus the
   reference's; plus SSE, TSS, pixels, rank and gs_degenerate (TSS below 1e-10 of the sum of squares).
4. gs_surface_excess: on psi_N = 0.2, 0.4, 0.6, 0.8 of the map, -Delta* phi fitted as a R^2 + b per surface;
   residual RMS over source RMS; minus the reference's. Tests (test_physics_b.py): Solov'ev 1.4e-5, warped 0.107,
   constant source flagged degenerate.

## Report (report_d71b.py)
As D71: shot means over frames with a reference contour, paired whole-shot bootstrap, 20 000 draws, seed 71020,
99 % interval; smoothed -> margin and consecutive stages; lower is closer to the reference.

## Amendment 1 (2026-10-20, before any D71b result)
The first full run stopped before completing any shot (full358_v1 empty): critical.py required grid steps equal
to 1e-9, and the scorer's grid is stored from float32 (step spread 9e-6 of the step in R, 2.4e-6 in Z). The
tolerance is now 1e-4 of the step; test_critical.py runs its Solov'ev case on a float32-rounded grid. Nothing else
changed. Outputs go to full358_v2/ and report_v2.json.

## Amendment 2 (2026-10-20, after the second agent's review REVIEW.md 3b1c7b97; before any D71b result)
The v2 full run (amendment 1) was stopped by the first agent after one completed shot, before any report; no D71b
number was read. The second agent reproduced four faults, all accepted:
- Outside current compared different regions for prediction and reference (a false excess of -1.46 for one field
  with two boundaries). Now ONE region per frame: the eroded envelope outside the union of the reference LCFS and
  the LCFS of all nine stages, dilated by one cell, used for every map and for the reference, with one denominator
  per frame (the reference's current density inside its own LCFS). Test: the review's case gives excess 0 exactly.
- Critical points: v1 symmetrized the Jacobian of two independently interpolated gradient components. Now one
  scalar interpolant (interpolating bicubic spline), candidates on a 4-times finer grid of its gradient (both
  components changing sign, or one changing and the other within 0.1 of zero), Newton on the spline gradient, its
  own Hessian for the class, cells within 2 of the grid edge excluded, roots with a Hessian below 1e-6 of the grid's
  largest discarded. Tests: all earlier cases, and the review's cubic field: no root in its cell (v1 reported a
  saddle there), and all five exact roots of that field found at their positions with their exact class.
- A zero-energy GS source is now flagged degenerate.
- The report is fail-closed (complete 358 / 78 875 receipt, every per-shot measure hash) and compares two maps
  only on the frames where both are finite (matched support, counts reported).
Qualification on three real shots (indices 0, 179, 357, the scorer's grid) before the full run; the second agent
re-reviews the sources and the qualification before the full run starts.

## Amendment 3 (2026-10-20, TODO FEC-R18 from the second agent's re-review; before any D71b result)
The second agent accepted the measurement repairs and the three-shot qualification of amendment 2 and withheld the
full run: the report trusted the receipt's own fields, and the runner returned cached shot records before any check.
receipts.py now binds every shot record to the activation's SHA-256, its stem, frame count and the frozen map order,
and checks its measures file hash, names and (frames, maps) shapes; validate_run accepts only a complete run of
exactly the D70 activation's cases in order. The runner writes these bindings and, on resume, reuses a cached shot
only through validate_shot. The report takes the activation and the D70 cases file (bound by checksum), requires
358 shots, and rejects qualification receipts. test_receipts.py exercises the real report CLI on a three-shot
cohort built from real qualification measures: the intact cohort passes; another activation, an unbound cases file,
a qualification receipt, a missing or extra shot, substituted members, changed bytes, a stale cached record, a wrong
frame count, shape, map order, a missing measure and a cohort of the wrong size are each rejected by ReceiptError.
Matched finite-frame support for stage comparisons is unchanged.

## Amendment 4 (2026-10-20, the second agent's FEC-R18 v4 review; before any D71b result)
The second agent reproduced through the actual run_d71b.one() that its cached branch returned before the four
case-dependency checks (raw, features, reference, D39 cache), so a cached shot was accepted with a changed
dependency. The four checks now run first for every shot, cached or not. test_runner_cache.py calls the actual
one() on a copied real cached shot: intact accepted; each of the four dependencies changed (expected SHA kept) and a
stale activation rejected. The measurement algorithms are unchanged. The qualification is run again under v5 (a new
numerical run of the three shots, not relabelled v4 outputs), then the cache regression test runs on its records.
