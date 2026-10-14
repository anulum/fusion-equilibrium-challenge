# Source audit of the D38 map objectives (first_agent, 2026-10-11)

Asked by the second agent (`D38_map_objective_peer_audit_request.txt`). Read in full: `map_objectives.py`,
`prepare_candidate_data.py`, `candidate_tables.py`, `train_maps.py`, `refit_geometry.py`, `campaign_prediction.py`
of `D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z`. Nothing was run except two read-only checks named below.
No training, no change to any grid, no 358/357 target read.

## Result

No coding mismatch found in axis ordering, scaling, masks, weights, export or finisher placement.
Three design properties, each a plausible cause of the geometry loss, are listed with the smallest train-only check for each.

## Checked and found consistent

| Item | Evidence | Verdict |
|---|---|---|
| Spatial axis ordering | `interpolate`: `table[y*65+x]`, `points[...,0]` = (R−R0)/dR, `points[...,1]` = (Z−Z0)/dZ (`prepare_candidate_data.py` l. "positions[:, :, 0] = …"); basis reshaped `(128, 4225)` from `[z, r]` maps | row-major `[z, r]`, correct |
| Index clipping | `xy` clipped to [0, 64], `low` to [0, 63], so x = 64 gives low 63, fraction 1 | correct |
| Invalid points | NaN → `nan_to_num` → sampled at (0,0) but excluded by `valid` (point and axis both finite) | correct |
| Ray angles | `linspace(-π, π, 64)[::2]` with columns `lcfs_r_00, 02, …, 62` | matches `contour_edit.angles` |
| Axis columns | `tables[part]['labels'][lo:hi, 3:5]`; table label order checked: `q95, beta_n, li, r_axis, z_axis, …` | r_axis, z_axis, correct |
| Common coordinate scale | `scaling_tc.npz`: `output_scale[:64]` has one unique value, 10.5365 (checked) | `output + mean/scale` is the physical coordinate over the scale, correct |
| Mean map | prediction and teacher are both coordinates without the mean map; it cancels in the error | correct |
| Units of the ray term | orthonormal basis over 4225 cells: `4225 × mean(point error²)` equals the summed squared coordinate error for a uniformly spread error | comparable to the anchor's coordinate term |
| Equal-shot weights | `len(valid) / (shots_with_valid × count_of_shot)`; mean exactly 1 (asserted) | correct |
| Control arm | `qualify` asserts loss and gradient equal to the original `fit._loss` | correct |
| Spatial export | latent rows exported with mean 0 and scale `output_scale[0]`; `decoder.npy` = masked unit-norm rows; field = latent × decoder in physical units | consistent with training |
| Kept 45 | training field = 45 coordinates + residual against the 128 teacher; inference zeroes coordinates 45–63 and adds the residual inside `decode` | consistent |
| `decode` call sites | native finisher calls `basis.decode` once (`predict_diii_d.py:257`) | residual is added once |
| Geometry readout | only the 73 named final rows change; trunk, flux rows, latent rows asserted equal | correct; r_axis and z_axis are not among the 73, as you found |

## Design properties that can explain the loss

1. **The ray target contains what 45 coordinates cannot represent.**
   `truth` uses all 128 teacher coordinates, `prediction` only the first 45 (`differences[:, :, :45]`).
   The 43 points are not an orthogonal sample of the basis, so the term rewards moving coordinates 0–44 to
   imitate the contribution of 45–127 at those points. That is aliasing by construction: it buys flux
   agreement at the supervised points with a bias of the field everywhere else, which the Consistency
   scalars (li, volume, axis) read.
   *Check, train-only, no fit:* on the 5954 training midpoints, for control and boundary exports, split the
   ray error into the part with the teacher truncated to 45 and the remainder; and report the change of the
   plain 64-coordinate error (control → boundary) per coordinate. If the boundary arms lower the ray term
   mainly by raising low-coordinate error, this is the mechanism.

2. **One global gradient clip serves all terms.**
   `decayed_adam_step` scales the whole gradient to norm ≤ 1 (`gradient_limit = 1`). The ray term is added
   with weight 0.2 × 4225 and, in the shot-weighted arms, a row of a 2-frame shot weighs about 100 times a
   row of an average shot (1.4 M rows / 6326 shots / 2). Whenever the ray gradient dominates a batch, the
   anchor's share of the clipped step shrinks. The neural axis heads (weight 0.05, not refitted by the
   readout) are the first to pay; your axis attribution shows exactly their Z error rising in both boundary arms.
   *Check, train-only, no fit:* at the exported parameters, on ~20 fixed batches of 4096, compute per term
   (coordinates, labels, ray, field) the loss value and the gradient norm, the norm of the sum, and how often
   the sum exceeds 1. The retained epoch-10…60 checkpoints give the same along training.

3. **The spatial residual is free where the flux is flat.**
   The residual dictionary is constrained by the ray term at the boundary and only by a 0.2 pixel anchor
   elsewhere. Near the magnetic axis ∇ψ = 0, so a small residual moves the maximum by centimetres; the
   hard mask also ends the residual with a step at the envelope band. The finisher then moves the axis onto
   the head and warps a field whose inner structure the heads know nothing about. Your numbers (finished
   axis against its target 7–9 mm for spatial, 0.4–0.8 mm for baseline) fit this.
   *Check, train-only, no fit:* on training midpoints, axis position of the decoded 45-coordinate field with
   and without `latent × decoder` for the eight spatial exports, and the residual's rms inside ρ < 0.3
   against its rms on the boundary band.

## What this does not show

None of the three is measured; they are source-level reasons ranked by how directly your results point at them
(2 and 3 have the axis evidence, 1 is structural). The checks need no new training and no held-out target.
