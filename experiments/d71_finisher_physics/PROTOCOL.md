# D71 — what the finisher does to the physics of the DIII-D map (frozen before the first result)

Entrant's word, 2026-10-19: "spusti fyzikálnu kontrolu finishera". Measurement only: no fit, no candidate, no change
to any submission. Self-reviewed by the first agent (no second agent available); not an independent review.

## Maps
For each of the 358 development shots (78 875 frames; exposed cohort, the evaluation copies of the incumbent are
excluded from their fits), the nine native stages of D70 are regenerated with the D70 sources and activation,
unchanged and checksum-verified: raw_64coords, kept_45coords, smoothed (the finisher's input), axis, contour,
shape, x_points, reshape, margin (the submitted map). The margin stage must equal the submitted float32 bits and
the field checksums D70 recorded for the shot, or the run stops. Reference = the released EFIT map; its LCFS is
the official reference contour. Each predicted stage's LCFS is extracted by the organisers' extract_lcfs_with_sign
with the scorer's masks, N_ITER and axis sign. Frames whose reference contour is missing are skipped.

## Measures (physics.py; identical operators and masks on every map)
phi = AXIS_SIGN * psi. Second-order central differences; Delta* phi = phi_RR - phi_R / R + phi_ZZ.
1. current_error_own: |G_map / G_ref - 1|, G = loop integral of |grad psi| / R on the map's own LCFS (the scorer's mu0 Ip).
2. current_error_ref: the same with the outward normal flux of grad phi / R through the reference LCFS (current
   inside the true boundary).
3. outside_excess: (sum |Delta* phi / R| over the eroded envelope outside the own LCFS dilated by one cell)
   / (sum inside the own LCFS), minus the reference's value.
4. gs_excess: fraction of the variance of -Delta* phi inside the own LCFS not of the form R^2 a(psi_N) + b(psi_N),
   a and b cubic, fitted per map; minus the reference's value. Profiles are fitted, never assumed.
5. bp_error_ref / bp_error_envelope: relative RMS error of the poloidal-field vector against the reference, inside
   the reference LCFS / inside the eroded envelope.
6. extra_o_points: |number of maxima of phi - reference's|; saddle_difference: |saddle cells - reference's|
   (a saddle can mark two adjacent cells; compared only against the reference counted the same way).
7. lcfs_failed: no official LCFS for the stage.

Qualification on the analytic solution phi = -(R^2 - R0^2)^2 / 8 - b Z^2 (test_physics_solovev.py, run before this
protocol was frozen): GS fraction 1.5e-25, 0.21 with a non-GS bump; area / loop 1.003; loop on own surface / signed
1.00005; critical points (1, 0, 0) and (2, 1, 2) with a second maximum. Data qualification: three shots (indices 0,
179, 357) must reproduce D70's field checksums and give reference median area / loop and loop / signed within 0.05
of 1; reported with the ratio of G_ref to mu0 * measured Ip as a convention check. Failure stops the run.

## Report and reading (report_physics.py)
Per measure, per stage: shot mean, frame median and 95th percentile. Paired whole-shot changes smoothed -> margin
(the whole finisher) and between consecutive stages; 20 000 bootstrap draws over shots, seed 71019, 99 % interval
of the mean shot-level change. Lower is closer to the reference for every measure: "worse" if the interval lies
above zero, "better" if below, otherwise inconclusive. The official score per stage is D70's (report_v2.json).
Reading limits: the 358 shots were used before; EFIT itself is a reconstruction, not truth; a 65 x 65 grid limits
derivatives, which is why every measure is reported against the reference computed the same way.

## Amendment 1 (2026-10-19, before any D71 result on data)
The v1 qualification unit was stopped by the first agent before it completed a single shot (qualification_v1 is empty); no
D71 number on data had been produced or read. On the entrant's request to do the check to the highest standard and to
add a reference / prediction / difference figure, v2 adds:
- Robustness of the Grad-Shafranov form test: the same fraction with quadratic and quintic profiles, and with a
  fourth-order Delta* operator (gs_excess_deg2, gs_excess_deg5, gs_excess_o4), reported beside the cubic,
  second-order result. Analytic check: fourth-order Delta* error 5.8e-12; all variants ~1e-25 on the exact solution.
- Figure frames, chosen by select_frames.py from D70's recorded per-frame boundary distance of the submitted stage
  (pooled over all 78 875 frames): nearest to the 50th, 90th and 99th percentile and the largest
  (figure_frames_v1.json). For those frames the reference, smoothed and margin maps and their LCFS are saved.
  The figure is labelled as development shots on which the evaluated copies of the networks were not fitted.
Outputs go to qualification_v2/ and full358_v2/.
