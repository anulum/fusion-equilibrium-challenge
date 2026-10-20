# D72 — a Grad-Shafranov interior for the submitted DIII-D maps (frozen before the first result on data)

Entrant's word, 2026-10-20: "áno, D72". One candidate, one gate. Self-reviewed by the first agent (no second agent
available); not an independent review.

## Hypothesis
D71 measured that the finishing step raises the score while the share of current density not of Grad-Shafranov
form rises from 0.35 % to 1.67 % (shot means above the reference), almost all in the contour bend. li, the weakest
derived scalar (R2 0.973), is computed from the poloidal field inside the boundary. Replacing the inside of each
finished map by a Grad-Shafranov solution with the same boundary values may lower the field error inside the plasma
and raise consistency without moving the boundary. It may also move the axis, which the axis stage placed for
+898 uS; the test decides.

## Candidate (gs_project.py), no free parameter
For each frame of the submitted pipeline's final map: the organisers' LCFS (extract_lcfs_with_sign, scorer masks,
N_ITER, axis sign); pixel centres inside it are unknowns; every other pixel keeps its value and is the Dirichlet
boundary. Delta* phi = -(R^2 a(psi_N) + b(psi_N)), a and b cubic, fitted by least squares to the map's own
-Delta* phi inside the LCFS (D71's operator); psi_N from the running axis maximum and the mean phi on the LCFS;
Picard iteration (at most 40 steps, change below 1e-9 of the flux drop), the source rescaled at every step to keep
the map's own area-integrated current; second-order 5-point Delta*; sparse LU. A frame without an LCFS, with fewer
than 30 unknowns or a non-finite solve keeps the final map. q95 and betaN unchanged. Analytic checks
(test_gs_project.py, before freezing): exact equilibrium returned within 3e-15 of the flux drop, outside unchanged;
a warped copy goes from non-GS share 0.37 to 5e-8.

## Measurement and gate
358 development shots (78 875 frames, exposed before; the evaluated copies of the networks exclude them), the D70
sources and activation unchanged; base maps must equal D70's final field checksums and the base official sufficient
statistics must equal D70's for every shot, or the run stops. Both arms scored with the organisers' scorer through
the D66 wrapper. Gate (gate_d72.py, scores() verbatim from D45): PASS iff delta S >= 1e-5, the 99 % interval of
20 000 whole-shot multinomial draws (seed 72020) lies above zero, and boundary and derived-scalar failures do not
increase. Qualification on shots 0, 179, 357 checks parity only; its scores are not read for any decision.

## If it passes
One comparison on the 357 confirmation shots under the same rule, with the assets that confirmed D43; if those
cannot be assembled faithfully before 24 October, the entrant decides. Then the projection is applied to the
frozen Final DIII-D archive (MAST unchanged), committed and tagged in the repository, QA'd with the official
validator and submitted as a second Final submission (best scores count). If it fails: recorded, no submission.
