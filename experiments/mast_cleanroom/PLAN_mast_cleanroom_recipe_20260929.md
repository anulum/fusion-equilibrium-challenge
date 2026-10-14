# MAST clean-room recipe — plan (entrant decision 2026-09-29: final MAST = clean-room)

Author first_agent. Status: plan, nothing frozen or run yet.

## Rule we must satisfy
Rule 3 (effective 2026-09-23): no MAST/MAST-U equilibrium data (FAIR-MAST EFIT, the three starter-kit demo shots,
leaderboard MAST scores, or aggregates of them) may train, fine-tune, calibrate, validate or SELECT any component that
produces MAST predictions. Allowed: MAST inputs, public MAST input data, machine geometry/wall/coils, other machines'
equilibria (DIII-D), our own solver's synthetic equilibria calibrated on MAST inputs only.

## Starting point
Submitted #949223 MAST = v7 thomson_isoflux + per-shot chord fix (MAST S 0.7296). Audit (read-only, 2026-09-29):
every structural choice predates the rule, several were selected with demo S or leaderboard MAST S.

## Decision table: every tunable, its old provenance, the clean-room rule that re-decides it

| Choice | Old provenance | Clean-room re-decision |
|---|---|---|
| Solenoid share 0.5 | demo EFIT fit | label-free evidence file (organisers' 2x vs pf_active input; FreeGSNKE multiplier 0.5) — keep 0.5, disclose |
| TF_TURNS 24 | derived while q95 labels visible, later re-derived from literature | literature (Morgan thesis, Morris 2014) — keep |
| PRESSURE_FACTOR 2.25, q95 routine | 40 DIII-D shots | DIII-D (allowed) — keep |
| MIN_DENSITY 2e18 | input census | inputs only — keep |
| Plasma ends at magnetic boundary (v7) | physics + input measurement | physics argument — keep, document |
| Current basis: 3 terms R(1-x),(1-x)/R,x(1-x)/R vs 4 | demo S looked at | re-decide a priori: 3 terms = the standard minimal p', FF' basis pair plus edge term; decide by synthetic yardstick only |
| Solver attempts/ITERATIONS (1 attempt, 100 its) | leaderboard (v3 0.6953 vs v2 0.7061) | re-decide by LABEL-FREE criteria frozen beforehand: Ip closure residual, isoflux pair residual, fraction converged, and synthetic yardstick |
| Frame-level density gate vs per-channel | demo S | re-decide on input census + synthetic |
| Current mask vs envelope | demo S (+ physics: PF coils inside vessel) | physics argument (coils inside envelope cannot carry plasma current) — keep mask, document |
| Z-height variants | demo S | keep the laser-height hold (physics: Thomson plane), no variant search |
| v7 vs v6 vs v3 as a whole | leaderboard | the recipe is re-assembled from the rows above; no whole-variant choice by score |

## Label-free yardsticks allowed for re-decisions
1. Synthetic MAST truth from our own free-boundary solver driven by MAST INPUT coil currents/Ip/profiles
   (rule 3 explicitly allows) — the existing M1 set (48 shots x 3 profile models) is a relative yardstick only,
   because the generator is weak on DIII-D; improve realism with FreeGS vessel-limited solves (the second agent pilots 29 Sep).
2. Physics self-consistency on real MAST inputs: Ip closure, equal-Te pair residuals, convergence, smoothness in time
   on the true time base.
3. Nothing that reads a MAST EFIT field, demo shot or leaderboard MAST number.

## Order of work
1. Freeze this table as a protocol before any re-run.
2. Build the synthetic yardstick upgrade (FreeGS, MAST inputs) — CPU on ML350 (Qwen may be stopped, entrant OK).
3. Re-decide the four open rows by the frozen criteria; build the clean-room archive; validate format only.
4. Methods report: provenance history disclosed, clean-room table as the final record.

Evidence: first_agent/BRAINSTORM (2026-09-29 audit entry), solenoid_share_label_free_evidence_20260929.md,
RUN_LEDGER.md sec. 3/5, second_agent-recovery results/freegs_* receipts.

## Two tracks (to put before the entrant)

**Track A — clean-room re-justification of v7 (cheap, days: ~1).** Every retained v7 choice gets a label-free
justification (table above: one attempt = the plain convergence rule, 3-term = minimal basis, density gate = input
census, mask = coils cannot carry plasma current, solenoid 0.5 = label-free evidence). Result: a clean-room-defensible
final MAST recipe at today's level (public MAST S about 0.73, Ch2 G_ratio about 0.73). History disclosed.

**Track B — a better MAST method (research, days: 4-7, CPU heavy on ML350).** Board after rule 3: best MAST S 0.9375
(flux 0.9957, D_LCFS 0.051, Consistency 0.873) vs ours 0.7296 (0.9805, 0.188, 0.273): our boundary and derived
shape are the loss, not the flux. Candidates, all rule-3 clean:
- B1 free-boundary physics solve per frame (FreeGS, real MAST coils + vessel wall, Ip, Thomson pressure), its free
  profile/passive-current settings calibrated on DIII-D truth (allowed) — FreeGS on DIII-D so far poor (8 frames, raw
  R2 -7.5..0.87, axis errors up to 0.3 m: missing solenoid/E-coil offset and vessel currents), so DIII-D calibration
  of passive/offset terms is the first gate.
- B2 synthetic MAST equilibria from our own solver driven by the MAST input distribution, then a network trained on
  them (rule 3 allows) — realism is the blocker (house-solver v2 failed on DIII-D); needs B1-level solver realism.
- B3 label-free domain adaptation of the DIII-D network to MAST inputs (exact vacuum flux from coils on both machines,
  plasma part in machine-normalised coordinates; self-supervised/test-time adaptation on MAST inputs is allowed).
Gate for any Track B method: DIII-D analogue score with known truth plus synthetic-MAST yardstick, frozen before a
single MAST archive is built; the MAST leaderboard number is observed only after, never used to choose.
