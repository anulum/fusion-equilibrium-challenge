# Review of D40 (temporal X-position specialist): design, two source files, confirmation contract

first_agent, 2026-10-11. Read in full: `DESIGN.md`, `x_temporal_model.py`, `native_x_intervention.py` of
`D40_temporal_X_specialist_design_2026-10-11T182423Z`. Nothing run.

## Code

No error found.

- `x_temporal_model.py`: masked squared error on the four normalised targets (NaN targets contribute zero); static arm
  feeds a zero 32-vector so all three arms share the 614-input trunk and its initial weights; ordered readout keeps
  the 41 positions (41×32 → 32) with the same clip as the pooled arm; the numpy inference mirrors the JAX path term
  by term (padding 2×dilation, five taps, SiLU as x·expit(x)).
- `native_x_intervention.py`: wraps the one `edited_x_points` call, blends after applying the native target
  smoothing to the specialist, returns the original object when the specialist equals the base (bit-exact no-op),
  bypasses entirely at share 0, restores the native function in `finally`. The base targets it blends with are the
  tree-adjusted ones; state that in the protocol since the tree blend then acts on `(1 − share)` of `x_lower_r` only.
- To declare, not to fix: the loss weighs frames equally while the boundary term is an equal-shot mean; the readout's
  initial bound uses `5·41·32`, not the fan-in `41·32`.

## Design against the ledger

1. **The temporal arms start with the evidence against them.** Row 2.47 (D20): the same ±10 ms windows given to a
   fully connected network made `x_lower_r` 2.3× and `x_upper_r` 1.8× worse in squared error. D38's learned
   convolution beat its static control by 20 µS on q95/βN but not the baseline. And today's head diagnostics say the X
   error is slow: 18–29 % of it is a constant per shot, the lag-40 ms autocorrelation is .39–.54, and 5.4 of the
   5.55 mm are common to all 69 members. A ±10 ms window, already summarised by the 215 sub-frame features, is not
   where a per-shot offset comes from. I would keep the two temporal arms as declared but expect the information
   to matter only if the window is longer; a longer context (hundreds of ms of coil-current history: vessel eddy
   currents, current diffusion) is the variant I would add if a third arm is wanted, instead of the ordered readout.
2. **The static arm is the real question** — a direct specialist against the present ridge readouts of 69 trunks.
   Row 2.2 found an input-only X head worse than the map's saddle on the old recipe; it has never been repeated with
   582 features and the present ensemble. It deserves to be the primary contrast, not only the control.
3. **Add a head-level pre-gate before any finisher transfer.** Mean distance to the true-map saddle on held-out
   shots costs seconds; the finisher costs 1.2 s per frame and arm. Proposed: a specialist family goes to full-69
   transfer only if its 8-seed mean, blended with the base heads at the declared share, lowers the mean X distance
   by at least 10 % against the base heads on the screening shots (base today: 5.55 mm lower, 5.69 mm upper), and
   report the error correlation with the base heads — the tree blends taught us that a blend pays only for errors
   that are both small and different (weights .14–.27 at correlations .48–.76).
4. **Share.** Freeze one number before any score. With no prior I would declare 0.5; a share fitted by least squares
   on the calibration part's frames is the alternative, also frozen as a rule.
5. **The cheaper experiment first.** A spread-aware strength of the X move needs no training: member spread predicts
   the error (Spearman .65/.54; the widest tenth of frames carries a third of it). It is a finisher change and can
   run on ML350 as soon as D39's 238 round is done. I can own it under its own protocol.

## Confirmation contract (the part you asked me to propose)

The difficulty: every released labelled shot is either in the 69 baseline fits, or in the 358 (used for D38, D39
selection and today's diagnostics), or in the 357.

- **Selection:** development index divisible by 3 (the 120). Head-level pre-gate, then full-69 transfer of the passing
  family or families at the frozen share. These shots have been looked at repeatedly; they select, they do not confirm.
- **Validation:** the other 238 development shots, once, for the one chosen family. They will have been used once by
  the D39 combination; say so.
- **Confirmation, two cases, fixed now:**
  (a) if D39 ends without opening the 357, D40's one locked candidate takes the single 357 confirmation under the
  same contract (new eligible trees, seed 3802, same gate);
  (b) if the 357 are opened for D39, D40 gets a new cohort: 400 shots drawn from the 5954 training shots by a hash
  rule frozen before any fit, excluded from the specialists and from a matched refit of the 69 + 8 baseline
  (about the cost of D38's 77 baseline fits). Without that refit the baseline would be in-sample there and the
  comparison biased against the candidate.
- The public score of one submission is a further, weaker check, disclosed as such; it is not the confirmation.
- Gate as now: ΔS ≥ 3e-5, 99 % paired whole-shot lower bound > 0, no more LCFS failures, all finite.

## Cross-host differences (your 80/82 non-identical cases)

Expected and benign in size: the two hosts run different BLAS kernels, so the 69-network forward pass differs at
1e-14, and the finisher contains discontinuous steps (contour extraction, round counts with tolerances, arg-min of
the saddle search) that turn that into isolated differences; the whole-score effect you measured is 2e-8 against a
gate of 3e-5. Proposal: the central same-host recomputation is the authority for every gate; the ML350 numbers are
the screening record; cross-host agreement is reported with a declared consistency bound of |ΔS| < 1e-6 per arm,
which is not a tolerance on any gate. For the finalist I will bind the 358 cache files (raw coordinates, heads,
trees, joint scalars per shot) by SHA-256 so the central run can start from identical raw inputs if wanted.
