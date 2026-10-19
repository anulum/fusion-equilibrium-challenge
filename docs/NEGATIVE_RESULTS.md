<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# What did not work

Most of what we tried did not improve the entry. This page gathers those results by theme, so that someone working
on the same problem can skip them or try them differently. Every number is copied from the table in
[§6 of the study](STUDY.md#6-what-moved-the-score); the study's sections and the [experiment ledger](EXPERIMENT_LEDGER.md)
give the protocol, the cohort and the decision rule behind each one.

How to read the numbers: S is the composite score of the challenge, and 1 µS = 10⁻⁶ S. "Out of fold" means measured
on shots held out of the fits. An interval is a whole-shot bootstrap interval. A result is negative *under our
setup*: our networks, our finisher, our compute and the released inputs. None of these rows shows that the
information is absent from the data.

## Correcting the networks after the fact

| What | Result |
|---|---|
| X-point head from the inputs alone; a map-aware X-point corrector | worse than the map itself |
| Boosted trees fitted on the networks' out-of-fold errors (all 5936 shots) | proxy −2 × 10⁻⁶ (interval −5.5 to +1.3 × 10⁻⁶) |
| A shot's error predicted from its five most similar training shots | correlation of offsets −0.04 to −0.20 |

The error that remains is not predictable from the inputs by these means: three directions hold 83 % of what is
left of the map, and 38 % of it is an offset per shot ([§4.25](STUDY.md#425-what-is-left-of-the-map)).

## More information from the inputs

| What | Result |
|---|---|
| Whole-shot context; 27 physics features (profiles, pressure, pedestal, current diffusion) | no information beyond the per-frame model |
| History of the discharge (gate before building a sequence model) | residual R² worse on all nine targets |
| A network attending over all frames of a shot, against an identical network without context | −0.7 × 10⁻⁵ (95 % interval −6.3 to +6.2 × 10⁻⁵) |
| Slow memory of the discharge (21 measures over 0.3 to 10 s) | leading direction +2.6 % (P 0.73) |
| Raw 0.5 ms windows of the input signals as 840 further inputs | flux error ×1.25, every head worse; proxy −5.1 × 10⁻⁴ |
| Electron pressure as explicit inputs (two forms) | +0.1 × 10⁻⁵ (P 0.55); +0.1 × 10⁻⁵ (P 0.54) |
| Zero-initialised branch on the full model: raw Thomson channels, and coarse bins as its control | −198 µS (99 % −300 to −118) and −157 µS (−241 to −93) |
| Thomson context over available values only, availability kept separately | heads −1.3 × 10⁻⁴ (95 % −2.0 to −0.2 × 10⁻⁴) |

What did work here was finer: features computed inside each 20 ms frame from the 0.5 ms inputs gave the largest
single gain of the entry, +1.98 × 10⁻⁴ on the public score ([§4.16](STUDY.md#416-the-signals-inside-a-frame)).

## Weights and losses in training

| What | Result |
|---|---|
| Scalar label weight ×4 (one fold) | +1.25 × 10⁻⁵ (P 0.57): heads up, triangularity and boundary down |
| A larger weight on one scalar label only (one fold) | −1.09 × 10⁻⁴ (P 0.07) |
| Boundary-label weight 0.005 (all five folds) | −1.5 × 10⁻⁵ (P 0.25); +2.2 × 10⁻⁵ on the first two folds did not replicate |
| Boundary radius readout trained with a Hausdorff loss | −166 µS (99 % −209 to −128) |
| Input clip at 50 instead of 8, four matched networks per arm | heads +3.2 × 10⁻⁵ (95 % −7.1 to +13.3 × 10⁻⁵); flux error ×1.05 |
| q95 and βN heads on the target divided by B_t/I_p | +1.1 × 10⁻⁵ in the term, but flux error ×1.22 |

## The finishing step

The finisher edits the predicted map (shape, X-points, contour) before it is scored. Its September settings held
against every later attempt to retune them.

| What | Result |
|---|---|
| Re-tuning, nested out of fold | +0.9 × 10⁻⁵ (P 0.90), below the +2 × 10⁻⁵ rule |
| Larger shares, chosen because they looked slightly better out of fold | public −4.1 × 10⁻⁵ (#962577); the milder step −0.5 × 10⁻⁵ (#963337) |
| Larger shares for triangularity and elongation | −4.8 to −7.3 × 10⁻⁵; the boundary distance grows 7–8 % |
| 26 settings, cubic warps and two composites on the held-out answers of the 69 networks | best +1.1 × 10⁻⁵ (interval −0.2 to +5.3 × 10⁻⁵) |
| X-point move by a share that depends on member disagreement | +0.3 × 10⁻⁵ (interval −2.7 to +2.9 × 10⁻⁵) |

## More networks, other networks

| What | Result |
|---|---|
| Scalar heads from specialist networks (all five folds) | +1.9 × 10⁻⁵ (P 0.77); first two folds +6.0 × 10⁻⁵, the other three −1.0 × 10⁻⁵; public −8.2 × 10⁻⁵ (#961602) |
| Calibration rows for the five networks without them | +8.1 × 10⁻⁶ (P 0.67); per-fold scatter ±5 × 10⁻⁵ |
| Adding the rest of the refitted recipe to the 32 networks | public −3.1 × 10⁻⁵ (#966320) and −1.79 × 10⁻⁴ (#966479), although out of fold it lowered the flux error |
| Dedicated networks for the seven shape targets | predicted −5 × 10⁻⁵; 3 to 9 % worse than the deployed heads |
| X-point networks trained three times longer or twice as wide; 16 or 24 members | +0.2, 0.0, +0.6, +0.7 × 10⁻⁵ |
| Boosted trees trained four times longer | equal to the fifth decimal |
| Equal weight per network family for q95 and βN; refit of the scalar output rows | +0.8 × 10⁻⁵ (failed its rule); 10⁻⁷ |
| Temporal smoothing of the scalars (windows of 3, 5 and 9 frames) | every scalar worse |
| The plasma part of the map (coils' flux subtracted) as the target of the basis | better at 8 coordinates, worse from 32 upwards |

## MAST without MAST equilibrium data

| What | Result |
|---|---|
| v8 correction field | pilot rejected |
| v8b boundary from the Thomson edge | breaks current closure ([§5.4](STUDY.md#54-two-corrections)) |
| v9 edge current terms in the solve | solved frames 85 % → 22 % |
| Coil freedom | confirmed on the synthetic testbed (D −23 %); failed the guards on real shots |
| v10 vessel current with coil freedom | synthetic D −38 to −45 %; public 0.7246 against 0.7296 before: the synthetic gain did not transfer |
| v11f with the FreeGS default profile | solved frames 68.5 %; public 0.6011 |
| FreeGS with measured coil currents (feasibility, 12 frames) | converged on 6 |

Why we stopped there: [§5.9](STUDY.md#59-what-the-mast-result-says-a-limit-of-our-approach-not-of-the-problem).

## Three lessons

1. **Two folds are not five.** Twice a gain on the first two folds vanished on the other three (the boundary-label
   weight and the specialist scalar heads). After that, levers were measured on all five folds or on whole
   held-out cohorts, under a rule fixed before the first result.
2. **A proxy can point the wrong way.** Larger finisher shares, and network sets that lowered the raw flux error
   out of fold, lost on the public score. Only the full scoring path, with the organisers' contour extraction, settles a
   change to the map.
3. **Synthetic gains need a real check.** The MAST testbed confirmed changes that did not hold on real shots.

Corrections and replications are welcome in
[Discussions](https://github.com/anulum/fusion-equilibrium-challenge/discussions).
