# Track A — v7-CR: clean-room re-justification of the submitted MAST recipe

Author first_agent, 2026-09-29. Status: FROZEN record (provenance verified 2026-09-29 against git history and RUN_LEDGER sec. 5); nothing re-run.
Scope: the MAST half of submission 949223 (archive mast_public_test.npz SHA 50b50184…c63a65), entrypoints
second_agent-recovery-20260926/results/joint_fixed_axis_volume_reproduction_56cb/entrypoints/
predict_mast.py (49c0f62c…) and thomson_isoflux.py (6588971a…); machine.py solenoid share; cross_machine/map_scalars.py.

Rule 3 (from 2026-09-23) forbids MAST equilibrium data for training/calibration/validation/SELECTION of MAST
components. For each tunable: value, how it was originally set, and the label-free justification that the final
phase relies on. "Verify" marks a provenance I have not yet confirmed in the primary records.

| Constant | Value | Original provenance | Label-free justification (final) |
|---|---|---|---|
| MAST_SOLENOID_ROW_SHARE | 0.5 | demo EFIT fit 21 Sep | solenoid_share_label_free_evidence_20260929.md (organisers' 2x vs pf_active input; FreeGSNKE multiplier 0.5) |
| TF_TURNS MAST | 24 | set while q95 labels visible; re-derived | literature (Morgan thesis; Morris 2014) |
| PRESSURE_FACTOR | 2.25 | 40 DIII-D shots | DIII-D (another machine, allowed) |
| Q_SHARE / CORE_DEPTH | 0.95 / 0.2 | q95 routine checked on DIII-D | DIII-D; Q_SHARE 0.95 is the definition of q95 |
| MIN_DENSITY | 2e18 m^-3 | input census 68 979 frames | inputs only |
| MIN_CURRENT | 1e5 A | a priori | below 100 kA no equilibrium to reconstruct |
| PLATEAU_SHARE | 0.9 | "high-current test shots, no label involved" (code docstring) | inputs only |
| ITERATIONS, one attempt | 100, no retry | leaderboard compared v2/v3 | the plain convergence rule: a frame that does not settle from a warm start is filled from its neighbour; no retry machinery (Occam). Relaxation/iterations set by converged-frame count, label-free (RUN_LEDGER l.1601, VERIFIED) |
| RELAXATION / HEIGHT_GAIN / CONVERGED | 0.3 / 0.4 / 2e-3 | set once in 988ab89 (22 Sep 00:52), never changed; RUN_LEDGER l.1601: "Numerics were chosen by the share of converged frames only" | numerical-stability settings chosen label-free (VERIFIED) |
| Current basis | 3 terms | demo S looked at | minimal p'/FF' pair + edge term; no search |
| Density gate per frame | frame median | demo S looked at | input census shows per-frame live/dead split |
| Current mask vs envelope | mask | demo S looked at | coils sit inside the vessel envelope and carry no plasma current (physics) |
| Laser-height vertical hold | Thomson plane | Z variants rejected by demo S | the only vertical information in the inputs is the Thomson plane (physics); no variant search |
| TEMPERATURE_SHARES, EDGE_SHARE, SMOOTHING_LENGTH, MIN_CHANNELS, MIN_PEAK_EV, MIN_PAIRS, PAIR_TOLERANCE, AXIS_TOLERANCE, SELF_CELL | as in code | each set exactly once in the first commit 988ab89 (22 Sep 00:52) and never changed (git log -G, VERIFIED); the demo check came after and changed none of them | a priori settings of the method; edge pair weight 3 = the edge pair is the most informative isoflux pair (RUN_LEDGER l.1631, physics) |
| Chord tables per shot | fix 27 Sep | correctness fix, MAST labels not used | inputs only |

Disclosure text (methods report): the MAST recipe was developed before rule 3 with checks on the three demo shots
and the development leaderboard; for the final phase every retained setting has a label-free justification listed
here, and no setting was changed or selected using MAST equilibrium data after 2026-09-23.

Remaining residual: MAST (not MAST-U) solenoid winding topology is not independently documented; stated in the solenoid evidence file.
