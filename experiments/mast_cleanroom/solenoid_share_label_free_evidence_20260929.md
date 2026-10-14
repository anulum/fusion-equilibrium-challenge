# MAST solenoid current share — label-free evidence (clean-room re-derivation)

Written 2026-09-28T23:15:27Z by first_agent. Purpose: replace the provenance of MAST_SOLENOID_ROW_SHARE = 0.5
(fec_adapter/machine.py:112), which on 2026-09-21 was FITTED on the three starter-kit MAST demo EFIT fields
(0.513/0.512/0.512) — a use of MAST equilibrium data that rule 3 (effective 2026-09-23) forbids for the final phase.

Label-free evidence (no MAST equilibrium, EFIT, demo field or leaderboard score used):

1. Challenge data page (organisers): "magnetics_sol_current reads exactly 2x the FAIR-MAST level-2 pf_active solenoid
   current" and "Its geometry has 656 single-turn elements". pf_active is a measured INPUT channel, permitted under
   rule 3 ("public MAST input data, used without targets").
2. Challenge geometry (input): the 656 solenoid rows form 4 radial layers x 164 turns (R = 0.1190, 0.1318, 0.1446,
   0.1574 m; Z -1.47..1.46 m), all mapped to magnetics_sol_current. Checked on mast_public_test inputs, 2026-09-28T23:15:27Z.
3. UKAEA/Hartree FreeGSNKE (LGPL-3.0, github.com/FusionComputingLab/freegsnke, freegsnke/mastu_tools.py
   get_machine_data): the MAST-U solenoid is built from the p1_inner and p1_outer windings together with
   "multiplier": 0.5 on the circuit current — the modelling convention of the machine's own operators for a
   multi-winding solenoid whose listed filaments together carry twice the circuit current.
   (Pentland et al., arXiv:2407.12432, section 4.1.1: current multiplier factor "used for the solenoid only".)

Conclusion: per-filament current = 0.5 x magnetics_sol_current = the FAIR-MAST pf_active solenoid current, which is
the same value the demo fit found. The value is unchanged; its justification no longer depends on MAST labels.
Disclosure for the methods report: the value was first obtained by a demo-EFIT fit before the rule; it is retained
because two independent label-free sources give the same factor. MAST (not MAST-U) winding topology is not
independently documented here; that residual uncertainty is stated, not hidden.
