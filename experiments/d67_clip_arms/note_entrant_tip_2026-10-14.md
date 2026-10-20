# D67 — note added before any result (2026-10-14)

- Arm B of D67 (clip 50) keeps the full gradation of the availability window means (their lowest z is about -13)
  and the 0/1 flags; it also unclips one continuous te column (32 sigma,
  0.04 % of frames). An alternative encoding of the same information is run only if D67 passes.
- Wording correction for the ledger: the "fired" flag of binned_profile means "at least one finite positive value
  in the bin", not a laser event; a 0 does not tell a dropout from a plasma that no longer reaches the bin.
- New hypothesis queued (not part of D67): the context means average te and ne with zeros for missing bins, mixing
  value and availability; a candidate representation is context means over available values only, with the
  availability share as its own column.
- Descriptive additions to the D67 report (not part of the frozen rule): per arm the share of beta_n squared error
  in the worst shots, and the change inside the D65 clipped frames against the change outside them.
