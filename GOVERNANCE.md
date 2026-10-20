<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Fusion Equilibrium Challenge entry — governance and decision rights
-->
# Governance

This repository is the record of one entry to the Fusion Equilibrium Challenge (NeurIPS 2026). It has one
maintainer and changes little; this page says who decides what, and what does not change.

## Roles

- **Maintainer and decision-maker:** the entrant, Miroslav Šotek (Anulum), protoscience@anulum.li. He decides what is
  merged, released or published, moderates issues and discussions under the [Code of Conduct](CODE_OF_CONDUCT.md),
  triages issues and owns security reports ([SECURITY.md](SECURITY.md)). `.github/CODEOWNERS` names the same owner.
- **Research system:** the code, experiments and texts were prepared by Arcane Sapience, the entrant's research
  system: agents that write code, run experiments and review one another's work under the entrant's direction. In
  the record they appear by role, as "the first agent" and "the second agent". They hold no decision right of their
  own; every submission, publication and licence decision is the entrant's.

## What does not change

- The submitted model, its weights and the inference arithmetic. The tags `final-submission-2026-10-19` (the Final
  submission) and `v1.0.0` (the record of the entry) are not moved, and published history is not rewritten.
- The weights stay at the revision pinned in `WEIGHTS.json`; `fetch_weights.py` checks every file against
  `WEIGHTS.sha256`.
- Numerical contents and results of frozen experiment records under `experiments/` are preserved. Publication
  sanitization is documented in their export manifests; scientific corrections are written beside the records.

## How changes are made

- Corrections to documentation and verification tools arrive as issues or pull requests (see
  [CONTRIBUTING.md](CONTRIBUTING.md)). Every changed number must cite the protocol, receipt or ledger row it comes
  from.
- `main` is protected: a pull request must pass the required checks (reproduction smoke test, documentation
  links, dependency audits, the table verifier) before it can be merged, and force pushes and branch deletion are
  blocked. The maintainer's administrator account may push directly (`enforce_admins` is off); such a push is
  accepted only after the same checks have passed on its exact commit, and a failure is repaired by a new commit,
  not by rewriting history.
- New experiments are recorded in `docs/EXPERIMENT_LEDGER.md` with a protocol fixed before the first result; a
  negative or inconclusive result is recorded like a positive one.
- Results of the challenge, when published by the organisers, are added through reviewed documentation; a release is
  made only if the entrant authorises one. They do not replace the record of what was submitted.

## Licence

Source code is AGPL-3.0-or-later; a commercial licence is available from the entrant. Material of others keeps its
own licence; see [NOTICE.md](NOTICE.md) and `REUSE.toml`.
