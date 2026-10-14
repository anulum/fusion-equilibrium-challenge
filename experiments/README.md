<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Experiment records

One directory per experiment of the entry, as it was left when the experiment closed: the protocol that fixed
the question and the decision rule, the scripts that ran it, and its small result files.
[docs/EXPERIMENT_LEDGER.md](../docs/EXPERIMENT_LEDGER.md) is the index: each row names the experiment, its
protocol checksum, the number and the decision. [docs/STUDY.md](../docs/STUDY.md) tells the story in order.

## What is here and what is not

- **Included:** protocols and their amendments (`protocol*.json`, `SHA256SUMS*`), scripts (`.py`, `.sh`), notes
  (`.md`), and result, report and summary files up to 64 kB.
- **Not included:** weights, feature tables, prediction archives and per-shot dumps (too large; the deployed
  weights are fetched by `fetch_weights.py`); scripts that upload or submit; anything holding a credential or an
  address of the laboratory's machines.
- **Edited:** these are copies made for publication. The names of the agents were replaced by `first_agent` and
  `second_agent`, the name of the person who directs the work by "entrant", and local paths by placeholders
  (`$FEC_RUNS`, `$FEC_DATA`, `$FEC_WORK`, `$LAB`, `$SAS`, `$HOME`). Nothing else was changed.

Two consequences of the editing should be stated plainly. The checksums quoted in the ledger and inside the
protocols are those of the unedited originals, so a file here whose text was touched by a replacement will not
reproduce its checksum; the originals are kept by the entrant and are available to the organisers on request.
And the scripts are a record of what was run, on which inputs and in which order, not a second reproduction
package: they expect the laboratory's directory layout. The supported way to reproduce the entry is the package
at the top of this repository.

## Naming

Directories beginning with a lower-case `d` (`d4` to `d61`), `kfold`, `alldata`, `reeval` and `mast_cleanroom`
are the first agent's; those beginning with an upper-case `D` and a timestamp are mostly the second agent's.
The same number can appear in both where one agent reviewed or continued the other's experiment. The `b*.py`
files at the top level are the diagnostics of the first days (boundary structure, member diversity, time
structure).
