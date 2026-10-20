# D71b — corrected diagnostics of the finishing step

Measurement only, 20 October 2026, on the same 358 development shots and nine regenerated stages as D71. `PROTOCOL.md` holds the frozen protocol and its four
amendments, each written before any D71b result; `activation_v5.json` binds the sources, tests and run script by
SHA-256. `report_v5.json` is the complete report of the full run; `supplement_critical_histograms_v1.json` holds the
complete histograms of detected critical points (the report's own reference-count histogram stops at three and
omits two reference frames with four detected saddles); study §4.27 reads both.

- `critical.py`: critical points from one interpolating bicubic spline (roots of its gradient, its own Hessian).
- `physics_b.py`: exterior current on one common region per frame, the polynomial GS-form fit with its parts and a
  degeneracy flag, and the flux-surface GS test. D71's `physics.py` is imported unchanged.
- `receipts.py`: every shot record and the run receipt bound to the activation and the exact cohort.
- Tests: `test_critical.py`, `test_physics_b.py` (analytic), `test_receipts.py` (the report CLI on a mini cohort
  built from real measures), `test_runner_cache.py` (the runner's cached-shot path). `qualification_v3_receipt.json`
  records the real-grid qualification of amendment 2.

`supplement_critical_histograms_v1.json` was computed after the run by a small unpublished helper (SHA-256 in the
manifest) that first validates the run with `receipts.validate_run` and reads the frozen measures; the manifest
records the run receipt and report it was computed from. `run_d71b.py` and `run_d71b.sh` differ from their frozen
bytes by the publication edits described in [`../README.md`](../README.md) (laboratory paths and agent names).
Like D71, the run needs the private evaluation copies of the model and their caches, so it does not run outside the
laboratory; the per-frame measures stay on the laboratory's storage.

## Custody of these files

The Python, shell and protocol files are archival copies, not maintained code. Their executable numerical
contents and recorded results remain frozen. Publication edits remove laboratory paths, agent identities and
non-scientific process wording from selected headers, titles or module documentation; original and published
SHA-256 values are retained in `export_manifest_v1.json`. These algorithms are not reformatted or covered by the
repository's maintained-code checks. Every file in this folder except this README, the manifest and the
`.license` sidecars is listed in the manifest with its published SHA-256, its category (frozen source, the
activation, a result of the frozen run, or a post-freeze supplement) and, for frozen files, its frozen SHA-256.
