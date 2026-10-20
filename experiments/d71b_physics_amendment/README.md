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

## Recompute the public shot-level results

From the repository root, with the pinned quality environment installed:

```bash
.venv/bin/python tools/d71b_table.py \
  --table experiments/d71b_physics_amendment/per_shot_v1.csv.gz \
  --manifest experiments/d71b_physics_amendment/per_shot_manifest_v1.json \
  --report experiments/d71b_physics_amendment/report_v5.json
```

The gzip CSV contains 61 218 rows: for each of the 358 shots and nine measures, ten per-map means and nine
pair-specific means, each with its finite-frame count. A pair mean is the mean of `after - before` on frames
where both values are finite, after the original reference-critical-point validity mask. It is not obtained by
subtracting independently supported map means. The manifest retains the exact original shot order, frame
counts, measurement-file hashes, activation, run receipt, frozen reporter, report and table digests. Its extraction
script digest identifies the private producer; that script and the per-frame measurements are not redistributed.
The published verifier checks the known cohort binding and report bytes, then recomputes all 90 map means,
81 paired changes, their support, 99 % whole-shot intervals and verdicts (20 000 draws, seed 71020).
The maximum arithmetic discrepancy observed with the pinned environment is about 1.4 × 10⁻¹⁷.

This is an arithmetic reproduction from published sufficient statistics. It does not independently authenticate
the private original measurements or recompute physics from maps. Frame sums, medians, 95th percentiles and
critical-point histograms are not reconstructed from this table; the original report and histogram supplement
retain those separate results. The reference validity count and total counts are retained, but individual
frame identities, times and invalid-frame masks are not present. This completed cohort happens to have finite
values for every map/measure; retaining pair-specific statistics preserves the original comparison definition.
The original frozen report and diagnostic code are unchanged. `tools/d71b_table.py` is maintained code under
the repository's strict gates; the original diagnostic runners remain archival.

## Custody of these files

The Python, shell and protocol files are archival copies, not maintained code. Their executable numerical
contents and recorded results remain frozen. Publication edits remove laboratory paths, agent identities and
non-scientific process wording from selected headers, titles or module documentation; original and published
SHA-256 values are retained in `export_manifest_v1.json`. These algorithms are not reformatted or covered by the
repository's maintained-code checks. Every file in this folder except this README, the manifest and the
`.license` sidecars is listed in the manifest with its published SHA-256, its category (frozen source, the
activation, a result of the frozen run, or a post-freeze supplement) and, for frozen files, its frozen SHA-256.
