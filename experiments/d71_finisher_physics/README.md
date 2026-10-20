# D71 — what the finishing step does to the physics of the DIII-D map

Measurement only, 19 October 2026. The nine native stages of the deployed DIII-D pipeline are regenerated with the
D70 sources, unchanged, on the 358 development shots, and each map is measured against the EFIT reference with the
same operators. `PROTOCOL.md` was frozen before the first result; `report_v2.json` is the historical complete report.
Study §4.27 states the corrections and limits found on 20 October. The corrected D71b full-cohort result is complete: see
[`report_v5.json`](../d71b_physics_amendment/report_v5.json) and its
[histogram supplement](../d71b_physics_amendment/supplement_critical_histograms_v1.json). The dependency inventory uses the same
publication replacements as the experiment sources; hashes retain their original values. The exported `encoded_features_digest` key names the original
`tokens_sha256` value, a feature-cache checksum rather than an access credential.

## What can be checked from this folder

- **Shot means and paired intervals of `report_v2.json`.** `per_shot_v1.csv.gz` holds, for each of the 358 shots and
  each of the ten maps (the reference and nine stages), the shot mean of every bootstrapped quantity and its number
  of valid frames. The table was exported from the original frame measures. The new publication helper
  `per_shot_table.py` independently recomputes every reported shot mean, stage change and 99 % interval from the
  CSV: largest difference 6.94 × 10⁻¹⁸. Frame percentiles need the private per-frame measures. Recomputing withdrawn
  topology rows verifies historical arithmetic, not the validity of those diagnostics. From the repository root:

  ```bash
  python experiments/d71_finisher_physics/per_shot_table.py \
    --table experiments/d71_finisher_physics/per_shot_v1.csv.gz \
    --report experiments/d71_finisher_physics/report_v2.json
  ```
- **The operators.** `physics.py` is the frozen file (its SHA-256 equals the activation's). `test_physics_solovev.py`
  checks it on an analytic Grad–Shafranov solution.
- **The stage sources.** `stage_sources/native_pipeline.py` and `stage_sources/stage_trace.py` are the D70 files the
  runner imports, byte-identical to the checksums in `dependencies_v1.json.gz`.

## What cannot be rerun from public files

The nine stages need the evaluation copies of the model (85 networks and 4 tree heads fitted without these 358
shots), their feature
caches and reference contours. `dependencies_v1.json.gz` lists 1 382 model/basis dependencies and the 358
shot dependency records with their SHA-256 values. They are
kept on the laboratory's storage and are not published; the deployed weights, which were fitted on all shots, are
public but are not the evaluated copies. `run_physics.py` therefore does not run outside the laboratory.

## Files that differ from their frozen checksums

`export_manifest_v1.json` lists the original experiment source files and their frozen and published SHA-256.
The table verifier, table and dependency inventory are publication additions. Three original files differ, by the publication edits
described in [`../README.md`](../README.md) and nothing else:

- `run_physics.py` and `select_frames.py`: the laboratory path of the runs folder replaced by the placeholder
  `$FEC_RUNS` (a literal string here, not expanded), and the second agent's name in a folder name replaced.
- `PROTOCOL.md`: "Owner" replaced by "Entrant", and the seat wording by "first agent" / "second agent".

## Known faults (20 October 2026)

- `critical_points()` is unreliable: one smooth maximum between grid nodes is counted as four maxima, an
  axis-aligned saddle between nodes is missed, a saddle on a node counts as four cells. The topology rows of the
  report are withdrawn; corrected whole-cohort counts are published in D71b and its histogram supplement.
- `gs_form_residual()` is undefined for a nearly constant source (it returns about 1 for an exact field). In these
  data the sensitivity to near-degenerate denominators needs explicit support counts. D71b adds the residual's
  parts and a degeneracy flag; its complete full-cohort result is in `../d71b_physics_amendment/report_v5.json`.
- The docstring of `loop_current()` says the signed loop integral equals *minus* the area integral of Δ*φ / R; it
  equals plus that integral. The computation and the report (which compares the area integral of −Δ*φ / R with
  minus the signed loop integral) are correct.

The cubic GS-form diagnostic is SSE/TSS of a polynomial fit to −Δ*ψ. It is not a percentage of physical current
that violates force balance. The original outside-current statistic uses a different mask for each map and
subtracts a reference value on another mask; it does not isolate current changes in a fixed exterior region.
The public stage sources and CSV are supporting evidence, not a complete independent regeneration package.
