# D72 — a Grad–Shafranov interior for the finished map

20 October 2026, one candidate under a rule fixed before the run (`PROTOCOL.md`, bound by `activation_v1.json`):
the grid nodes inside each finished map's official boundary are replaced by a Grad–Shafranov solution whose
boundary values are the unchanged exterior nodes, with cubic p′ and FF′ fitted to the map and its own plasma current
(`gs_project.py`). `test_gs_project.py` is an analytic probe that prints its results (an exact equilibrium returned
within 3 × 10⁻¹⁵ of the flux drop; a warped copy from residual share 0.37 to 5 × 10⁻⁸); it asserts nothing and is not
an enforced test. The Picard solve stops at 40 steps: 3 435 of the 78 875 frames (185 shots) reached that limit
without meeting the convergence criterion and used the 40-step solution. Scored with the organisers' scorer on the 358 previously used development shots against the
evaluation pipeline of D70 (69 map, 8 joint and 8 static networks fitted without these shots); the base maps and official statistics equal D70's on every shot. `gate_v1.json` is the gate (`gate_d72.py`,
`scores()` copied verbatim from the qualified D45 report): −157 µS, 99 % interval −239 to −93, failed; nothing was
submitted. Study §4.28. `export_manifest_v1.json` lists frozen and published SHA-256; `run_d72.py`,
`d39_common.py` and `run_d72.sh` differ by the publication edits described in [`../README.md`](../README.md).
The run script was not bound by the activation; its recorded original and published hashes are listed separately.

## Reproduce the pooled score and intervals

From the repository root, install `requirements-quality.txt` in a Python 3.12 environment, then run:

```sh
python tools/d72_table.py \
  --data experiments/d72_gs_projection/pooled_statistics_v1.json.gz \
  --manifest experiments/d72_gs_projection/pooled_manifest_v1.json \
  --gate experiments/d72_gs_projection/gate_v1.json
```

The public gzip JSON carries reference sums, sums of squares and counts for each shot and each of ten quantities
(flux, q95, βN and seven derived quantities), both arms' residual sums, per-shot boundary distances, extraction
counters and convergence counts. The manifest fixes the original shot order and records hashes of the original
per-shot records and convergence arrays. Its extraction-source hash identifies the private export script; the
script and raw maps are not redistributed. Extraction checked the activated sources and all 358 base score parts
against D70's margin statistics, using the same fields as the frozen runner.

The maintained verifier resamples **both reference moments and residual sums** in each of the original 20 000
whole-shot multinomial draws (seed 72020). It reconstructs pooled R² and every consistency component, boundary
means over finite-distance shots, failure totals, the score difference and 95 % and 99 % intervals. It does not
average per-shot R² or per-shot scores. All 78 875 frames, including the 3 435 nonconverged frames in 185 shots,
remain included. Output reports the reconstructed −157 µS change and the failed admission rule.

The public computation matches the original scoring function on the same host, including all bootstrap draws.
Compared with the stored report from the original host, the largest observed difference is 2.6 × 10⁻¹⁴ in one
consistency component; the arithmetic comparison uses an absolute tolerance of 10⁻¹³. This numerical tolerance
does not alter the experiment's admission threshold. Invalid inputs return exit status 2 and a fixed refusal
message. Transport hashes and bound identities check the published package; they do not independently
authenticate the original private measurements.

This reproduces score arithmetic and aggregate convergence counts. It does **not** regenerate scored maps,
remeasure physical diagnostics, reconstruct per-frame displacement, or establish a physical improvement from
the projection. Those conclusions cannot be recovered from these pooled statistics alone.

## Physical diagnostics on the actual scored arrays

The subsequent [scored-map diagnostic supplement](../d72_scored_map_physics/README.md) remeasures the existing
base/projected float32 maps against EFIT with corrected D71b operators and retains all 358 shots and
78,875 frames. The tested GS-form residuals and field error against EFIT decrease, while the interface-crossing
current proxy increases; the original score loss and failed gate remain unchanged. Its separate public
sufficient-statistics consumer reproduces diagnostic means and intervals. The pooled-score table above remains
a score-arithmetic check and does not itself contain those physical measurements.

## A correction to the frozen protocol's wording

`PROTOCOL.md` motivates the test with "the share of current density not of Grad-Shafranov form rises from 0.35 % to
1.67 %". That number is D71's polynomial GS-form residual (SSE/TSS) above the reference's, a normalised fit residual
for one profile family, not a share of the physical current. The frozen bytes are kept; study §4.27 and §4.28 use the
corrected wording.

## Custody of these files

The Python, shell and protocol files are archival copies of the frozen originals, not maintained code: their bytes
are the frozen bytes except for the publication edits listed in `export_manifest_v1.json`, so they are not
reformatted, given new headers or linted, and the repository's maintained-code checks do not cover them. Every file
in this folder except this README, the manifest and the `.license` sidecars is listed in the manifest with its
published SHA-256, its category (frozen source, the activation, a result of the frozen run, or a post-freeze
supplement) and, for frozen files, its frozen SHA-256.
