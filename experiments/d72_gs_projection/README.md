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
