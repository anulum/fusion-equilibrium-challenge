# Calibrated audit cases and an official-method baseline

Recompute all eight analytic controls and twelve illustration audits from the repository root:

```bash
.venv/bin/python -m tools.audit_calibration
```

Install `requirements-quality.txt` with `--require-hashes` first. The command verifies the actual published
payloads against `baseline_manifest_v1.json`; it requires no private networks, prediction cache or training data.
`report_v1.json` records the output. Floating-point last bits can differ across hosts; the mathematical
calibration tests use explicit tolerances and exact counts.

The [guide](../../docs/AUDIT_CALIBRATION.md) defines units, operators, unavailable cases, original frame selection,
source licences, sampling and the separate offline fit command. The baseline is the organisers’ independently
authored coil/current PCA–Ridge method reproduced by this entry, not an independently executed external
replication. No new competitive model, candidate score or MAST-label use is involved.

## Reference-assisted illustration results

Each row compares the same fixed EFIT plasma region and central stencil across all three maps. Boundary flux
is each map's mean on the reference contour; axis flux is its grid extremum inside that reference mask. GS
surfaces remain own-map contours and their actual support is retained. These diagnostic normalisations use
reference geometry; the baseline is not predicting a boundary in this test.

All entries below are diagnostic ratios multiplied by 100, not fractions of current outside equilibrium.
`B` is the relative poloidal-field L2 error against the reference on the fixed region. `GS` is the cubic fitted
source RMS residual divided by source RMS in that same reference plasma mask.

| Original illustration | Entry B, % | Baseline B, % | Reference GS, % | Entry GS, % | Baseline GS, % |
|---|---:|---:|---:|---:|---:|
| p50 | 2.105 | 11.915 | 0.626 | 8.986 | 1.836 |
| p90 | 3.485 | 43.277 | 3.427 | 17.670 | 4.972 |
| p99 | 4.356 | 67.944 | 0.110 | 12.453 | 2.846 |
| max | 39.153 | 53.998 | 3.544 | 35.876 | 7.481 |

The baseline has a smaller fitted GS ratio and a larger field error on each of these four examples. This
illustrates why a small fitted residual alone is insufficient to select a reconstruction. It does not rank
methods on a population or establish which map satisfies physical force balance.

The four cases were selected by this entry's original boundary-error percentiles, not the baseline's results.
Their shots were excluded from fitting, but they are reused, biased illustrations with different model capacity
and input usage. Do not infer an average score, an information ceiling or a generalisation confidence interval
from this table.

## Fit lineage

The pinned upstream starter is `823f6eef1f527e1b5566a8fc1e952da6e2fadf43`. The manifest records source URLs and
SHA-256 values, 512 selected training identities and raw file hashes, all 358 excluded cohort identities,
all four evaluation identities/times, environment and fitted regularisation. Selection used only identifiers,
before any baseline outcome. There were 112,374 finite training frames and no trimmed frames.

The baseline archive contains just four float32 prediction maps. No fitted model or full training dataset is
included. The original reference/entry maps remain the unchanged `tools/reference/d71_figure_frames.npz`.
See [NOTICE.md](../../NOTICE.md) for the MIT starter attribution and the separately attributed CC BY 4.0
DIII-D training data. The analytic calibration is not a vacuum-matched whole-vessel solution or a pressure,
topology or force-balance certificate.

On the checked host, a full 512-shot replay reproduced the four stored maps and their prediction archive
bit for bit. Alignment was also compared with the pinned upstream definitions on six real shots, and the
50-component fit/decoder on 300 training frames with four independent queries. The maintained estimator
arithmetic is identical to the full replay; raw-file identity and refusal checks are tested separately.
These are checks by the entry, not an independent external reproduction or an attestation of a historical fit.
