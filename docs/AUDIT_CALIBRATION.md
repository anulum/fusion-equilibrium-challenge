---
title: Calibrate the audit
layout: default
nav_order: 10
---

# Calibrate the map audit

A small fitted GS residual and a small derivative error answer different questions. This calibration combines
known analytic controls with four previously published DIII-D illustrations and an independently authored
baseline method. The baseline was reproduced by this entry; this is not an outside replication or a new
competition submission.

## Recompute the diagnostics

Install the hash-pinned diagnostic environment from [the audit guide](AUDIT_TOOL.md), then run from the repository root:

```bash
.venv/bin/python -m tools.audit_calibration
```

The command prints deterministic JSON. It generates eight analytic controls in memory and measures all twelve
illustration maps. It does not train a model, download data, replace files or compute the competition score.
The small stored predictions and the original illustration archive are verified against the manifest, using
bounded byte snapshots before array allocation. Missing or altered inputs produce a fixed refusal and exit 2.
The [scientific record and input manifest](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/audit_calibration)
retain the exact shot identities, exclusions, input hashes, fit parameters and output hashes.

## Analytic controls

The field in metres and Wb/rad is

φ(R,Z) = −(R² − 1.7²)²/8 − 0.9Z²,

on R ∈ [0.84, 2.54], Z ∈ [−1.6, 1.6]. Its continuous source is −Δ*φ = R² + 1.8. On the
central stencil used by the audit, the exact sampled source is R² + 1.8 − ΔR²/4. The constant source bias
therefore decreases by four when spacing is halved across the 33, 65 and 129 point grids. A cubic fitted
source can absorb this bias: a tiny fitted residual does not imply an exact continuous derivative.
The report separates the known pointwise bias, measured source RMS, continuous source RMS and fitted residual.
The RMS difference is a difference of norms, not an RMS of pointwise errors.

Two controlled perturbations add α(R−1.7)²Z² with α = 0.1 and 0.4 on the 65 point grid. Their comparison
uses the same unperturbed plasma mask and reference field. These amplitudes are calibration controls, not
estimates of experimental noise. Additional controls retain the actual unavailable reasons for a missing mask,
open surfaces and an exact zero source. They never substitute zero for an unavailable diagnostic.

## An external method, with explicit limits

The baseline follows the organisers' PCA + Ridge recipe at pinned starter commit
[`823f6eef`](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/tree/823f6eef1f527e1b5566a8fc1e952da6e2fadf43):
21 coil/plasma-current inputs, corrected Ip timestamps, linear temporal interpolation with extrapolation,
training-only StandardScaler, 50 PCA coordinates and RidgeCV over ten regularisations from 10⁻³ to 10³.
Thomson profiles and a finisher are absent from this baseline. The component count follows the pinned code;
it is not the 20 components mentioned in the starter website's shorter description.

Before fitting, all 358 shots of the existing diagnostic cohort were excluded from the 7,041 released DIII-D
training shots. Of the 6,683 remaining, 512 were selected by the smallest SHA-256 of
`fec-audit-baseline-v1:` followed by the shot stem, then fitted in lexical order. No prediction outcome selected
the subset. All 112,374 selected frames were finite; none were trimmed. The fit used NumPy seed 42 and
float32 PCA coordinates/predictions with float64 PCA arithmetic. RidgeCV uses the starter's default framewise
leave-one-out selection within the training data. It does not use the four evaluation shots.

This differs from the starter's small default sample and internal train/validation/test partition: the fixed
512 shots are all training shots here, with the entire diagnostic cohort excluded separately. The exact-source
collection used Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and PyArrow 25.0.1. A separate
maintained offline reproduction command is available:

```bash
.venv/bin/python -m tools.official_ridge \
  --data /path/to/data/diii_d_train \
  --manifest experiments/audit_calibration/baseline_manifest_v1.json \
  --out /path/to/new-baseline-output
```

This command performs the declared fit and requires the original released shot files. Its output directory
must be new. It writes four prediction maps and a receipt with training/evaluation identities and checksums;
it checks the observed raw-file hashes against the manifest before fitting, and does not serialise a model or modify the submitted weights. Allow sufficient memory: the reported
112,374-frame fit was run under a 24 GiB limit on a two-Xeon host. Different numerical libraries or BLAS
execution can change the last bits; the checksummed stored predictions are the input to the public audit.

## What the four-frame comparison measures

The frames are the original D71 illustration: nearest the median, 90th and 99th percentile of this entry's
finished boundary error, and the largest error. They were selected using this entry's errors, not the baseline's.
They are reused illustrative cases, not a representative population sample, a new confirmation cohort or a
basis for ranking methods. Model capacity and use of Thomson signals also differ.

The measured values below are diagnostic ratios multiplied by 100. `B` is the poloidal-field L2 error
against the reference; `GS` is the cubic fitted source RMS residual divided by source RMS. All values
come from the [published calibration report](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/experiments/audit_calibration/report_v1.json).

| Original illustration | Entry B, % | PCA–Ridge B, % | Reference GS, % | Entry GS, % | PCA–Ridge GS, % |
|---|---:|---:|---:|---:|---:|
| p50 | 2.105 | 11.915 | 0.626 | 8.986 | 1.836 |
| p90 | 3.485 | 43.277 | 3.427 | 17.670 | 4.972 |
| p99 | 4.356 | 67.944 | 0.110 | 12.453 | 2.846 |
| max | 39.153 | 53.998 | 3.544 | 35.876 | 7.481 |

PCA–Ridge has a smaller fitted GS ratio and a larger field error on each of these four examples.
This illustrates the limited information supplied by a fitted residual alone. These deliberately selected,
reused cases do not rank methods on a shot population or certify physical force balance.

For each frame, reference, entry and baseline use the same fixed EFIT polygon mask and central-stencil support.
This **reference-assisted diagnostic** uses each map's flux sampled on the reference contour for its boundary
value, and its own grid extremum inside that mask for axis normalisation. It is not the baseline predicting a
boundary. Own-map surface contours can still differ, so their individual support, rank and conditioning remain
in the report. Reference field error and current ratios use the identical spatial region across all three maps.
A shared flux-sign reversal is a calibration check, not a determination of physical toroidal direction.
These regions and normalisations differ from the whole-cohort D71b and scored-map study; their numbers should
not be treated as the same experiment.

Use the [external-map command](AUDIT_TOOL.md) for maps with your own explicitly declared units, masks and
reference policy. Small residuals here do not establish pressure profiles, topology, complete force balance
or an irreducible information limit.
