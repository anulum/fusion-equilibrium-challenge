---
title: Audit your maps
layout: default
nav_order: 9
---

# Audit an axisymmetric flux map

The audit command measures discrete field, current and Grad–Shafranov (GS) form diagnostics on a supplied map.
It needs no trained model or private prediction cache. A reference map and measured plasma current are optional.
These diagnostics do not compute the competition score or certify physical force balance.

## Run an example

From the repository root, install the maintained diagnostic environment:

```bash
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements-quality.txt
```

Create an analytic field, with coordinates in metres and flux per radian:

```python
import numpy as np

R = np.linspace(0.84, 2.54, 65)
Z = np.linspace(-1.6, 1.6, 65)
r, z = np.meshgrid(R, Z)
psi = -(r**2 - 1.7**2)**2 / 8 - 0.9*z**2
np.savez("analytic.npz", R=R, Z=Z, psi=psi,
         plasma_mask=psi >= -0.3, boundary_flux=-0.3)
```

```bash
.venv/bin/python -m tools.equilibrium_audit \
  --map analytic.npz --length-unit m --flux-unit Wb/rad --axis-sign 1
```

For this example the axis is a maximum. `--axis-sign -1` declares an axis that is a minimum. The command orients
each input as φ = axis_sign × flux per radian. Returned signed fields and currents use this orientation; they do
not establish a physical toroidal direction. `--flux-unit Wb-total` divides full poloidal flux by 2π.
Length units are `m`, `cm` or `mm`; a unit applies to both coordinate arrays.

An optional comparison uses the same unit declarations; its sign can differ:

```bash
.venv/bin/python -m tools.equilibrium_audit \
  --map analytic.npz --length-unit m --flux-unit Wb/rad --axis-sign 1 \
  --reference analytic.npz --reference-axis-sign 1 --measured-current-a 1000000
```

The current value here illustrates the interface; it is not an independently measured current for this analytic
example. Compare measured currents by magnitude only. Output is deterministic JSON on stdout. Valid maps can
have unavailable diagnostics, represented by null values, reasons and support counts. Invalid files, declarations
or numerical range produce exit code 2 and a fixed refusal without file paths or exception text. The command
does not write files; shell redirection is optional and controlled by the caller.

## Select a map from a competition archive

The adapter reads the prediction format written by this repository's `infer.py`:
`shot_0000_psirz`, `shot_0001_psirz`, and subsequent zero-based shot indices. Each
selected array must have shape `(frames, 65, 65)`, with rows along Z and columns
along R, and contain real float32 or float64 values. The archive itself supplies
neither physical coordinates nor timestamps; the adapter does not infer them
from the machine name, shot index or array shape.

Supply a separate `grid.npz` containing `R` and `Z` for the selected map, then
run the adapter and the existing audit as two commands:

```bash
.venv/bin/python -m tools.competition_map \
  --archive predictions.npz --shot-index 0 --frame 12 \
  --grid grid.npz --out selected-frame \
  --length-unit m --flux-unit Wb/rad --axis-sign 1
.venv/bin/python -m tools.equilibrium_audit \
  --map selected-frame/map.npz \
  --length-unit m --flux-unit Wb/rad --axis-sign 1
```

The output directory must be new. It contains the selected `map.npz` and a
`competition-map-selection.v1` receipt. The frame's numerical values, dtype and
Z/R orientation are preserved, including for a Fortran-ordered source array.
The receipt binds the full archive bytes, array key, frame index, selected frame
bytes, supplied grid, exported map and adapter source with SHA-256 identities.
An optional `--archive-sha256 HASH` requires the caller's expected archive
identity. Hashing reads the entire archive, while selection allocates only one
frame; unselected frames and scalar predictions are not validated. This command
does not compute the competition score.

To select by time, replace `--frame 12` with `--time-ms 1200 --times times.npz`.
The supplied NPZ must contain only `times_ms`: one finite, strictly increasing
timestamp per frame of the selected shot. The requested time must match exactly;
there is no nearest-time selection or interpolation. Passing `--times` with a
frame index instead records that frame's declared timestamp. The receipt also
records the time-axis checksum. These are caller-supplied timestamps, not time
metadata discovered inside the prediction archive.

The grid file may additionally contain `plasma_mask`, `axis_flux` and
`boundary_flux` for this particular frame, under the single-map contract below.
The adapter passes them through for validation without constructing a plasma
region from the predictions. Without suitable metadata, dependent diagnostics
remain unavailable. A reference map and fixed comparison region can be supplied
to the audit command in the usual way. Units and sign must be declared again
consistently when running that command. Invalid selection, input or destination
returns exit code 2 and a fixed refusal; an existing output directory is retained.

### Interpret an input refusal

Refusals retain exit code 2, `status: refused` and the same fixed message. The additive
`equilibrium-audit-refusal.v1` schema and `code` fields identify a controlled category without exposing
exception text, file paths or a traceback. Successful audit reports keep `equilibrium-audit.v1`.

| Code | What to check |
|---|---|
| `INVALID_ARGUMENT` | Supply required options, supported signs and correctly typed arguments. |
| `INPUT_UNREADABLE` | Check that the supplied files exist and are readable. |
| `MISSING_ARRAY` | Supply every required NPZ array for the map or comparison region. |
| `INVALID_LENGTH_UNIT` | Declare `m`, `cm` or `mm`. |
| `INVALID_FLUX_UNIT` | Declare `Wb/rad` or `Wb-total`. |
| `GRID_NONUNIFORM` | Supply coordinates with the uniform spacing required below. |
| `NUMERIC_RANGE_EXCEEDED` | Check magnitudes, unit declarations and grid spacing; arithmetic exceeded numerical range. |
| `AUDIT_REFUSED` | The remaining input or calculation was refused; check the full input contract. |

Codes classify the detected failure, not every possible error in a file. Unavailable diagnostics within
a successful audit still use their individual reasons and support counts; they are not command refusals.

### Identify the measuring tool

Successful reports include an additive `tool` object with schema `equilibrium-audit-tool.v1`. It hashes the
actual four measuring source files, records the distributed `requirements-quality.txt` hash, and identifies
Python, machine architecture and the installed NumPy, SciPy and scikit-image versions. The
`implementation_sha256` is a content identifier for those sources. The `measurement_identity_sha256` binds
that identifier, the lock hash and the recorded environment together. A source or lock change changes its
corresponding identity, including when running from a source archive without Git history.

If the lock is absent, its hash is null and its status is `not_distributed`; the tool does not invent an
unavailable release or clean commit. These fields identify the files and environment observed by the command,
not a signature authenticating the run, proof that installed packages match the lock, or a guarantee of
identical floating-point results across hosts. Input and region hashes remain separate from tool identity.

## Input contract

The map is an NPZ archive containing:

| Key | Shape and meaning |
|---|---|
| `R`, `Z` | One-dimensional, finite, strictly increasing coordinates; R must be positive. |
| `psi` | Finite real array of shape `(len(Z), len(R))`, with no missing values. |
| `plasma_mask` | Optional boolean array of the same shape, defining this map's plasma region. |
| `axis_flux` | Optional finite scalar, in the input flux convention. Otherwise the grid extremum within a nonempty plasma mask, or over the full grid, is explicitly labelled an estimate. |
| `boundary_flux` | Optional finite scalar in the same convention; never inferred. |

At least three coordinates are required along each dimension. Spacing must be uniform within relative 10⁻⁵;
the report records its actual deviation. Derivatives use mean spacing. Coordinates are co-registered exactly
after SI conversion; reference maps are not silently interpolated.

NPY v1/v2 numerical arrays of at most 64 bits are accepted; masks must have boolean dtype. Pickle, object,
structured and complex arrays, duplicate/extra archive keys and rank greater than two are refused. A byte
snapshot binds the processed input to its SHA-256. Limits are 64 MiB compressed archive, 128 MiB declared
expansion, and 1,048,576 elements per array and grid nodes per map. Headers are checked before array allocation.

Without a plasma mask or boundary flux, field diagnostics remain usable but GS fits are unavailable. A supplied
axis-boundary pair must be ordered after orientation. The command does not locate a magnetic axis, infer a
boundary or count critical points.

## Fix the spatial comparison

By default the region policy is `grid-interior.v1`: all grid nodes are requested, and only nodes with a complete
central stencil are used. To supply an independent region, write an NPZ with keys `R`, `Z` and boolean `mask`
and pass `--comparison-region region.npz`. Coordinates use the map's declared length unit.

The `fixed-input-region.v1` identity hashes policy, SI coordinates and mask, independently of flux values.
Use the same identity when comparing models. Adding another model never revises an earlier region. The caller
is responsible for documenting how a supplied mask was obtained; a checksum does not authenticate its origin.
Per-map plasma masks and flux surfaces remain distinct, potentially moving domains, not fixed spatial regions.

## Read the diagnostics

The versioned central five-point operator is Δ*φ = φ_RR − φ_R/R + φ_ZZ. First derivatives give
B_R = −φ_Z/R and B_Z = φ_R/R in tesla. Axis-oriented Jφ = −Δ*φ/(μ₀R), with μ₀ = 4π × 10⁻⁷ H/m,
is in A/m². Edge nodes lack a central stencil. Current sums multiply supported nodal values by ΔRΔZ;
these are discrete area quadratures, not exact continuous integrals. Requested and used counts accompany them.

The polynomial GS-form diagnostic fits −Δ*φ to eight columns `[R² ψN^k, ψN^k]`, k = 0…3,
where ψN = (axis − φ)/(axis − boundary), on at least 30 supported plasma nodes. It reports
√(SSE/source energy), SSE, TSS, residual/source RMS, coefficients, numerical rank, singular values and condition.
The variance-degeneracy threshold is TSS ≤ 10⁻¹⁰ × source energy. Rank-deficient fits retain a value with an
explicit flag; a zero source gives no ratio. Unconstrained fitted coefficients are not measured source profiles.

The surface GS diagnostic uses levels 0.2, 0.4, 0.6, 0.8 of this map's own flux. It retains the original
marching-squares vertex weighting and affine R² source fit, with per-level point counts, rank and conditioning.
A finite ratio can use fewer than four surfaces: inspect `full_support` and `used_levels`. Unavailable closed
surfaces and zero sources are separate outcomes. These definitions match the maintained
[surface diagnostic](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/tools/surface_gs.py) used to examine the corrected D71b result.

With a reference, relative field error is √(Σ|B−B_ref|²/Σ|B_ref|²) on the fixed region. A zero denominator
produces null. An additional diagnostic divides candidate absolute current on that region by reference absolute
current inside the reference plasma mask. This ratio is not a fraction of current outside equilibrium.

A small residual tests the selected discretisation, domain and fitted source family. It does not independently
establish pressure balance, correct source profiles, topology or adequate spatial resolution. The analytic
example has a known stencil error: −Δ*φ = R² + 1.8 − ΔR²/4, even though its continuous source is R² + 1.8.
Pixelwise agreement alone says little about derivative errors. The
[scored-map experiment](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/experiments/d72_scored_map_physics/README.md) shows why score and these diagnostics
need separate measurement.

[Calibrate the audit](AUDIT_CALIBRATION.md) with known discretisation errors, unavailable controls and the organisers’ independently authored PCA–Ridge method on four reference-assisted illustrations.


## Remeasure the original scored-map example

The [scored-field replay](https://github.com/anulum/fusion-equilibrium-challenge/tree/main/experiments/d72_scored_map_replay)
provides three actual reference/base/projection frames with the original grid and contours.
It runs the historical diagnostic definitions and compares newly measured statistics to retained
original values, without model weights. It also exports surface-fit support and paired full-support
sensitivity. This is a separate input contract from the general audit above: flux values remain in
the unchanged released convention, rather than being assigned inferred absolute units. The small
example validates a measurement path and does not reproduce the whole-cohort claim.
