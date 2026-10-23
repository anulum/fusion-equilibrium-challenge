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
