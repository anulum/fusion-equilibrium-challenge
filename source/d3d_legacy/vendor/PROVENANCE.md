<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Vendored solver modules — provenance

These files are byte-identical copies of our own SCPN Fusion Core solver, taken
so that this package can choose its own JAX build without touching that
repository or its pinned environment. The solver modules are not third-party
code; the vessel masks in `challenge_masks/` are, and have their own section below.

| | |
|---|---|
| Source repository | `03_CODE/SCPN-REACTOR-SYSTEMS/repositories/SCPN-FUSION-CORE` |
| Source commit | `09ec449b02029009944b3d4a668716bb7ff1b728` |
| Source path | `src/scpn_fusion/core/` |
| Copied | 2026-09-20T17:45Z |
| Modules | 11 |

Every copied module was clean in git at that commit — verified before copying,
because a copy taken from a dirty tree has no provenance at all.

## This copy is not a mirror — do not re-sync it

**No file here is ever refreshed from SCPN Fusion Core.** The copy was taken
once, at the commit above, and from that moment it is the source of truth for
this project. The commit is recorded so the divergence can be read, not so the
files can be pulled again.

The direction of travel is outward: fixes made here are kept as patches under
`patches/`, against the upstream paths, so they can be reviewed and landed in
Fusion Core as that repository's own change. Nothing travels inward.

A re-sync would be actively destructive: it would silently revert the two fixes
below, and the failure mode they cause — the magnetic axis walking to the
chamber wall — looks like a physics problem rather than a lost patch. The tests
in `tests/test_vendor_patches.py` exist to make that loud if it ever happens
anyway.

The two `__init__.py` shims are **ours** as well: the upstream package
initialiser imports the whole of Fusion Core.

## Checksums at the time of copying

```
5523093a374d4ba3093d8ddca005b43dadd66ae90fd8cab3b57bdbe1eeaa4e0e  jax_continuation_history.py
ba1576d406e52f63a44c677a1eabba2440713bfb4df917b87abb503d2d4e9b99  jax_equilibrium_solver.py
2d877ed057afe9b8709903a174e18166d17c0aea1a64ae5f03481bea88eedada  jax_free_boundary_gs.py
4a112778cf9f6c9617d1270e12a3cefbc84b1b2b39eec85d36a6e571ed6bb8ac  jax_free_boundary_gs_implicit.py
6864163d4c4ccd43ee7b3bf289262400400d12587f35ce3858067d0dbfa600d9  jax_free_boundary_predictive.py
59e5a2e0b9777d55c1c43dd1c49b3f88db8d21bb7490ed7d3b5b9698426727da  jax_multigrid_precond.py
e5c5d6bc85e116137527ec9cb2ed177c10e23bac5d64a484f04ae66d7929c214  jax_o_point.py
5838a285a02e3f433c42e672c1cd98263e2439c63bd48bf82ff50f8d96225eb9  jax_plasma_support.py
a0cfa87e519ce1ddc5f38d1d3c04e3a7732bceaa7c2e9e26ddda2f4292ae6199  jax_predictive_checkpoint_trace.py
b6b6efe157701e9e60f1591a0743718dc9d5fc13ec1ca6a49c3ab1f1cf46b0f5  jax_predictive_forward_compiled.py
853816efe6b1b73a211f093764d844c6ca54967a2f98537d9a32131123786f24  jax_x_point.py
```

## Divergence from upstream — two fixes and one extension carried here, not in the repository

Two of the eleven modules are **no longer byte-identical**. All changes are
recorded as patches in `patches/`, against the upstream paths, so they can be
reviewed and carried back to SCPN Fusion Core as its own change. Until they
are, the repository is untouched by design.

The two fixes were found by running the solver on real DIII-D shots, which is
the validation the repository states it has never had.

### 0001 — Anderson least squares taken through the normal equations

`solve_predictive_equilibrium_compiled` solved its Anderson step with
`jnp.linalg.lstsq(df, f, rcond=None)`. The step is severely ill-conditioned:
on a representative case the least-squares and the regularised normal-equation
solutions differ by 1e8, so any change in operation order inside XLA moves the
whole trajectory.

`jnp.linalg.lstsq` itself is bit-identical across JAX 0.7.1 and 0.8.2 —
measured, not assumed — but the compiled forward built on it is not:

| Back-end | JAX 0.7.1 axis | JAX 0.8.2 axis |
|---|---|---|
| `lstsq` (was the default) | (1.6017, −0.0732) | (1.6749, **−0.8500**) |
| `normal_eq` (now the default) | (1.5638, −0.0416) | (1.5638, −0.0416) |

The normal-equation path is bit-identical between the two versions and lands
closer to the reconstruction's own axis. The option already existed in the
module and was documented there as "mathematically the same least squares up to
regularisation"; only the default changes.

### 0002 — the predictive seed is preferred over the coil-only-wall seed

The Newton solve chose between two seeds by whichever had the lower residual.
The two are not equals. The fixed-boundary seed imposes the wall flux from the
coils alone, and this module's own opening paragraph records that the plasma
self-field at the wall is about 2.7 times the coil field on the DIII-D
reference — so that wall condition is roughly 70 % wrong and the seed is not
predictive from coils alone. Its residual can still be the lower one, because
it is the residual of a different problem.

On real shots that is exactly what happened: the residual comparison sent the
solve onto the coil-only seed and the magnetic axis onto the chamber wall. The
predictive seed is now taken whenever it is admissible, and the coil-only one
is kept as the fallback it always was.

### 0003 — the batched solve takes one plasma current and one start field per sample

Not a defect fix: an extension the time-series use needs.
`solve_predictive_equilibrium_batched` held the whole batch to a single scalar
`ip_target` and a single shared `psi_init`, and discarded each sample's
iteration count. That fits its original purpose — an ensemble of parameter
samples around one equilibrium — and does not fit a discharge, where every
frame has its own measured current and its own best start (a converged
neighbour in time).

- `ip_target`: a scalar, or one value per sample, shape `(B,)`.
- `psi_init`: one shared field `(NZ, NR)`, or one per sample `(B, NZ, NR)`.
- `return_iterations=True` returns `(ψ, k)` with `k` of shape `(B,)`;
  `k < n_iter` proves that sample met the tolerance, `k == n_iter` must be read
  as not converged.

The scalar and shared forms are unchanged in meaning (test-pinned: a scalar
equals the constant vector exactly). Each batch element equals the single
compiled solve given that element's own inputs to 1e-9 of the flux span —
`tests/test_batched_per_sample.py`. The patch is taken against the file with
0001 already applied; applied in order, the three patches reproduce this copy
byte for byte.

### Result

With the two fixes, the same reference frame solves identically on JAX 0.7.1 and
0.8.2: seed `total_field_predictive`, both Newton stages converged in 4 and 3
iterations, scaled residuals 1.511e-09 and 5.843e-11, axis at
(R = 1.584, Z = −0.050) against the reconstruction's (1.735, −0.042).

This also removes the version ceiling that blocked the AMD GPUs: the ROCm 7.2
wheels ship jaxlib 0.8.x, which the solver now runs on.

## Vessel envelope masks — third-party, MIT

`challenge_masks/d3d_envelope.npz` and `challenge_masks/mast_envelope.npz` are
byte-identical copies of `fusion_scoring/masks/` in the organisers' starter kit
(`github.com/Sophelio/fusion-equilibrium-challenge-starter`, MIT, © 2026 Sophelio
and the Fusion Equilibrium Challenge authors; licence text in
`challenge_masks/LICENSE`). They are machine geometry — the region of the grid
inside each vessel — and are the same files the scorer uses to find the last
closed surface, which is why they are copied and not redrawn.

| File | SHA-256 |
|---|---|
| `d3d_envelope.npz` | `5fc7f52a8b253db94020b46f20bd299560a799bc0135a1eda0c611b8001a07f1` |
| `mast_envelope.npz` | `50ab5a3e99205978db235ec3b85baa084f87308e7f8bdda6f3c57c7d5ff03616` |

`tests/test_vessel.py` pins both digests; `FEC_MASK_DIR` points the package at
another directory.
