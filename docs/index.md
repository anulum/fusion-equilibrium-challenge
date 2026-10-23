---
title: Home
nav_order: 1
---

# Fusion Equilibrium Challenge — the Anulum entry

The permanent record of one entry to the [Fusion Equilibrium Challenge](https://fusion-equilibrium-challenge.sophelio.io/)
(NeurIPS 2026, organised by [Sophelio](https://github.com/Sophelio)): reconstructing the poloidal flux map of a
tokamak plasma, and q95 and βN, from coil currents, measured plasma current I_p and Thomson-scattering profiles,
without magnetic probes or flux-loop measurements.

## A higher score does not certify force balance

On 358 development shots, the finishing step added 1981 µS to the score while increasing several measured
GS-form residuals and field/current errors. An interior projection subsequently reduced the tested GS-form
residuals but lost 157 µS and increased an interface-current proxy. These are specific numerical diagnostics,
not percentages of nonequilibrium current or a force-balance certificate.
[Physics audit](PHYSICS_AUDIT.md) explains the corrected measurements, public table reproduction and tools for
examining another model's maps.

## The entry

The entry has two parts that share nothing but the task:

- **DIII-D:** an ensemble of networks trained on the organisers' labelled DIII-D shots, followed by a finishing step
  that edits each predicted map towards predicted shape targets ([Methods §2](METHODS_REPORT.md)).
- **MAST:** a physics reconstruction frame by frame. No model is trained on MAST and no constant is fitted to MAST
  equilibria; public MAST scores were used to compare pipelines, with disclosure ([Methods §3](METHODS_REPORT.md)). "Zero-shot" means exactly this; it is not the DIII-D model transferred to a
  second machine.

## Result

Development phase, public test folds, submission 986068 (14 October 2026); 11th at the close of the Development
phase. The blind Final was submitted on 19 October 2026 (Codabench 1003274); its result is added here when it is
published.

| | DIII-D | MAST |
|---|---:|---:|
| Benchmark score S | 0.996359 | 0.714450 |
| Flux map R² | 0.999883 | 0.979450 |
| q95 and βN R² | 0.998005 | 0.330685 |
| Boundary distance D_LCFS | 0.008180 | 0.191791 |
| Consistency of derived scalars | 0.987704 | 0.226646 |

S = 0.55 R²ψ + 0.15 R²{q95,βN} + 0.10 (1 − D_LCFS) + 0.20 Consistency. Where the remaining 1 − S comes from, by
our arithmetic on the components above:

| Share of 1 − S | DIII-D | MAST |
|---|---:|---:|
| Flux map | 1.8 % | 4.0 % |
| q95 and βN | 8.2 % | 35.2 % |
| Boundary distance | 22.5 % | 6.7 % |
| Consistency of derived scalars | 67.5 % | 54.2 % |

This splits the benchmark loss; it does not say how much of it can be removed. On DIII-D pixelwise flux-map R² is
near one; this does not establish accurate derivatives or force balance. About 90 % of the remaining score deficit
is the boundary and the geometry derived from the map; on MAST, 89 % is the
scalars and their consistency.

![Reference, map after the finisher and difference on four DIII-D development frames](figures/fig6_reconstruction.png)

Four DIII-D development frames, chosen by a fixed rule (median, 90th and 99th percentile and largest boundary
error): the reference, the map after the finisher and their difference. The evaluated networks exclude these
shots from fitting, but the shots were used before in model comparisons ([Study §4.27](STUDY.md#427-what-the-finishing-step-does-to-the-physics-of-the-map)).

## Other findings

1. **Representation beat ensemble size.** 215 features describing what the 0.5 ms input signals do within ±20 ms of
   each 20 ms frame let 32 refitted networks score higher than the 195-network ensemble before them: +1.98 × 10⁻⁴ on the public
   score, the largest single gain of the entry ([Study §4.16](STUDY.md#416-the-signals-inside-a-frame)).
2. **Most ideas did not help, and the record says by how much.** Correctors, history models, new input forms,
   losses and finisher settings were tested with the exclusions and limitations recorded for each experiment;
   early in-sample screens and capacity-confounded comparisons are disclosed
   ([What did not work](NEGATIVE_RESULTS.md)).
3. **A physics reconstruction without MAST labels reached 0.7145 on MAST,** and label-free repairs confirmed on a
   synthetic testbed did not transfer to the real machine ([Study §5.9](STUDY.md#59-what-the-mast-result-says-a-limit-of-our-approach-not-of-the-problem)).

## Read

- [Physics audit](PHYSICS_AUDIT.md): what the corrected measurements establish, and how to recompute them.
- [Audit your maps](AUDIT_TOOL.md): measure GS-form, field and current diagnostics with explicit units,
  support and fixed comparison regions, without our trained models.

- [Methods report](METHODS_REPORT.md): the method in two pages.
- [Accompanying study](STUDY.md): what we tried, how we measured it, and what moved the score, including the many
  experiments that failed.
- [Experiment ledger](EXPERIMENT_LEDGER.md): one row per experiment, with the protocol checksum, the number and the
  decision.
- [What did not work](NEGATIVE_RESULTS.md): the negative results by theme, with the numbers.
- [What the entry cost](COSTS.md): time, money, compute and energy.

## The challenge

Rules and announcements: [fusion-equilibrium-challenge.sophelio.io](https://fusion-equilibrium-challenge.sophelio.io/).
Starter kit: [Sophelio/fusion-equilibrium-challenge-starter](https://github.com/Sophelio/fusion-equilibrium-challenge-starter).
Data: [Sophelio/fusion-equilibrium-challenge](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge) on Hugging Face.
Platform: [Codabench 17739](https://www.codabench.org/competitions/17739/).

## Discuss

[Discussions](https://github.com/anulum/fusion-equilibrium-challenge/discussions): open problems, reproduction questions and the negative results we kept.

## Reproduce

Code, pinned environments, training and inference entrypoints: [github.com/anulum/fusion-equilibrium-challenge](https://github.com/anulum/fusion-equilibrium-challenge).
Weights: [huggingface.co/anulum/fusion-equilibrium-challenge](https://huggingface.co/anulum/fusion-equilibrium-challenge).
**[Reproduce the Final submission](https://github.com/anulum/fusion-equilibrium-challenge/releases/tag/final-submission-2026-10-19):**
the exact commit, weight revision, input revision and output checksums. A first look in five minutes:
[the quickstart notebook in Colab](https://colab.research.google.com/github/anulum/fusion-equilibrium-challenge/blob/main/notebooks/quickstart.ipynb).

On the hardware the entry was built on (two Intel Xeon E5-2640), the DIII-D package reproduces the submitted public
fold bit for bit. The whole-fold MAST check used the earlier `package_v2`; the current package has not been rerun
on the whole MAST fold. On other CPUs the last bits of float32 can differ. On one DIII-D shot, hosted CI
observed identical q95/βN and maximum flux difference 7.5 × 10⁻⁹; CI enforces numerical tolerances. [All checks](https://github.com/anulum/fusion-equilibrium-challenge#reproduce-the-predictions).

Cite as: [doi:10.5281/zenodo.23480092](https://doi.org/10.5281/zenodo.23480092). Code archived by [Software Heritage](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/anulum/fusion-equilibrium-challenge) (`swh:1:snp:143cae236c490f64c48de43ca03761a9623e16d6`).

Developed by Anulum (Miroslav Šotek) with Arcane Sapience, a research system of agents built on large language models.
AGPL-3.0-or-later. Contact: protoscience@anulum.li.

The [audit calibration](AUDIT_CALIBRATION.md) provides exact stencil controls and a reference-assisted comparison with the organisers’ PCA–Ridge method.
