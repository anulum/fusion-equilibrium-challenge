---
title: Home
nav_order: 1
---

# Fusion Equilibrium Challenge — the Anulum entry

The permanent record of one entry to the [Fusion Equilibrium Challenge](https://fusion-equilibrium-challenge.sophelio.io/)
(NeurIPS 2026, organised by [Sophelio](https://github.com/Sophelio)): reconstructing the poloidal flux map of a tokamak plasma, and q95 and βN, from coil currents and
Thomson-scattering profiles alone, without any magnetic measurement. DIII-D with training data; MAST zero-shot.

**Development phase:** DIII-D composite S 0.996359 (flux-map R² 0.99988), MAST S 0.7145, 11th at the close of the
Development phase. The blind Final was submitted on 19 October 2026; its result is added here when it is published.

## Read

- [Methods report](METHODS_REPORT.md): the method in two pages.
- [Accompanying study](STUDY.md): what we tried, how we measured it, and what moved the score, including the many
  experiments that failed.
- [Experiment ledger](EXPERIMENT_LEDGER.md): one row per experiment, with the protocol checksum, the number and the
  decision.
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
The published package reproduces the submitted predictions bit for bit.

Cite as: [doi:10.5281/zenodo.23480092](https://doi.org/10.5281/zenodo.23480092).

Developed by Anulum (Miroslav Šotek) with Arcane Sapience, a research system of agents built on large language models.
AGPL-3.0-or-later. Contact: protoscience@anulum.li.
