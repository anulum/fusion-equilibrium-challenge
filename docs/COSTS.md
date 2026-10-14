<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# What this entry cost

An estimate of the time, money and energy that went into the entry, from 20 September 2026, when work began,
to 26 October 2026, when the Final phase closes: 37 days. The money figures rest on three numbers stated by the
entrant; the compute figures come from the run records where those exist and are marked as estimates where they
do not.

## Money

| Item | Basis | Amount |
|---|---|---|
| Laboratory operation | 750 CHF a day in its minimal mode, for 37 days; about twelve projects ran in parallel, so one twelfth is charged to this entry | 2 300 CHF |
| The entrant's own time | 17 to 19 hours a day at the workstation, one twelfth of it on this entry: 52 to 59 hours, at 150 CHF an hour | 7 900 to 8 800 CHF |
| **Total** | | **about 10 000 to 11 000 CHF** |

The laboratory figure is the entrant's stated running cost of the laboratory as a whole; it is not itemised
here. Cloud compute was used only on free notebook tiers.

The entrant's hours are the smaller part of the human-equivalent effort. Nearly all of the code, experiments,
analysis and writing was done by the agents of Arcane Sapience under his direction; that work is counted here
only through the laboratory's running cost.

## Time and scale of the work

| Measure | Count |
|---|---|
| Calendar days | 37 |
| Development-phase submissions used | 53 of 100 |
| Experiments and measurements recorded in the ledger | 128 |

## Compute for the deployed model

These are the fits that make up the submitted model, from their own fit reports; [COMPUTE.md](../COMPUTE.md)
gives the detail and the caveats.

| Component | Fits | Summed fit time | Hardware |
|---|---|---|---|
| Map networks fitted on GPU | 32 | 2.3 h | NVIDIA Tesla T4 (free notebook tier) |
| Map networks fitted on CPU | 37 | 49.9 h | notebook-tier and Xeon E5-2640 CPUs |
| q95 and βN networks | 8 | 0.43 h | AMD RX 6600 XT |
| X-point networks | 8 | 0.36 h | AMD RX 6600 XT |
| Tree heads | 4 | 0.53 h | NVIDIA Tesla T4 |

With the full recipe, one network of each family fits in 4.7 to 5.4 minutes on a single NVIDIA GTX 1060 (6 GiB).
All 85 networks would therefore take about seven hours on that one card.

Inference for one public fold: about 50 CPU-hours for DIII-D (874 shots) and about 15 CPU-hours for MAST
(1 206 shots), measured.

## Compute for the whole search (estimate)

No complete meter of the search exists; the following is reconstructed from the run records and is an order of
magnitude, not a measurement.

| Kind of run | Estimate | Basis |
|---|---|---|
| Full-fold builds of candidate archives | 1 500 CPU-hours | about 30 DIII-D builds at up to 50 CPU-hours and about 15 MAST builds at 15 to 18 |
| Studies of the map-finishing step and other CPU experiments | 1 500 to 3 000 CPU-hours | 1.3 CPU-seconds per frame and setting in the finishing studies; feature extraction; tree fits |
| Network fits on local GPUs | 100 to 200 GPU-hours | AMD RX 6600 XT cards, several hundred fits of 2 to 10 minutes and the September campaigns |
| Network and tree fits on free notebook GPUs | 30 to 60 GPU-hours | NVIDIA Tesla T4 |

## Energy (estimate)

| Source | Assumption | Energy |
|---|---|---|
| CPU work | 3 000 to 4 500 CPU-hours at about 12.5 W per thread (a two-socket Xeon E5-2640 server drawing 300 W on 24 threads) | 40 to 55 kWh |
| Local GPU work | 100 to 200 GPU-hours at 160 W | 15 to 30 kWh |
| Notebook GPU work | 30 to 60 GPU-hours at 70 W | 2 to 4 kWh |
| **Total attributable to the entry** | | **roughly 60 to 90 kWh** |

This counts the load of the runs, not the idle draw of machines that were on for other work as well. At
household electricity prices it is a few tens of francs: the cost of the entry is human attention and
laboratory time, not energy.

## What the estimate leaves out

The value of code and infrastructure that existed before the challenge and was reused, among them the entrant's
Grad-Shafranov solver and the coordination system the agents work in; and any valuation of the agents' work
beyond the laboratory's running cost.
