<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Methods report: SCPN Reactor Systems entry, Fusion Equilibrium Challenge

*Version of 2026-10-14, for the entry displayed as Codabench submission 986068 (Development phase); reproduction and governance
wording corrected 2026-10-20.*

**Entrant:** Anulum (Miroslav Šotek), protoscience@anulum.li.
**Code:** https://github.com/anulum/fusion-equilibrium-challenge; the commit of each Final submission is named
with it. **Weights:** https://huggingface.co/anulum/fusion-equilibrium-challenge at revision
398d05cc14b5b0a67b77d9ddfd900e899a94b2da, pinned in the repository's `WEIGHTS.json`. Licence AGPL-3.0-or-later;
the organisers' scoring functions vendored in the code keep their MIT notice.

## 1. How the work was done

The entry was developed by Arcane Sapience, the entrant's research system built on large language models: a
control-bound system of agents that write code, run experiments, analyse results and review each other's work
under the entrant's governance.
The entrant sets direction, decides, and takes responsibility for every submission.

The governance is concrete:

- later candidate experiments have a written decision rule frozen before fitting or scoring;
- the 28 September audit identified early in-sample correction screens (Study §4.5). Later evaluation copies
  exclude the scored shots from fitting and use paired whole-shot comparisons, but reused cohorts retain
  adaptive-selection uncertainty;
- a failed rule closes that fixed experiment, rather than a whole method class. Two DIII-D choices were made on
  the public score instead, both named in the study (§4.17, §4.23, §7);
- every submitted archive is rebuilt from recorded sources and weights, with SHA-256 checks at each hand-off.

## 2. DIII-D (Challenge 1)

**Inputs.** Only the released inputs: the 18 shaping-coil currents, the E-coil current, plasma current,
toroidal-field current and the Thomson temperature and density profiles. Per reconstruction frame they become
582 features: currents and their rates, currents over the plasma current, discharge phase, Thomson profiles on
fixed position bins, means of these over ±0.08 s and over the preceding 0.3 s and 1.3 s, and 215 sub-frame features from the 0.5 ms
signals within ±20 ms of the frame.

**Representation.** A flux map is written in the principal directions of the training maps. The model answers
64 coordinates and keeps the leading 45; beyond them a predicted coordinate adds more error than it removes.

**Model.**

- 69 multilayer perceptrons (width 512, depth 4) in two families answer the flux coordinates and auxiliary heads:
  magnetic axis, elongation, triangularities, volume, internal inductance, X-point positions, q95 and βN, and (48 of
  them) the boundary radius about the axis in 64 directions. The shape labels are the organisers' own derivation
  applied to the true maps, so what is learned is what is scored. The geometry heads are refitted by ridge
  regression on each network's last activations.
- Four gradient-boosted tree heads (internal inductance, axis radius, elongation, lower X-point radius) are blended
  into the corresponding network heads with fixed weights 0.26, 0.14, 0.09 and 0.16.
- Eight networks that answer only q95 and βN are blended with the 69: their share is 0.46 in q95 and 0.87 in βN.
- Eight networks that answer only the X-point positions supply half of the X-point targets.

**Finishing the maps.** The ensemble-mean map is edited so that the scorer reads the predicted quantities: the
axis is moved onto the predicted axis, the boundary is bent towards the predicted radii, a smooth deformation
sets shape and inductance, and the X-points are moved to their predicted positions. The edits change the map
only within a margin of the vessel.

**Validation.** The last admitted change (the X-point networks) was confirmed once on each of two cohorts of
whole shots that no network in the comparison had seen: +45.5 µS on 238 shots (99 % interval 22.2 to 75.5) and
+36.4 µS on 357 shots (12.0 to 62.4), where 1 µS is 10⁻⁶ of the composite score. The public Development score
moved from 0.996300 to 0.996342. The shares of the scalar networks were then set by least squares on 358
development shots; a nested estimate gave +14 µS with a 99 % interval that included zero, so the change was
checked once on the public test, where the score moved to 0.996359. That last step is a choice made on shots used
before and confirmed on the public fold, and is stated as such.

**What did not help.** More than a dozen further experiments between 11 and 14 October ended without an
admissible gain, among them: 32 settings of the finishing step; dedicated networks for the geometry heads; longer and wider
X-point networks; larger ensembles; explicit electron-pressure features (chord integrals and binned profile);
equal weights for the network families; refitting the scalar output rows of all 77 networks; transferring a
shot's error from similar shots; a network attending over all frames of a shot against a matched per-frame
control; and exponentially weighted discharge history over 0.3 s to 10 s. An analysis of the map that remains
after the model's answer shows that it is low-dimensional (three directions hold 83 % of its energy), persistent
within a shot, and not recovered from the released inputs by any model we fitted (linear, boosted trees, slow
discharge memory). One reading is that it is the part of the equilibrium only the magnetic measurements carry;
the tests do not separate that from what our models fail to extract, so it stays a hypothesis.

## 3. MAST (Challenge 2, zero-shot)

**Method.** An equal-temperature reconstruction, frame by frame: the vacuum flux of the MAST coil set from the
released geometry and currents, plus a plasma current whose amplitudes are fitted to the measured plasma current
and to equal flux where the midplane Thomson profile reads equal electron temperature. q95 is read from the
resulting map; βN comes from the Thomson pressure with a factor for the total over the electron pressure.

**What is fitted to what.** No model is trained on MAST. The pressure factor and the q95 routine were set on
DIII-D training shots. A shot without any solved frame takes the mean current pattern of the solved frames of
the supplied fold, and missing scalars take the fold's median; both are derived from inputs only, so such a shot's
answer depends on the fold it is supplied with.

**Disclosures required by rule 3.**

- Development-phase scores returned by the platform were used to compare MAST pipelines.
- The share of the solenoid current per conductor row (0.5) was first set on 21 September on the three labelled
  demonstration shots, before the rule took effect. It was re-derived on 28 September without labels, from the
  organisers' statement that the released solenoid current reads twice the archive's value, and kept.
- No constant fitted to labelled MAST data and no constant chosen by offline scoring against MAST equilibria
  enters the deployed pipeline; the organisers' boundary-envelope mask is not read (the build refuses to open it).

## 4. External resources

No external dataset and no pre-trained model. Third-party software: NumPy, SciPy, PyArrow, scikit-image,
scikit-learn, JAX and XGBoost at the versions pinned in the package.

## 5. Reproduction

- **Inference.** One command per machine answers any directory of shot files from its inputs alone. On the
  two-Xeon build host, both public folds reproduced every map and scalar bit-identically: 874 DIII-D shots
  (193 359 frames) and 1 206 MAST shots (69 075 frames). Another CPU in hosted CI reproduced one DIII-D shot with
  identical scalars and a maximum flux difference of 7.5 × 10⁻⁹; cross-hardware bit identity is not assumed.
  Measured on the 24-thread build host with 8 worker
  processes: 6 h 14 min for DIII-D and 1 h 52 min for MAST.
- **Training.** `train.py` implements the DIII-D rebuild from raw training files without loading existing DIII-D
  weights; the frozen MAST solver source is retained. Full-production retraining and its resulting composite
  score have not been qualified. The reduced complete chain and one full-recipe fit per family were checked. One
  network of each of the four families, with the full recipe, took 279 to 326 s on one NVIDIA GTX 1060 (6 GiB) and
  265 to 403 s on one AMD RX 6600 XT; extrapolating those fits gives about seven hours for all 85 networks on the
  GTX 1060, excluding data preparation, trees and geometry readouts. For a single A100
  the dense training arithmetic (5.7 × 10¹⁶ FLOPs) gives 0.8 h at peak and 1.6 to 3.3 h at 25 to 50 % of peak;
  this is an estimate, not a measurement. `COMPUTE.md` gives the basis and declares the CPU stages.
- **Weights.** All 85 networks, the four tree models with their leaf tables, and the flux basis are in the package
  with SHA-256 checksums.
