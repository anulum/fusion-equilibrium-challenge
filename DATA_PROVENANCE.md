# Data provenance

## DIII-D

The production DIII-D neural models use the organiser-provided labelled
DIII-D training data. The 69 deployed field models have verified original
encoder and geometry-readout provenance. Their encoder fit reports record
1,476,543 frames for training-only members and 1,559,340 frames for members
that also train on calibration. Geometry readouts retain their original
training/calibration flags. Production joint scalar and static X-point
fits use the declared all-7,041-shot production cohort. The original
tree models and their selected iteration counts are retained unchanged.

The 358-shot development and 357-shot confirmation cohorts used for local
experiments have both been exposed in previous model comparisons. They
must not be described as untouched holdouts. The experimental 77-member
scalar-readout refit is separate from this deployed inference snapshot.
Its encoders and readouts use the same original fitting shots, with
calibration included only for the original calibration-training members;
its two evaluation cohorts are excluded from those fitting rows.

At inference, features and tree heads are computed from the supplied
actuator and kinetic-profile inputs and prediction timestamps. The package
does not read target maps or scalar labels from a supplied test fold and
does not use precomputed test heads or a previous prediction archive.

## MAST

MAST v12 uses the supplied MAST inputs with the frozen physical solver,
geometry and profile treatment. It does not fit on MAST target labels.
The MAST source snapshot does not contain `mast_envelope.npz`; inference
also rejects attempts to open that organiser target-mask file.

The physical solver includes the declared DIII-D-based calibrations and
engineering constants retained in its frozen source. Historical MAST
Development scores were observed with disclosure; organiser MAST labels
are not used for fitting or offline model selection. The numerical
reproduction reference is the previously deployed MAST v12 archive,
SHA-256 `2febdf3e6baed5cefa4e48cf09a2a5a5d24f6830e1e55c2eee30db2b2acaed51`.

MAST scalar and entirely unsolved-shot fallback statistics are computed
from the supplied split's own inputs and solved predictions. They do not
come from a fixed public-test lookup table.

## State of the reproduction (19 October 2026)

| What | State |
|---|---|
| Inference from the published weights | done: both public folds bit-identical to the #986068 archives on the build host (two Xeon E5-2640); see the README |
| Other CPUs | observed on the GitHub-hosted runner, first public DIII-D shot: identical q95/βN, maximum flux difference 7.5 × 10⁻⁹; CI enforces numerical tolerances |
| Training chain in its limited mode | done: raw shot files to a model directory and its own inference, on two hosts ([COMPUTE.md](COMPUTE.md)) |
| Full recipe, one network per family | done on one NVIDIA GTX 1060 and one AMD RX 6600 XT ([COMPUTE.md](COMPUTE.md)) |
| Full production retrain and the score of its weights | not done |
| Training time on an A100 | not measured; an estimate is in [COMPUTE.md](COMPUTE.md) |
| Final phase | submitted 19 October 2026 (Codabench 1003274); result pending |

Historical fit reports are evidence for recovery; they are not a substitute for an executed training recipe.

## The manifest as frozen

`deployment.json` is kept as it was committed with the package on 14 October; its SHA-256, `ca4d98d2…`, is written
into the receipts of the Final runs, so it is not edited afterwards. Four of its fields describe that day, not the
present: `status` (the full check then pending has since passed, see above), `published` and
`training_reproduction_complete` (both `false` then), and `inference_entrypoint_sha256` (`caf77861…`), which is the
`infer.py` of the earlier package versions (v3 and v4, used for the training-chain checks), not the one in this
repository. The `infer.py` in this repository, `1f4d05d4…`, produced both Final archives and the regeneration of the whole public DIII-D fold; each of their receipts records it as `source_sha256` (the whole public MAST fold was regenerated with the earlier `59af1eeb…`). `verify()` checks the weights and source files listed
in the manifest, not that field.

## Public examples and historical source comments

The full datasets are not redistributed. `tools/reference/d71_figure_frames.npz` contains four selected DIII-D
EFIT reference frames and contours, together with the evaluated predictions used by the reconstruction figure.
Those examples are CC BY 4.0 data excerpts, attributed in `NOTICE.md`. The separate single-shot CI fixture is a
prediction from submission #986068, not a target archive. These files are diagnostics and reproduction checks;
the inference entrypoint does not load them as prediction inputs.

The frozen legacy source also retains historical explanatory docstrings. In particular,
`source/mast_entrypoints/thomson_isoflux.py` describes checks against the three MAST demonstration shots from an
earlier method. That history is disclosed in Study §5.1 and §7; it is not a claim that the deployed v12 was
validated on those labels. The current data-dependency statement is the v12 description above. The frozen source
is preserved for reproducing the submitted model, rather than silently rewriting its recorded hashes.
