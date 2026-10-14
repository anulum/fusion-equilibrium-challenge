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

## Remaining reproducibility work

Full public inference qualification, runnable training from scratch,
the pinned CUDA training environment and the measured training GPU-hour
declaration remain pending. Historical fit reports are evidence for
recovery; they are not a substitute for an executed training recipe.
