# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — training-only native pressure information measurement
"""Measure channel-product information lost by independent profile binning."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    protocol = json.loads((here / "protocol_v1.json").read_text())
    source = Path(protocol["adapter_source"])
    assert digest(source / "fec_adapter/diii_d_accuracy/features.py") == protocol["features_source_sha256"]
    assert digest(Path(__file__)) == protocol["measurement_source_sha256"]
    sys.path.insert(0, str(source))
    from fec_adapter.shot_reader import aligned_shot, input_columns
    from fec_adapter.diii_d_accuracy.features import AlignedInputs, HEIGHT_EDGES, RADIUS_EDGES, VERTICAL_TOLERANCE, binned_profile
    cohort = json.loads(Path(protocol["cohort_source"]).read_text())
    assert set(protocol["stems"]) <= set(cohort["training"])
    assert not set(protocol["stems"]) & (set(cohort["holdout"]) | set(cohort["reserved_confirmation"]))
    statistics = {"shots": 0, "frames": 0, "bins": 0, "joint_positive_bins": 0,
                  "multi_channel_bins": 0, "availability_mismatch_bins": 0,
                  "sum_abs_pressure_mean_difference": 0., "sum_native_pressure_mean": 0.}
    ratios = []
    for stem in protocol["stems"]:
        path = Path(protocol["data_root"]) / (stem + ".parquet")
        columns = input_columns(dict.fromkeys(pq.read_schema(path).names))
        table = pq.read_table(path, columns=columns, use_threads=False)
        row = {name: table[name][0].as_py() for name in columns}
        inputs = AlignedInputs.from_shot(aligned_shot(row, stem, np.asarray(row["efit_times"], dtype=float)))
        te, ne = inputs.thomson_te / 1e3, inputs.thomson_ne / 1e19
        vertical = np.abs(inputs.thomson_r - np.median(inputs.thomson_r)) < VERTICAL_TOLERANCE
        joint_valid = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        product = np.where(joint_valid, te * ne, np.nan)
        for member, positions, edges in ((vertical, inputs.thomson_z, HEIGHT_EDGES),
                                          (~vertical, inputs.thomson_r, RADIUS_EDGES)):
            tmean, tfired = binned_profile(te, positions, member, edges)
            nmean, nfired = binned_profile(ne, positions, member, edges)
            pmean, pfired = binned_profile(product, positions, member, edges)
            assignment = np.digitize(positions, edges) - 1
            counts = np.zeros(pmean.shape, dtype=int)
            for bin_index in range(len(edges) - 1):
                counts[:, bin_index] = joint_valid[:, member & (assignment == bin_index)].sum(axis=1)
            proxy = tmean * nmean
            valid = pfired > 0
            difference = proxy - pmean
            statistics["bins"] += pmean.size
            statistics["joint_positive_bins"] += int(valid.sum())
            statistics["multi_channel_bins"] += int((counts >= 2).sum())
            statistics["availability_mismatch_bins"] += int(np.count_nonzero(tfired != nfired))
            statistics["sum_abs_pressure_mean_difference"] += float(np.abs(difference[valid]).sum())
            statistics["sum_native_pressure_mean"] += float(pmean[valid].sum())
            ratios.append((difference[valid] / pmean[valid]).ravel())
        statistics["shots"] += 1
        statistics["frames"] += len(inputs.times_ms)
    relative = np.concatenate(ratios)
    assert len(relative) and np.isfinite(relative).all()
    result = {"status": "fixed_training_only_input_information_measurement_complete_NOT_score_admission",
              "statistics": statistics,
              "absolute_relative_difference_quantiles": np.quantile(np.abs(relative), [0, .5, .9, .99, 1]).tolist(),
              "pressure_weighted_absolute_relative_difference": statistics["sum_abs_pressure_mean_difference"] / statistics["sum_native_pressure_mean"],
              "bins_with_relative_difference_above_one_percent": int((np.abs(relative) > .01).sum()),
              "physical_interpretation": "Mean of channel-wise Ne*Te differs from product of independently binned Ne and Te means; this measures representation loss, not predictability of labels.",
              "target_columns_read": False, "heldout_or_public_scores_read": False,
              "measurement_source_sha256": digest(Path(__file__)), "protocol_sha256": digest(here / "protocol_v1.json")}
    with (here / "measurement_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
