# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — training-only scalar readout qualification
"""Probe the existing ridge kernel on real eligible training rows, without admission."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Return a scientific artifact's SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Qualify four fixed models using only the first eligible training block."""
    here = Path(__file__).resolve().parent
    group = Path("$LAB")
    prior = group / "the internal record"
    package = Path("$FEC_RUNS/ml350/runs/legacy_tree_fec_first_agent_20261005/deploy/pkg")
    sys.path.insert(0, str(prior))
    from training_data import initialize
    sys.path.insert(0, str(package / "entrypoints"))
    from geometry_readout_streaming import corrected_from_moments, cpu_moments, validity_groups
    from fec_adapter.diii_d_accuracy.network import Network
    protocol = json.loads((here / "probe_protocol.json").read_text())
    data = initialize()
    rows = data["rows"]["training"][:1024]
    features = data["base"]["training"]["features"][rows].astype(float)
    labels = list(data["metas"]["training"]["label_names"])
    names = ["q95", "beta_n"]
    truth = data["base"]["training"]["labels"][rows][:, [labels.index(n) for n in names]].astype(float)
    groups = validity_groups(truth)
    tasks = json.loads((prior / "baseline_tasks.json").read_text())["tasks"]
    reports = []
    for name in protocol["pilot_models"]:
        task = next(task for task in tasks if task["name"] == name)
        path = prior / ("fits" if task["family"] == "joint" else "readouts") / name / "network.npz"
        network = Network.load(path)
        if network.feature_mean.size < features.shape[1]:
            raise ValueError("Unexpected narrower pilot feature contract")
        inputs = np.pad(features, ((0, 0), (0, network.feature_mean.size - features.shape[1])))
        moments = cpu_moments(network, inputs, truth, names, groups)
        corrected, fit_report = corrected_from_moments(network, moments, names, groups, ridge=.001)
        old_coordinates, old_heads = network.predict(inputs)
        new_coordinates, new_heads = corrected.predict(inputs)
        np.testing.assert_array_equal(old_coordinates, new_coordinates)
        unchanged = sorted(set(old_heads) - set(names))
        for label in unchanged:
            np.testing.assert_array_equal(old_heads[label], new_heads[label])
        indices = [network.coordinates + network.label_names.index(label) for label in names]
        untouched = np.ones(network.weights[-1].shape[0], dtype=bool)
        untouched[indices] = False
        for old, new in zip(network.weights[:-1], corrected.weights[:-1], strict=True):
            np.testing.assert_array_equal(old, new)
        for old, new in zip(network.biases[:-1], corrected.biases[:-1], strict=True):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(network.weights[-1][untouched], corrected.weights[-1][untouched])
        np.testing.assert_array_equal(network.biases[-1][untouched], corrected.biases[-1][untouched])
        np.testing.assert_array_equal(network.linear, corrected.linear)
        assert all(np.isfinite(a).all() for a in corrected.weights + corrected.biases)
        errors = {}
        for index, label in enumerate(names):
            valid = np.isfinite(truth[:, index])
            before = float(np.mean((truth[valid, index] - old_heads[label][valid]) ** 2))
            after = float(np.mean((truth[valid, index] - new_heads[label][valid]) ** 2))
            assert after < before
            errors[label] = {"rows": int(valid.sum()), "training_MSE_before": before, "training_MSE_after": after}
        reports.append({"model": name, "input_sha256": digest(path), "fit": fit_report,
                        "flux_and_other_heads_bitexact": True, "training_implementation_check_only": errors})
    result = {"status": "training_only_public_prediction_readout_probe_PASS_no_admission", "models": reports,
              "rows": len(rows), "heldout_or_calibration_targets_read": False,
              "network_class_source_sha256": digest(Path(sys.modules[Network.__module__].__file__)),
              "kernel_source_sha256": digest(package / "entrypoints/geometry_readout_streaming.py"),
              "protocol_sha256": digest(here / "probe_protocol.json"), "source_sha256": digest(Path(__file__))}
    with (here / "probe_receipt_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "models": len(reports), "rows": len(rows)}))


if __name__ == "__main__":
    main()
