# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71: gather the four figure frames into one small archive for the repository
import json
import sys
from pathlib import Path

import numpy as np

run, out = Path(sys.argv[1]), Path(sys.argv[2])
selection = json.loads((run / "figure_frames_v1.json").read_text())["chosen"]
packed = {"order": np.array(list(selection))}
for key, item in selection.items():
    z = np.load(run / "full358_v2" / item["stem"] / "figure_frames.npz", allow_pickle=False)
    k = int(item["frame"])
    packed["R"], packed["Z"] = z["R"], z["Z"]
    for name in ("reference", "smoothed", "margin"):
        packed[f"{key}_{name}_psi"] = z[f"frame{k}_{name}_psi"]
        packed[f"{key}_{name}_lcfs"] = z[f"frame{k}_{name}_lcfs"]
    packed[f"{key}_stem"] = np.array(item["stem"])
    packed[f"{key}_time_ms"] = z["times_ms"][list(z["frames"]).index(k)]
    packed[f"{key}_distance_m"] = np.array(item["distance_metres"])
np.savez_compressed(out, **packed)
print(out, out.stat().st_size)
