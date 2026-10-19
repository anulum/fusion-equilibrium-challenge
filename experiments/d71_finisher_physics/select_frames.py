# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71 figure frames, chosen by a rule fixed before any D71 result
"""Pool D70's per-frame boundary distance of the submitted stage (margin) over all 358 shots and pick the frames
nearest to the 50th, 90th and 99th percentiles, and the largest. Ties break by shot order, then frame."""
import json
from pathlib import Path

import numpy as np

D70 = Path("$FEC_RUNS/d70_boundary_second_agent_20261015")
cases = json.loads((D70 / "activation_v1.json").read_text())["cases"]
rows = []
for i, case in enumerate(cases):
    with np.load(D70 / "full358_v2" / case["stem"] / "margin.npz", allow_pickle=False) as z:
        for k, d in enumerate(z["distances"]):
            if np.isfinite(d):
                rows.append((float(d), i, k, case["stem"]))
d = np.array([r[0] for r in rows])
chosen = {}
for name, q in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.)):
    target = float(np.quantile(d, q))
    j = min(range(len(rows)), key=lambda t: (abs(rows[t][0] - target), rows[t][1], rows[t][2]))
    chosen[name] = {"quantile": q, "target_metres": target, "distance_metres": rows[j][0], "case_index": rows[j][1],
                    "frame": rows[j][2], "stem": rows[j][3]}
out = {"rule": __doc__, "frames_pooled": len(rows), "chosen": chosen}
Path("figure_frames_v1.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
