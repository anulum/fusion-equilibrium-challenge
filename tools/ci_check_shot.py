# SPDX-License-Identifier: AGPL-3.0-or-later
# Fusion Equilibrium Challenge entry — compare one regenerated public shot with the submitted Development archive
"""python tools/ci_check_shot.py ARCHIVE.npz

The reference is shot_0000 (d3d_shot_0047b7837b) exactly as submitted in Codabench #986068, stored in
tools/reference/. On the hardware the entry was built on (two Xeon E5-2640) the regeneration is bit-identical; on
other CPUs floating-point results may differ in the last bits. The check reports bit identity and the largest
differences, and fails only if the flux map moves by more than 1e-5 of its range or a scalar by more than 1e-4.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference" / "d3d_shot_0047b7837b_submitted_986068.npz"
DIGESTS = {
    "shot_0000_psirz": "908d3d5e756c2e4c11487bd138b84b94a9cc3ed9490c5fc2dcdf985056761ee6",
    "shot_0000_q95": "84ae16249f1277cd7c3676adb7b8fc586bc41e56fb70bf803f18011e222cd1d7",
    "shot_0000_betaN": "e41095668132e6c97c2a673769a57bfad0627dfdd821c678f9ddeb52c184bb1b",
}


def main() -> int:
    new = np.load(sys.argv[1], allow_pickle=False)
    ref = np.load(REFERENCE, allow_pickle=False)
    report, ok = {}, True
    for key, digest in DIGESTS.items():
        a, b = new[key], ref[key]
        assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == digest, "reference file changed"
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape and np.isfinite(a).all(), key
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        scale = float(np.ptp(b)) if key.endswith("psirz") else 1.0
        limit = 1e-5 if key.endswith("psirz") else 1e-4
        report[key] = {"bit_identical": bool(np.array_equal(a, b)), "max_abs_difference": diff,
                       "max_difference_over_scale": diff / scale, "limit": limit}
        ok &= diff / scale <= limit
    print(json.dumps(report, indent=1))
    print("bit-identical" if all(r["bit_identical"] for r in report.values()) else
          ("within tolerance (not bit-identical on this CPU)" if ok else "OUTSIDE tolerance"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
