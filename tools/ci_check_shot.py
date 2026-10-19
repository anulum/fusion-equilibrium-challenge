# SPDX-License-Identifier: AGPL-3.0-or-later
# Fusion Equilibrium Challenge entry — compare one regenerated public shot with the submitted Development archive
"""python tools/ci_check_shot.py ARCHIVE.npz

The reference hashes and values are those of shot_0000 (d3d_shot_0047b7837b) in the DIII-D archive submitted as
Codabench #986068. Bit identity is reported; the check fails only if a value moves by more than 1e-5 (relative,
flux) or 1e-5 (absolute, scalars), far inside the organisers' 1e-4 rebuild tolerance on the composite score.
"""
import hashlib
import json
import sys

import numpy as np

REFERENCE = {
    "shot_0000_psirz": ("908d3d5e756c2e4c11487bd138b84b94a9cc3ed9490c5fc2dcdf985056761ee6", (240, 65, 65)),
    "shot_0000_q95": ("84ae16249f1277cd7c3676adb7b8fc586bc41e56fb70bf803f18011e222cd1d7", (240,)),
    "shot_0000_betaN": ("e41095668132e6c97c2a673769a57bfad0627dfdd821c678f9ddeb52c184bb1b", (240,)),
}


def main() -> int:
    archive = np.load(sys.argv[1], allow_pickle=False)
    report = {}
    for key, (digest, shape) in REFERENCE.items():
        value = archive[key]
        assert value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(), key
        report[key] = {"bit_identical": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == digest}
    print(json.dumps(report, indent=1))
    if all(item["bit_identical"] for item in report.values()):
        print("shot_0000 is bit-identical to the submitted archive")
        return 0
    print("not bit-identical on this runner; see the README on numerical identity across CPUs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
