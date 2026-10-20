# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71b cache regression: the actual run_d71b.one() on cached shot records
"""python test_runner_cache.py QUALIFICATION_DIR ACTIVATION  (exits non-zero on the first unexpected outcome)

Copies one real qualified shot into a temporary output folder and calls the actual run_d71b.one() on its cached
record: the intact case is accepted; a case whose raw, features, reference or D39-cache path points to a changed copy
(the expected SHA-256 kept) is rejected for each of the four dependencies; a stale activation is rejected. The
dependency checks run before the cache is consulted, so no regeneration is attempted on these paths.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import run_d71b
from native_pipeline import digest

QUAL, ACT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
cases = {c["stem"]: c for c in json.loads((run_d71b.D70 / "activation_v1.json").read_text())["cases"]}
results = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp); out = root / "out"; out.mkdir()
    shot = sorted(p for p in QUAL.iterdir() if (p / "complete.json").exists())[0]
    shutil.copytree(shot, out / shot.name)
    run_d71b.STATE.update(output=out, activation_sha256=digest(ACT))
    base = cases[shot.name]

    def attempt(name, case, expect_ok, activation=None):
        if activation:
            run_d71b.STATE["activation_sha256"] = activation
        try:
            run_d71b.one(case); ok = True; why = ""
        except ValueError as error:
            ok = False; why = str(error)
        run_d71b.STATE["activation_sha256"] = digest(ACT)
        passed = ok == expect_ok
        results.append((name, passed)); print(("ok  " if passed else "FAIL"), name, "accepted" if ok else f"rejected: {why}")

    attempt("intact cached shot", dict(base), True)
    for key in ("path", "features", "reference", "D39_cache"):
        changed = root / f"changed_{key}{Path(str(base[key])).suffix}"
        data = bytearray(Path(str(base[key])).read_bytes()); data[len(data) // 2] ^= 0xFF; changed.write_bytes(bytes(data))
        attempt(f"cached shot, changed {key} (expected SHA kept)", {**base, key: str(changed)}, False)
    attempt("cached shot, stale activation", dict(base), False, activation="0" * 64)
failed = [n for n, p in results if not p]
print("FAILURES:", failed)
sys.exit(1 if failed else 0)
