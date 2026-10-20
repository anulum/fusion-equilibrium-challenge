# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71b FEC-R18 guard tests through the real report CLI and the runner's cached-shot check
"""python test_receipts.py QUALIFICATION_DIR  (exits non-zero on the first unexpected acceptance or rejection)

Builds a three-shot mini cohort in a temporary folder from real qualification measures (shots 0, 179, 357), with
its own cases file and activation, then checks that the report accepts the intact cohort and rejects: another
activation, a cases file the activation does not bind, a qualification receipt, a missing shot, an extra shot,
substituted members, changed measure bytes, a stale cached record, a wrong frame count, a wrong array shape, a wrong
map order, and a cohort of the wrong size. The same validate_shot() is what the runner uses before reusing a cached
shot on resume.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import receipts

HERE = Path(__file__).resolve().parent
QUAL = Path(sys.argv[1]).resolve()
PY = sys.executable
results: list[tuple[str, bool]] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(root: Path) -> tuple[Path, Path, Path]:
    run = root / "run"; run.mkdir()
    shots = sorted(p for p in QUAL.iterdir() if (p / "measures.npz").exists())
    cases = []
    for p in shots:
        with np.load(p / "measures.npz", allow_pickle=False) as z:
            frames = int(z["times_ms"].shape[0])
        cases.append({"stem": p.name, "frames": frames})
    cases_file = root / "cases.json"; cases_file.write_text(json.dumps({"cases": cases}))
    activation = root / "activation.json"
    activation.write_text(json.dumps({"name": "test", "D70_activation_sha256": sha(cases_file)}))
    a_sha = sha(activation)
    members = []
    for p, case in zip(shots, cases):
        (run / p.name).mkdir()
        shutil.copy(p / "measures.npz", run / p.name / "measures.npz")
        summary = {"stem": p.name, "frames": case["frames"], "measures_sha256": sha(run / p.name / "measures.npz")}
        (run / p.name / "complete.json").write_text(json.dumps({"summary": summary, "maps": list(receipts.MAPS),
                                                                "activation_sha256": a_sha}))
        members.append(summary)
    (run / "complete.json").write_text(json.dumps({"status": "complete", "shots": len(cases),
                                                    "frames": sum(c["frames"] for c in cases),
                                                    "activation_sha256": a_sha, "members": members}))
    return run, cases_file, activation


def report(run: Path, cases_file: Path, activation: Path, shots: int = 3) -> tuple[int, str]:
    out = run.parent / "report.json"
    done = subprocess.run([PY, "-B", str(HERE / "report_d71b.py"), "--measures", str(run), "--out", str(out),
                           "--activation", str(activation), "--cases", str(cases_file), "--require-shots", str(shots)],
                          capture_output=True, text=True, cwd=HERE)
    return done.returncode, done.stderr


def case(name: str, mutate, expect_ok: bool, shots: int = 3) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run, cases_file, activation = build(Path(tmp))
        mutate(run, cases_file, activation)
        code, err = report(run, cases_file, activation, shots)
        ok = code == 0
        # a rejection counts only if the validator raised it, not another failure
        passed = ok if expect_ok else (not ok and "ReceiptError" in err)
        results.append((name, passed))
        print(("ok  " if passed else "FAIL"), name, "accepted" if ok else "rejected")


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text()); change(data); path.write_text(json.dumps(data))


first = lambda run: sorted(p for p in run.iterdir() if p.is_dir())[0]  # noqa: E731

case("intact three-shot cohort", lambda r, c, a: None, True)
case("another activation", lambda r, c, a: edit_json(a, lambda d: d.update(name="other")), False)
case("cases file not bound by the activation", lambda r, c, a: edit_json(c, lambda d: d["cases"].reverse()), False)
case("qualification receipt", lambda r, c, a: edit_json(r / "complete.json", lambda d: d.update(status="qualification")), False)
case("missing shot", lambda r, c, a: shutil.rmtree(first(r)), False)
case("extra shot folder", lambda r, c, a: shutil.copytree(first(r), r / "d3d_shot_extra"), False)
case("substituted members", lambda r, c, a: edit_json(r / "complete.json", lambda d: d["members"].reverse()), False)


def flip_bytes(r, c, a):
    p = first(r) / "measures.npz"; data = bytearray(p.read_bytes()); data[-10] ^= 0xFF; p.write_bytes(bytes(data))


case("changed measure bytes", flip_bytes, False)
case("stale cached record (other activation)",
     lambda r, c, a: edit_json(first(r) / "complete.json", lambda d: d.update(activation_sha256="0" * 64)), False)
case("wrong frame count in the record",
     lambda r, c, a: edit_json(first(r) / "complete.json", lambda d: d["summary"].update(frames=d["summary"]["frames"] + 1)), False)


def rewrite(r, change):
    p = first(r) / "measures.npz"
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    change(arrays); np.savez_compressed(p, **arrays)
    edit_json(first(r) / "complete.json", lambda d: d["summary"].update(measures_sha256=sha(p)))
    edit_json(r / "complete.json", lambda d: d["members"][0].update(measures_sha256=sha(p)))


case("wrong array shape (hash updated)", lambda r, c, a: rewrite(r, lambda m: m.update(cp_max=m["cp_max"][:-1])), False)
case("wrong map order (hash updated)", lambda r, c, a: rewrite(r, lambda m: m.update(maps=m["maps"][::-1])), False)
case("missing measure (hash updated)", lambda r, c, a: rewrite(r, lambda m: m.pop("gs_surface")), False)
case("cohort of the wrong size (358 required)", lambda r, c, a: None, False, shots=358)
# the runner's cached-shot path
with tempfile.TemporaryDirectory() as tmp:
    run, cases_file, activation = build(Path(tmp))
    shot = first(run); listed = json.loads(cases_file.read_text())["cases"]; c0 = next(c for c in listed if c["stem"] == shot.name)
    for name, act, expect in (("cached shot, current activation", sha(activation), True),
                              ("cached shot, stale activation", "f" * 64, False)):
        try:
            receipts.validate_shot(shot, c0, act); ok = True
        except receipts.ReceiptError:
            ok = False
        results.append((name, ok == expect)); print(("ok  " if ok == expect else "FAIL"), name, "accepted" if ok else "rejected")
failed = [n for n, p in results if not p]
print("FAILURES:", failed)
sys.exit(1 if failed else 0)
