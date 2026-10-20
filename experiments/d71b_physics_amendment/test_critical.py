# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71b checks of the critical-point detector: the review's counterexamples, rotations,
# shifts and resolutions. Exits non-zero on the first failure.
import sys

import numpy as np

from critical import counts, critical_points

failures = []


def expect(name, phi, R, Z, want):
    got = counts(critical_points(phi, R, Z, np.ones(phi.shape, bool)))
    ok = got == want
    print(("ok  " if ok else "FAIL"), name, got, "expected", want)
    if not ok:
        failures.append(name)


for n in (33, 65, 129):
    h = 1 / ((n - 1) // 2)
    R = 2 + np.arange(-(n // 2), n // 2 + 1) * h; Z = np.arange(-(n // 2), n // 2 + 1) * h
    RR, ZZ = np.meshgrid(R, Z); x, z = RR - 2, ZZ
    expect(f"n{n} max off node", -((x - h / 2) ** 2 + (z - h / 2) ** 2), R, Z, (1, 0, 0))
    expect(f"n{n} max on node", -(x ** 2 + z ** 2), R, Z, (1, 0, 0))
    expect(f"n{n} min off node", (x - .3 * h) ** 2 + 2 * (z + .2 * h) ** 2, R, Z, (0, 1, 0))
    expect(f"n{n} saddle axis-aligned off node", (x - h / 2) ** 2 - (z - h / 2) ** 2, R, Z, (0, 0, 1))
    expect(f"n{n} saddle 2xz on node", 2 * x * z, R, Z, (0, 0, 1))
    expect(f"n{n} saddle shifted", 2 * (x - h / 2) * (z - h / 2), R, Z, (0, 0, 1))
    for deg in (10, 30, 45, 73):
        t = np.radians(deg); xr = np.cos(t) * (x - .17) + np.sin(t) * (z + .11); zr = -np.sin(t) * (x - .17) + np.cos(t) * (z + .11)
        expect(f"n{n} saddle rotated {deg}", xr ** 2 - 2 * zr ** 2, R, Z, (0, 0, 1))
        expect(f"n{n} ellipse max rotated {deg}", -(xr ** 2 + 3 * zr ** 2), R, Z, (1, 0, 0))
    two = np.exp(-((x) ** 2 + (z - .4) ** 2) / .05) + np.exp(-((x) ** 2 + (z + .4) ** 2) / .05)
    expect(f"n{n} two maxima and a saddle", two, R, Z, (2, 0, 1))
# The cubic counterexample (20 October): a cubic field in one cell; the truth from the exact derivatives.
h = 1 / 32
R = 2 + np.arange(-32, 33) * h; Z = np.arange(-32, 33) * h; RR, ZZ = np.meshgrid(R, Z); x, z = RR - 2, ZZ
cubic = x ** 3 * z - 2 * h ** 2 * x * z + .05 * h ** 2 * z ** 2 - .14 * h ** 3 * x + .79 * h ** 3 * z
cell = (RR >= 2 - 1e-12) & (RR <= 2 + h + 1e-12) & (ZZ >= -1e-12) & (ZZ <= h + 1e-12)
exact = []
for u0 in np.linspace(-30, 30, 61):
    for v0 in np.linspace(-30, 30, 61):
        xr, zr = u0 * h, v0 * h
        for _ in range(80):
            g = np.array([3 * xr ** 2 * zr - 2 * h ** 2 * zr - .14 * h ** 3, xr ** 3 - 2 * h ** 2 * xr + .1 * h ** 2 * zr + .79 * h ** 3])
            H = np.array([[6 * xr * zr, 3 * xr ** 2 - 2 * h ** 2], [3 * xr ** 2 - 2 * h ** 2, .1 * h ** 2]])
            try:
                xr, zr = np.array([xr, zr]) - np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
        if abs(xr) < 29 * h and abs(zr) < 29 * h and np.max(np.abs(g)) < 1e-15 and not any(
                abs(xr - a) < 1e-6 * h and abs(zr - c) < 1e-6 * h for a, c, _k in exact):
            det = 6 * xr * zr * .1 * h ** 2 - (3 * xr ** 2 - 2 * h ** 2) ** 2
            exact.append((xr, zr, "saddle" if det < 0 else ("max" if 6 * xr * zr + .1 * h ** 2 < 0 else "min")))
print("exact roots of the cubic field (u, v, kind):", [(round(a / h, 4), round(c / h, 4), k) for a, c, k in exact])
in_cell = [e for e in exact if 0 <= e[0] <= h and 0 <= e[1] <= h]
got = critical_points(cubic, R, Z, cell)
print(("ok  " if len(got) == len(in_cell) == 0 else "FAIL"), "review cubic counterexample: no exact root in its cell; found", got)
if len(got) != len(in_cell):
    failures.append("review cubic counterexample cell")
found = critical_points(cubic, R, Z, np.ones(cubic.shape, bool))
for xr, zr, kind in exact:
    match = [p for p in found if p[0] == kind and abs(p[1] - 2 - xr) < 1e-6 * h and abs(p[2] - zr) < 1e-6 * h]
    print(("ok  " if match else "FAIL"), "cubic root", round(xr / h, 4), round(zr / h, 4), kind)
    if not match:
        failures.append("cubic root")
if len(found) != len(exact):
    failures.append("cubic root count"); print("FAIL cubic count", len(found), len(exact))
R = np.linspace(0.84, 2.54, 65).astype(np.float32).astype(float); Z = np.linspace(-1.6, 1.6, 65).astype(np.float32).astype(float)
RR, ZZ = np.meshgrid(R, Z)  # a float32-stored grid, as the scorer's
inner = (np.hypot(RR - 1.7, ZZ) < 0.9)
pts = critical_points(-(RR ** 2 - 1.7 ** 2) ** 2 / 8 - .9 * ZZ ** 2, R, Z, inner)
print("Solov'ev", counts(pts), pts)
if counts(pts) != (1, 0, 0):
    failures.append("solovev")
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
