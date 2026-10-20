# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71b analytic checks of the added measures (exits non-zero on failure)
import sys

import numpy as np
import skimage.measure as sm
from scipy import ndimage

import physics
import physics_b

R = np.linspace(0.84, 2.54, 65); Z = np.linspace(-1.6, 1.6, 65); RR, ZZ = np.meshgrid(R, Z)


def bil(f, R, Z, rq, zq):
    fr = np.interp(rq, R, np.arange(len(R))); fz = np.interp(zq, Z, np.arange(len(Z)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(R) - 2); z0 = np.clip(np.floor(fz).astype(int), 0, len(Z) - 2)
    tr, tz = fr - r0, fz - z0
    return (f[z0, r0] * (1 - tr) * (1 - tz) + f[z0, r0 + 1] * tr * (1 - tz) + f[z0 + 1, r0] * (1 - tr) * tz
            + f[z0 + 1, r0 + 1] * tr * tz)


def pip(rq, zq, C):
    r1, z1 = C[:, 0], C[:, 1]; r2, z2 = np.roll(r1, -1), np.roll(z1, -1); out = np.zeros(rq.shape, bool)
    for e in range(len(r1)):
        if z1[e] != z2[e]:
            cross = (z1[e] > zq) != (z2[e] > zq)
            out ^= cross & (rq < r1[e] + (zq - z1[e]) * (r2[e] - r1[e]) / (z2[e] - z1[e]))
    return out


def erode(m, k):
    for _ in range(k):
        e = m.copy(); e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]; m = e
    return m


def o_point(phi, R, Z, mask):
    iz, ir = np.unravel_index(np.argmax(np.where(mask, phi, -np.inf)), phi.shape)
    return R[ir], Z[iz], iz, ir, phi[iz, ir]


kit = {"points_in_polygon": pip, "bilinear": bil, "find_o_point": o_point, "erode": erode}
env = np.hypot(RR - 1.7, ZZ) < 1.0
fails = []


def run(name, phi, level, expect):
    c = max(sm.find_contours(phi, level), key=len)
    C = np.column_stack([np.interp(c[:, 1], np.arange(65), R), np.interp(c[:, 0], np.arange(65), Z)])[:-1]
    mask = physics.inside(C, R, Z, pip)
    region = erode(env, 2) & ~physics.dilate(mask, 1)
    ref_in = float(physics_b.density(phi, R, Z, 1.0)[mask].sum())
    m = physics_b.measure_b(phi, R, Z, C, C, 1.0, env, kit, region, ref_in)
    print(name, {k: (round(v, 8) if isinstance(v, float) else v) for k, v in m.items()})
    for key, test in expect.items():
        if not test(m[key]):
            fails.append(f"{name}:{key}")


sol = -(RR ** 2 - 1.7 ** 2) ** 2 / 8 - .9 * ZZ ** 2
run("solovev", sol, -.3, {"gs_surface": lambda v: v < 1e-3, "gs_degenerate": lambda v: v == 0,
                           "cp_max": lambda v: v == 1, "cp_saddle": lambda v: v == 0,
                           "outside_fixed_over_ref": lambda v: v > 0})
const = -(RR ** 2 + ZZ ** 2)
run("constant source", const, -3.2, {"gs_degenerate": lambda v: v == 1, "gs_surface": lambda v: v < 1e-6})
t = np.arctan2(ZZ, RR - 1.7); rad = np.hypot(RR - 1.7, ZZ)
st = 1 + .06 * np.cos(3 * t) * np.exp(-np.clip(rad / .6 - 1, 0, None) ** 2 / .0625)
warp = ndimage.map_coordinates(sol, [((ZZ / st) - Z[0]) / (Z[1] - Z[0]), ((1.7 + (RR - 1.7) / st) - R[0]) / (R[1] - R[0])],
                               order=1, mode="nearest")
run("warped", warp, -.3, {"gs_surface": lambda v: v > 1e-2})
# The mask counterexample: one field, reference circle 0.4, own circle 0.6. With one common region
# and one denominator the reference and the prediction must give the same value (excess 0).
phi = -((RR - 2) ** 2 + ZZ ** 2)
t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
circle = lambda a: np.column_stack((2 + a * np.cos(t), a * np.sin(t)))  # noqa: E731
ref_c, own_c = circle(.4), circle(.6)
union = physics.inside(ref_c, R, Z, pip) | physics.inside(own_c, R, Z, pip)
envelope = np.hypot(RR - 2, ZZ) < .8
region = erode(envelope, 2) & ~physics.dilate(union, 1)
ref_in = float(physics_b.density(phi, R, Z, 1.0)[physics.inside(ref_c, R, Z, pip)].sum())
a = physics_b.measure_b(phi, R, Z, ref_c, ref_c, 1.0, envelope, kit, region, ref_in)["outside_fixed_over_ref"]
b = physics_b.measure_b(phi, R, Z, own_c, ref_c, 1.0, envelope, kit, region, ref_in)["outside_fixed_over_ref"]
print("mask counterexample: reference", a, "prediction", b, "excess", b - a)
if not (np.isfinite(a) and a == b):
    fails.append("mask counterexample")
zero = physics_b.gs_parts(np.zeros_like(RR), np.zeros_like(RR), R, np.ones_like(RR, bool), 1.0, 0.0)
print("zero source:", zero)
if zero["gs_degenerate"] != 1:
    fails.append("zero source not flagged")
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
