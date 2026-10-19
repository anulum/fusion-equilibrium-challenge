# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D71 operator check on an analytic Grad-Shafranov solution (no data)
import numpy as np
import skimage.measure as sm

import physics

R = np.linspace(0.84, 2.54, 65); Z = np.linspace(-1.6, 1.6, 65)
RR, ZZ = np.meshgrid(R, Z); R0, b = 1.7, 0.9
phi = -(RR ** 2 - R0 ** 2) ** 2 / 8 - b * ZZ ** 2          # Delta* phi = -R^2 - 2b: constant p', FF'


def bil(f, R, Z, rq, zq):
    fr = np.interp(rq, R, np.arange(len(R))); fz = np.interp(zq, Z, np.arange(len(Z)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(R) - 2); z0 = np.clip(np.floor(fz).astype(int), 0, len(Z) - 2)
    tr, tz = fr - r0, fz - z0
    return (f[z0, r0] * (1 - tr) * (1 - tz) + f[z0, r0 + 1] * tr * (1 - tz)
            + f[z0 + 1, r0] * (1 - tr) * tz + f[z0 + 1, r0 + 1] * tr * tz)


def pip(rq, zq, C):
    r1, z1 = C[:, 0], C[:, 1]; r2, z2 = np.roll(r1, -1), np.roll(z1, -1); out = np.zeros(rq.shape, bool)
    for e in range(len(r1)):
        if z1[e] != z2[e]:
            cross = (z1[e] > zq) != (z2[e] > zq)
            out ^= cross & (rq < r1[e] + (zq - z1[e]) * (r2[e] - r1[e]) / (z2[e] - z1[e]))
    return out


c = max(sm.find_contours(phi, -0.3), key=len)
C = np.column_stack([np.interp(c[:, 1], np.arange(65), R), np.interp(c[:, 0], np.arange(65), Z)])[:-1]
pr, pz, gs = physics.derivatives(phi, R, Z)
print("max |Delta* - analytic|", float(np.nanmax(np.abs(gs - (-(RR ** 2) - 2 * b)))))
m = physics.inside(C, R, Z, pip)
ax, bv = float(phi.max()), float(np.mean(bil(phi, R, Z, C[:, 0], C[:, 1])))
print("gs_fraction exact", physics.gs_form_residual(phi, gs, R, m, ax, bv))
o4 = physics.laplacian_star_o4(phi, R, Z)
print("max |Delta* o4 - analytic|", float(np.nanmax(np.abs(o4 - (-(RR ** 2) - 2 * b)))))
print("gs_fraction exact o4, deg2, deg5", physics.gs_form_residual(phi, o4, R, m, ax, bv),
      physics.gs_form_residual(phi, gs, R, m, ax, bv, 2), physics.gs_form_residual(phi, gs, R, m, ax, bv, 5))
a, s = physics.loop_current(phi, R, Z, C, bil)
area = float((-gs / RR)[m].sum() * np.mean(np.diff(R)) * np.mean(np.diff(Z)))
print("area/-signed", area / -s, "loop_abs/-signed", a / -s)
bump = phi + 0.02 * np.exp(-((RR - 2.0) ** 2 + (ZZ - 0.3) ** 2) / 0.02)
_, _, gs2 = physics.derivatives(bump, R, Z)
print("gs_fraction with a non-GS bump", physics.gs_form_residual(bump, gs2, R, m, ax, bv))
env = np.ones_like(phi, bool)


def lmax(p, mask, e):
    q = np.where(np.isfinite(p), p, -np.inf); out = np.zeros_like(q, bool); c = q[1:-1, 1:-1]; ge = np.ones_like(c, bool)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dz or dr:
                ge &= c >= q[1 + dz:q.shape[0] - 1 + dz, 1 + dr:q.shape[1] - 1 + dr]
    out[1:-1, 1:-1] = ge; out[:e] = out[-e:] = False; out[:, :e] = out[:, -e:] = False
    return out & mask


print("critical points exact (max, min, saddle)", physics.critical_points(phi, env, lmax))
xpt = phi - 0.6 * np.exp(-((RR - 1.7) ** 2 + (ZZ + 1.1) ** 2) / 0.05) + 0.25 * np.exp(-((RR - 1.7) ** 2 + (ZZ + 1.3) ** 2) / 0.02)
print("critical points with a second maximum", physics.critical_points(xpt, env, lmax))
