# SPDX-License-Identifier: AGPL-3.0-or-later
# Equilibrium Challenge — D72 check of the projection on an analytic equilibrium and on a warped copy (no data)
import numpy as np
import skimage.measure as sm
from scipy import ndimage

import gs_project
import physics

R = np.linspace(0.84, 2.54, 65); Z = np.linspace(-1.6, 1.6, 65)
RR, ZZ = np.meshgrid(R, Z); R0, b = 1.7, 0.9
phi = -(RR ** 2 - R0 ** 2) ** 2 / 8 - b * ZZ ** 2


def pip(rq, zq, C):
    r1, z1 = C[:, 0], C[:, 1]; r2, z2 = np.roll(r1, -1), np.roll(z1, -1); out = np.zeros(rq.shape, bool)
    for e in range(len(r1)):
        if z1[e] != z2[e]:
            cross = (z1[e] > zq) != (z2[e] > zq)
            out ^= cross & (rq < r1[e] + (zq - z1[e]) * (r2[e] - r1[e]) / (z2[e] - z1[e]))
    return out


def lcfs(field, level):
    c = max(sm.find_contours(field, level), key=len)
    return np.column_stack([np.interp(c[:, 1], np.arange(65), R), np.interp(c[:, 0], np.arange(65), Z)])[:-1]


def check(field, name):
    C = lcfs(field, -0.3)
    mask = physics.inside(C, R, Z, pip)
    _, _, gs = physics.derivatives(field, R, Z)
    iz, ir = np.unravel_index(np.argmax(np.where(mask, field, -np.inf)), field.shape)
    out, info = gs_project.projected(field, R, Z, mask, 1.0, (iz, ir), -0.3, gs)
    _, _, gs2 = physics.derivatives(out, R, Z)
    before = physics.gs_form_residual(field, gs, R, mask, field[iz, ir], -0.3)
    after = physics.gs_form_residual(out, gs2, R, mask, out[mask].max(), -0.3)
    drop = field[iz, ir] + 0.3
    print(name, info, "max change / flux drop", float(np.max(np.abs(out - field)) / drop),
          "outside unchanged", bool(np.array_equal(out[~mask], field[~mask])), "GS share before/after", before, after)
    return out


check(phi, "exact")
theta = np.arctan2(ZZ, RR - R0); radius = np.hypot(RR - R0, ZZ)
stretch = 1 + 0.06 * np.cos(3 * theta) * np.exp(-np.clip(radius / 0.6 - 1, 0, None) ** 2 / 0.25 ** 2)
warped = ndimage.map_coordinates(phi, [((ZZ / stretch) - Z[0]) / (Z[1] - Z[0]), ((R0 + (RR - R0) / stretch) - R[0]) / (R[1] - R[0])],
                                 order=1, mode="nearest")
check(warped, "warped")
