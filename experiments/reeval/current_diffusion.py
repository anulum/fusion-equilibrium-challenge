# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — input-only current-diffusion estimate of the internal inductance
"""A 1-D cylindrical poloidal-field diffusion driven by the measured plasma current and the core
Thomson electron temperature, giving an input-only estimate of the current-profile peaking.

The internal inductance is the peaking of the current profile, and the profile is set by resistive
diffusion over hundreds of milliseconds: the plasma resistivity follows Spitzer (T_e^-3/2), the edge
field follows the plasma current. The model is deliberately crude and label-free:

* the core Thomson laser is vertical (R = 1.94 m on DIII-D); a profile's centre is the T_e^2-weighted
  mean height and its half-height the furthest channel above 10 % of the peak; rho = |Z - Z0| / b;
* T_e(rho) from the channels of both sides, binned on NR cells, held between Thomson frames;
* eta = 5.6e-8 * T_e[keV]^-1.5 Ohm m (Zeff 2, ln Lambda 17), floor T_e 20 eV;
* cylinder radius A (fixed), implicit Euler for B_theta(r, t) on a 5 ms grid from the first frame with
  |Ip| >= 100 kA, flat initial current;
* outputs per step: cylindrical li = 2 int B_theta^2 r dr / (a^2 B_theta(a)^2), the same for the
  relaxed profile j ~ T_e^1.5 (no history), and j(0)/<j>.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi
NR = 40
A = 0.7
DT = 5e-3
TE_FLOOR_KEV = 0.02


def _profile(z, te):
    """Te on the rho grid from one Thomson frame, or None."""
    ok = np.isfinite(te) & (te > 0)
    if ok.sum() < 8:
        return None
    z, te = z[ok], te[ok]
    peak = te.max()
    if peak < 50:
        return None
    w = te ** 2
    z0 = float(np.sum(w * z) / np.sum(w))
    hot = te > 0.1 * peak
    b = float(np.max(np.abs(z[hot] - z0))) if hot.any() else 0.0
    if b < 0.1:
        return None
    rho = np.abs(z - z0) / b
    order = np.argsort(rho)
    rho, te = rho[order], te[order]
    grid = (np.arange(NR) + 0.5) / NR
    prof = np.interp(grid, rho, te, left=te[0], right=te[-1])
    return np.maximum(prof / 1000.0, TE_FLOOR_KEV)


def simulate(ip_t, ip_ka, th_t, th_z, th_te, out_t):
    """Return (li_hist, li_relaxed, peaking) at out_t (ms). Units: ip in kA, te in eV, z in m."""
    out = np.full((len(out_t), 3), np.nan)
    ok = np.isfinite(ip_ka)
    if ok.sum() < 10 or th_te is None or len(th_t) == 0:
        return out
    ip_t, ip_ka = ip_t[ok], ip_ka[ok]
    live = np.nonzero(np.abs(ip_ka) >= 100.0)[0]
    if live.size == 0:
        return out
    t0, t1 = ip_t[live[0]], max(float(np.max(out_t)), ip_t[live[0]])
    steps = np.arange(t0, t1 + DT * 1e3, DT * 1e3)
    ip = np.abs(np.interp(steps, ip_t, ip_ka)) * 1e3
    profiles = [(_profile(th_z, te)) for te in th_te]
    have = [i for i, p in enumerate(profiles) if p is not None]
    if not have:
        return out
    th_times = np.asarray(th_t)[have]
    th_prof = np.asarray([profiles[i] for i in have])
    dr = A / NR
    r_face = np.arange(1, NR + 1) * dr            # outer faces
    r_c = (np.arange(NR) + 0.5) * dr               # cell centres
    # state: enclosed current I(r_face) ; B_theta = mu0 I / (2 pi r)
    enclosed = ip[0] * (r_face / A) ** 2           # flat initial current
    rec_t, rec = [], []
    for k, t in enumerate(steps):
        idx = np.searchsorted(th_times, t, side='right') - 1
        te = th_prof[max(idx, 0)]
        eta = 5.6e-8 * te ** -1.5                   # at cell centres
        # j at cell centres from enclosed current; E_z = eta j must be uniform in steady state.
        # Diffusion of the enclosed current: dI/dt = (2 pi r / mu0) d/dr (eta_face * (1/r) d(r ... )) —
        # use the flux form on psi: d psi/dt = eta * j, j = (1/(2 pi r)) dI/dr.
        area = np.pi * (r_face ** 2 - np.concatenate(([0.0], r_face[:-1])) ** 2)
        j = np.diff(np.concatenate(([0.0], enclosed))) / area
        # implicit step on j: dj/dt = (1/mu0 r) d/dr (r d(eta j)/dr) with the edge fixed by Ip
        n = NR
        lam = DT / MU0
        main = np.zeros(n); lo = np.zeros(n - 1); up = np.zeros(n - 1)
        for i in range(n):
            rw = r_face[i - 1] if i > 0 else 0.0
            re = r_face[i]
            vol = r_c[i] * dr
            cw = lam * rw / (vol * dr) if i > 0 else 0.0
            ce = lam * re / (vol * dr) if i < n - 1 else 0.0
            main[i] = 1.0 + (cw + ce) * eta[i]
            if i > 0:
                lo[i - 1] = -cw * eta[i - 1]
            if i < n - 1:
                up[i] = -ce * eta[i + 1]
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n)); ab[0, 1:] = up; ab[1] = main; ab[2, :-1] = lo
        j_new = solve_banded((1, 1), ab, j)
        j_new = np.clip(j_new, 0.0, None)
        total = float(np.sum(j_new * area))
        if total > 0:
            j_new *= ip[k] / total
        enclosed = np.cumsum(j_new * area)
        b_theta = MU0 * enclosed / (2 * np.pi * r_face)
        li_hist = 2 * np.sum(b_theta ** 2 * r_face * dr) / (A ** 2 * b_theta[-1] ** 2)
        jr = te ** 1.5
        enc_r = np.cumsum(jr * area)
        b_r = enc_r / r_face
        li_relaxed = 2 * np.sum(b_r ** 2 * r_face * dr) / (A ** 2 * b_r[-1] ** 2)
        peaking = float(j_new[0] / (total / (np.pi * A ** 2))) if total > 0 else np.nan
        rec_t.append(t); rec.append((li_hist, li_relaxed, peaking))
    rec_t, rec = np.asarray(rec_t), np.asarray(rec)
    for c in range(3):
        out[:, c] = np.interp(out_t, rec_t, rec[:, c], left=np.nan, right=np.nan)
    return out
