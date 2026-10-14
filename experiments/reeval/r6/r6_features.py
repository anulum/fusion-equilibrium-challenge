# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — R6 physics block P: input-only features aimed at li and kappa
"""Per reconstruction frame, from the reader's aligned inputs (fec_adapter.shot_reader.read_shot, the same
alignment the model's features use) and the raw plasma-current trace; no efit column is read.

Why each group (li is the current-profile peaking, kappa follows li and the shaping field):

core profile (vertical laser at R 1.94 m): Te and ne peak values, peaking (peak over mean), Te-weighted centre
    and half-width in Z, the pressure p = ne Te integrated over Z (a kinetic energy proxy) and its peaking;
beta_p proxy: the Z-integrated pressure over Ip^2 — the equilibrium's vertical field sets beta_p + li/2 from the
    coil currents, so an independent beta_p estimate lets a model separate li from beta_p;
midplane profile (the radial laser, R 1.67-2.06 m): peak Te and p, the largest outward Te and p gradients and
    their radii (pedestal height and position), the pressure integrated over R;
current history (on the reader's corrected plasma-current clock — the release's is ~3 s early on most
    shots, see fec_adapter.machine.plasma_current_times): exponential means of Ip with 0.1, 0.3, 1 and 3 s memory relative to Ip now (the current profile
    relaxes on the resistive time, so a recent ramp leaves the profile hollow or peaked), dIp/dt over 100 ms;
conductivity history: exponential means of Te_peak^1.5 over 0.3 and 1 s relative to now;
current diffusion: li_hist, li_relaxed and peaking of the 1-D model in reeval/current_diffusion.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fec_adapter import machine
from fec_adapter.shot_reader import read_row, read_shot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import current_diffusion as cd  # noqa: E402

DATA = Path('$FEC_DATA/data/diii_d_train')
EV = 1.602176634e-19
TAUS_IP = (100.0, 300.0, 1000.0, 3000.0)
TAUS_TE = (300.0, 1000.0)
NAMES = (['core_te_peak', 'core_ne_peak', 'core_te_peaking', 'core_ne_peaking', 'core_z0', 'core_te_halfwidth',
          'core_p_int', 'core_p_peaking', 'betap_proxy', 'core_te_int_over_ip',
          'mid_te_peak', 'mid_p_peak', 'mid_te_grad_max', 'mid_te_grad_r', 'mid_p_grad_max', 'mid_p_grad_r', 'mid_p_int']
         + [f'ip_ema{int(t)}_rel' for t in TAUS_IP] + ['dip_dt_100ms']
         + [f'te15_ema{int(t)}_rel' for t in TAUS_TE] + ['li_hist', 'li_relaxed', 'j_peaking'])


def _ema(times, values, at, tau):
    """Exponential mean with memory tau (ms) of a trace sampled at `times`, evaluated at `at`."""
    out = np.full(len(at), np.nan)
    order = np.argsort(times)
    t, v = times[order], values[order]
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < 2:
        return out
    acc, last, j = v[0], t[0], 0
    for i, ti in enumerate(at):
        while j < t.size and t[j] <= ti:
            w = np.exp(-(t[j] - last) / tau)
            acc = w * acc + (1 - w) * v[j]
            last = t[j]
            j += 1
        out[i] = acc
    return out


def features(stem: str) -> np.ndarray:
    """Return the (frames, len(NAMES)) block for one shot."""
    shot = read_shot(DATA / f'{stem}.parquet')
    th = shot.thomson
    n = shot.n_frames
    ip = np.abs(shot.plasma_current)                                   # A
    ip_ma = np.maximum(ip / 1e6, 0.05)
    out = np.full((n, len(NAMES)), np.nan)
    core = np.abs(th.r - 1.94) < 0.005
    mid = ~core
    zc, rm = th.z[core], th.r[mid]
    zo, ro = np.argsort(zc), np.argsort(rm)
    for f in range(n):
        te, ne = th.te[f, core][zo], th.ne[f, core][zo]
        z = zc[zo]
        ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        if ok.sum() >= 5:
            t, e, zz = te[ok], ne[ok], z[ok]
            p = e * t * EV
            z0 = float(np.sum(t ** 2 * zz) / np.sum(t ** 2))
            pint = float(np.trapezoid(p, zz))
            span = max(float(zz.max() - zz.min()), 1e-3)
            out[f, 0:10] = [t.max(), e.max() / 1e19, t.max() / t.mean(), e.max() / e.mean(), z0,
                            float(np.sqrt(np.sum(t * (zz - z0) ** 2) / np.sum(t))), pint / 1e3,
                            p.max() / (pint / span), pint / 1e3 / ip_ma[f] ** 2,
                            float(np.trapezoid(t, zz)) / 1e3 / ip_ma[f]]
        te, ne = th.te[f, mid][ro], th.ne[f, mid][ro]
        r = rm[ro]
        ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        if ok.sum() >= 4:
            t, e, rr = te[ok], ne[ok], r[ok]
            p = e * t * EV
            gt, gp = -np.gradient(t, rr), -np.gradient(p, rr)
            out[f, 10:17] = [t.max(), p.max() / 1e3, gt.max() / 1e3, rr[np.argmax(gt)], gp.max() / 1e6,
                             rr[np.argmax(gp)], float(np.trapezoid(p, rr)) / 1e3]
    row = pq.read_table(DATA / f'{stem}.parquet', columns=['magnetics_plasma_current', 'magnetics_plasma_current_times',
                                                            'thomson_core_times', 'thomson_core_Te', 'thomson_chord_name',
                                                            'thomson_chord_Z', 'efit_times'])
    # the release's plasma-current clock is ~3 s early on most shots; use the reader's corrected clock (erratum)
    it = np.asarray(machine.plasma_current_times(read_row(DATA / f'{stem}.parquet')), float)
    iv = np.abs(np.asarray(row['magnetics_plasma_current'][0].as_py(), float))
    at = shot.times_ms
    now = np.interp(at, it, iv)
    for k, tau in enumerate(TAUS_IP):
        out[:, 17 + k] = (_ema(it, iv, at, tau) - now) / np.maximum(now, 50.0)
    out[:, 21] = (now - np.interp(at - 100.0, it, iv)) / 100.0
    te15 = np.where(np.isfinite(out[:, 0]), np.maximum(out[:, 0], 20.0) / 1e3, np.nan) ** 1.5
    for k, tau in enumerate(TAUS_TE):
        out[:, 22 + k] = _ema(at, te15, at, tau) / np.maximum(te15, 1e-3) - 1.0
    names = list(row['thomson_chord_name'][0].as_py())
    zch = np.asarray(row['thomson_chord_Z'][0].as_py(), float)
    tcore = np.asarray(row['thomson_core_Te'][0].as_py(), float)
    zcore = zch[[i for i, s in enumerate(names) if s.startswith('TS_core')]][:tcore.shape[1]]
    out[:, 24:27] = cd.simulate(it, iv, np.asarray(row['thomson_core_times'][0].as_py(), float), zcore, tcore, at)
    return out
