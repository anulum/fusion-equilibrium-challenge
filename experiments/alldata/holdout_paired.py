# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — paired per-shot comparison of two holdout evaluations (shot bootstrap)
"""Denominators per term from candidate A's pooled SSE and its printed R2 (fixed); per-shot deltas of each
term's contribution to S; 20000-resample shot bootstrap of the S difference (seed 930)."""
import json, re, sys
import numpy as np
D = sys.argv[1]; a, b = sys.argv[2], sys.argv[3]
def load(t): return {r['stem']: r for r in json.load(open(f'{D}/{t}/per_shot.json'))}
def r2s(t):
    txt = open(f'{D}/{t}.log').read()
    names = ['R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li']
    der = [float(re.search(rf'\n\s+{n}\s+([0-9.]+)', txt).group(1)) for n in names]
    flux = float(re.search(r'flux R2 ([0-9.]+)', txt).group(1))
    return flux, der
A, B = load(a), load(b); stems = sorted(A); assert sorted(B) == stems
fluxA = np.array([A[s]['flux'] for s in stems]); fluxB = np.array([B[s]['flux'] for s in stems])
derA = np.array([A[s]['derived'] for s in stems]); derB = np.array([B[s]['derived'] for s in stems])
scA = np.array([A[s]['scalars'] for s in stems]); scB = np.array([B[s]['scalars'] for s in stems])
dA = np.array([A[s]['boundary_distance'] for s in stems]); dB = np.array([B[s]['boundary_distance'] for s in stems])
f2, d2 = r2s(a)
Tf = fluxA.sum() / (1 - f2); Td = derA.sum(0) / (1 - np.array(d2))
# scalar denominators: from B's own log is not printed per scalar; use the pooled q/b R2 of a via combined line
txt = open(f'{D}/{a}.log').read(); sc = float(re.search(r'scalars ([0-9.]+)', txt).group(1))
# assume equal R2 split is unknown -> use pooled per-scalar variance proxy: T_k = SSE_k / (1 - sc)
Ts = scA.sum(0) / (1 - sc)
n = len(stems)
def contrib(idx):
    flux = -0.55 * (fluxB[idx].sum() - fluxA[idx].sum()) / Tf * n / len(idx)
    sca = -0.15 * np.mean((scB[idx].sum(0) - scA[idx].sum(0)) / Ts) * n / len(idx)
    bnd = -0.10 * (dB[idx].mean() - dA[idx].mean())
    der = -0.20 * np.mean((derB[idx].sum(0) - derA[idx].sum(0)) / Td) * n / len(idx)
    return flux, sca, bnd, der
full = contrib(np.arange(n))
rng = np.random.default_rng(930); boots = np.array([sum(contrib(rng.integers(0, n, n))) for _ in range(20000)])
perder = -0.20 / 7 * (derB.sum(0) - derA.sum(0)) / Td
print(json.dumps({'pair': f'{b} - {a}', 'delta_S': sum(full), 'terms': dict(zip(('flux', 'q95_betaN', 'boundary', 'consistency'), full)),
                  'consistency_per_scalar': dict(zip(['R_axis', 'Z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume', 'li'], perder.round(7).tolist())),
                  'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'p_better': float((boots > 0).mean()),
                  'shots_better_flux': int((fluxB < fluxA).sum()), 'shots': n}, indent=1))
