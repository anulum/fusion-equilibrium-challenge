# SCPN Reactor Systems — M3c-1b step 4b (fec venv): tune the coil freedom on TUNE, judge on TEST (protocol step4b, frozen)
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step34_testbed as tb
P45 = (("magnetics_p4u_current", "magnetics_p4l_current"), ("magnetics_p5u_current", "magnetics_p5l_current"))
P456 = P45 + (("magnetics_p6u_current",), ("magnetics_p6l_current",))
ARMS = {f'{cs}_s{sc}': (circ, sc) for cs, circ in (('P45', P45), ('P456', P456)) for sc in (0.01, 0.03, 0.1, 0.3)}
def init():
    tb._init()
    arms = {'v7': tb._W['arms']['v7']}
    for name, (circ, sc) in ARMS.items():
        mod = tb._module('thomson_isoflux', tb.FREE, f'ti_{name}')
        mod.FREE_CIRCUITS, mod.FREE_SCALE, mod.FREE_FLOOR_SHARE = circ, sc, 0.1
        arms[name] = mod
    tb._W['arms'] = arms
def main():
    frames = json.loads((HERE / 'step3_frames.json').read_text())
    rows = {(r['shot'], r['frame']): r for r in json.loads((HERE / 'step3_rows.json').read_text())}
    with np.load(HERE / 'step3_truth.npz') as z:
        tasks = [(rows[(f['shot'], f['frame'])], f, z[k].astype(np.float64) if (k := f"shot_{f['shot']:04d}_frame_{f['frame']:03d}") in z.files else None) for f in frames]
    with Pool(4, initializer=init) as pool:
        recs = pool.map(tb.run, tasks, chunksize=2)
    names = ['v7', *ARMS]
    summ = {f'{sp}/{v}/{a}': tb.summarise([r for r in recs if r['split'] == sp], v, a) for sp in ('TUNE', 'TEST') for v in ('exact', 'measured') for a in names}
    base_exact = summ['TUNE/exact/v7']['D_mean']
    ok = [a for a in ARMS if summ[f'TUNE/exact/{a}']['D_mean'] <= 1.05 * base_exact]
    chosen = min(ok, key=lambda a: summ[f'TUNE/measured/{a}']['D_mean']) if ok else None
    out = {'tune_table': {a: {'measured_D': summ[f'TUNE/measured/{a}']['D_mean'], 'exact_D': summ[f'TUNE/exact/{a}']['D_mean']} for a in names},
           'chosen_on_TUNE': chosen}
    if chosen:
        test = [r for r in recs if r['included'] and r['split'] == 'TEST']
        shots = sorted({r['shot'] for r in test}); idx = {s: i for i, s in enumerate(shots)}
        dv, dc, n = np.zeros(len(shots)), np.zeros(len(shots)), np.zeros(len(shots))
        for r in test:
            i = idx[r['shot']]; dv[i] += r['measured/v7']['D']; dc[i] += r[f'measured/{chosen}']['D']; n[i] += 1
        rng = np.random.default_rng(934)
        w = np.stack([np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)) for _ in range(20000)])
        boot = (w @ dc) / (w @ n) - (w @ dv) / (w @ n)
        tm, tc = summ['TEST/measured/v7'], summ[f'TEST/measured/{chosen}']
        xe, xc = summ['TEST/exact/v7'], summ[f'TEST/exact/{chosen}']
        rule = {'D_measured': [tm['D_mean'], tc['D_mean']], 'D_rel_change_measured': tc['D_mean'] / tm['D_mean'] - 1,
                'P_D_lower': float(np.mean(boot < 0)), 'kappa_bias': [tm['bias']['kappa'], tc['bias']['kappa']],
                'D_exact': [xe['D_mean'], xc['D_mean']], 'cons_exact': [xe['consistency'], xc['consistency']],
                'cons_measured': [tm['consistency'], tc['consistency']], 'flux_measured': [tm['flux_r2'], tc['flux_r2']],
                'gap_recovered': (tm['D_mean'] - tc['D_mean']) / (tm['D_mean'] - xe['D_mean'])}
        rule['passes'] = bool(rule['D_rel_change_measured'] <= -0.10 and rule['P_D_lower'] >= 0.9 and abs(tc['bias']['kappa']) <= abs(tm['bias']['kappa'])
                              and xc['D_mean'] <= 1.05 * xe['D_mean'] and xc['consistency'] >= xe['consistency'] - 0.02)
        out['test_rule'] = rule
    (HERE / 'step4b_result.json').write_text(json.dumps({'out': out, 'summary': summ}, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)
if __name__ == '__main__':
    main()
