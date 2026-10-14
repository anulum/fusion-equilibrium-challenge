# SCPN Reactor Systems — M3c-1b step 8 A/B (fec venv): VP_s0.1r on a testbed, all included frames (protocol step8, frozen)
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('STEP_SRC', str(HERE.parent)))
SEED = int(os.environ.get('STEP_SEED', '936'))
ONLY = os.environ.get('STEP_ONLY')  # step 6c: a single pre-chosen arm, judged on all included frames
sys.path.insert(0, str(HERE.parent))
import step34_testbed as tb
VES = HERE.parent / 'entrypoints_v7_vessel'
P45 = (("magnetics_p4u_current", "magnetics_p4l_current"), ("magnetics_p5u_current", "magnetics_p5l_current"))
ARMS = {'P45': (P45, 0.0)}
for s in (0.03, 0.1, 0.3):
    ARMS[f'V_s{s}'] = ((), s)
    ARMS[f'VP_s{s}'] = (P45, s)
ARMS = {'VP_s0.1r': (P45, 0.1)}
ONLY = 'VP_s0.1r'
def init():
    tb._init()
    arms = {'v7': tb._W['arms']['v7']}
    for name, (circ, vs) in ARMS.items():
        mod = tb._module('thomson_isoflux', VES, f'ti_{name}')
        mod.FREE_CIRCUITS, mod.FREE_SCALE, mod.FREE_FLOOR_SHARE, mod.FREE_BOUND = circ, 0.03, 0.1, 0.10
        mod.VESSEL_SCALE, mod.VESSEL_BOUND, mod.SHIFT_RELAXATION = vs, 0.5, 0.3
        arms[name] = mod
    tb._W['arms'] = arms
def judge(recs, summ, arm, split):
    sel = [r for r in recs if r['included'] and (split == 'ALL' or r['split'] == split)]
    shots = sorted({r['shot'] for r in sel}); idx = {s: i for i, s in enumerate(shots)}
    dv, dc, n = np.zeros(len(shots)), np.zeros(len(shots)), np.zeros(len(shots))
    for r in sel:
        i = idx[r['shot']]; dv[i] += r['measured/v7']['D']; dc[i] += r[f'measured/{arm}']['D']; n[i] += 1
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(rng.integers(0, len(shots), len(shots)), minlength=len(shots)) for _ in range(20000)])
    boot = (w @ dc) / (w @ n) - (w @ dv) / (w @ n)
    tm, tc = summ[f'{split}/measured/v7'], summ[f'{split}/measured/{arm}']
    xe, xc = summ[f'{split}/exact/v7'], summ[f'{split}/exact/{arm}']
    rule = {'frames': len(sel), 'D_measured': [tm['D_mean'], tc['D_mean']], 'D_rel_change_measured': tc['D_mean'] / tm['D_mean'] - 1,
            'P_D_lower': float(np.mean(boot < 0)), 'kappa_bias': [tm['bias']['kappa'], tc['bias']['kappa']],
            'D_exact': [xe['D_mean'], xc['D_mean']], 'cons_exact': [xe['consistency'], xc['consistency']],
            'cons_measured': [tm['consistency'], tc['consistency']], 'flux_measured': [tm['flux_r2'], tc['flux_r2']]}
    rule['passes'] = bool(rule['D_rel_change_measured'] <= -0.10 and rule['P_D_lower'] >= 0.9 and abs(tc['bias']['kappa']) <= abs(tm['bias']['kappa'])
                          and xc['D_mean'] <= 1.05 * xe['D_mean'] and xc['consistency'] >= xe['consistency'] - 0.02
                          and (split != 'ALL' or len(sel) >= 100))
    return rule
def main():
    frames = json.loads((SRC / 'step3_frames.json').read_text())
    rows = {(r['shot'], r['frame']): r for r in json.loads((HERE / 'step6_rows.json').read_text())}
    with np.load(HERE / 'step6_truth.npz') as z:
        tasks = [(rows[(f['shot'], f['frame'])], f, z[k].astype(np.float64) if (k := f"shot_{f['shot']:04d}_frame_{f['frame']:03d}") in z.files else None) for f in frames]
    with Pool(4, initializer=init) as pool:
        recs = pool.map(tb.run, tasks, chunksize=2)
    names = ['v7', *ARMS]
    splits = ('ALL',) if ONLY else ('TUNE', 'TEST', 'ALL')
    summ = {f'{sp}/{v}/{a}': tb.summarise([r for r in recs if sp == 'ALL' or r['split'] == sp], v, a) for sp in splits for v in ('exact', 'measured') for a in names}
    out = {'included': int(sum(r['included'] for r in recs)), 'frames': len(recs)}
    if ONLY:
        out['confirm'] = judge(recs, summ, ONLY, 'ALL')
    else:
        base = summ['TUNE/exact/v7']['D_mean']
        cand = [a for a in ARMS if a != 'P45' and summ[f'TUNE/exact/{a}']['D_mean'] <= 1.05 * base]
        out['tune_table'] = {a: {'measured_D': summ[f'TUNE/measured/{a}']['D_mean'], 'exact_D': summ[f'TUNE/exact/{a}']['D_mean']} for a in names}
        out['chosen_on_TUNE'] = min(cand, key=lambda a: summ[f'TUNE/measured/{a}']['D_mean']) if cand else None
        if out['chosen_on_TUNE']:
            out['test_rule'] = judge(recs, summ, out['chosen_on_TUNE'], 'TEST')
            out['test_P45_for_reference'] = judge(recs, summ, 'P45', 'TEST')
    (HERE / 'step8_result.json').write_text(json.dumps({'out': out, 'summary': summ}, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)
if __name__ == '__main__':
    main()
