# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D15-0 gate: networks refitted with the label error of termination frames weighted up
"""K-fold refit of four S16 members with tail-weighted label loss (d15_tail_weighted_nets/protocol_d15_0_gate.json).

The fit is fec_adapter.diii_d_accuracy.fit_on_accelerators unchanged except for the row weights of the label error:
a frame of the tail regime (fewer than 25 frames to the end of its record, or Ip below 0.6 of the discharge peak,
or less than 0.3 s since the discharge start: the D14 definition) weighs TAIL_WEIGHT times a frame outside it, the
weights scaled to mean one. The flux coordinates' error is not reweighted (the fit's own row-weight path).
For each fold the four networks then answer together on the frames of the fold's held-out training shots, and the
result is written in the layout of the D9 gate (gbm_oof_fold{k}.npz), so d9-style nested blending applies.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FOLDS = __FOLDS__
RECIPE = [0, 22, 24, 31]
TAIL_WEIGHT = 5.0
TARGETS = ['q95', 'beta_n', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume',
           'x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
OUT = Path('/kaggle/working')

WRAPPER = '''
import sys
import numpy as np
from fec_adapter.diii_d_accuracy import fit_on_accelerators as fit

TAIL_WEIGHT = float(sys.argv.pop(1))


def tail_weights(training):
    shot = np.asarray(training.shot_of_row)
    n = len(shot)
    last = np.r_[shot[1:] != shot[:-1], True]
    from_end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1] - np.arange(n)
    features = training.features
    tail = (from_end < 25) | (features[:, 62] < 0.6) | (features[:, 64] < 0.3)
    weights = np.where(tail, TAIL_WEIGHT, 1.0)
    print({"tail_share": float(tail.mean()), "weight_tail": float(TAIL_WEIGHT / weights.mean()),
           "weight_rest": float(1.0 / weights.mean()), "rows": int(n)}, flush=True)
    return (weights / weights.mean()).astype(np.float32)


def balance_on(training, settings, device):
    assert settings.shot_balance == 0.0
    return fit.on_device(tail_weights(training), device)


fit._balance_on = balance_on
raise SystemExit(fit.main())
'''


def sh(command):
    print('$', command, flush=True)
    return subprocess.run(command, shell=True).returncode


def find(pattern):
    return sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))


def main():
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
    assert sh('pip install -q "jax[cuda12]==0.8.2"') == 0
    kdir = os.path.dirname([p for p in find('kfold_run.py') if '/kfold/' in p][0])
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(find('fec_adapter/diii_d_accuracy/fit_on_accelerators.py')[0])))
    lcfs = os.path.dirname(find('table_training.npz')[0])
    for line in open(os.path.join(lcfs, 'SHA256SUMS')).read().splitlines():
        want, name = line.split()
        h = hashlib.sha256()
        with open(os.path.join(lcfs, name), 'rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                h.update(block)
        assert h.hexdigest() == want, name
    env = dict(os.environ, PYTHONPATH=pkg, PYTHONUNBUFFERED='1', JAX_ENABLE_X64='1')
    sys.path.insert(0, kdir); sys.path.insert(0, pkg)
    import numpy as np
    import kfold_run as kr
    from fec_adapter.diii_d_accuracy.network import Network, predict_together
    protocol = json.loads(Path(kdir, 'protocol_kfold_oof_maps_20260929.json').read_text())
    splits = json.loads(Path(kdir, 'splits.json').read_text())
    reserved = set(json.loads(Path(kdir, 'reserved_for_evaluation.json').read_text())['stems'])
    members = [m for m in protocol['members'] if m['recipe_index'] in RECIPE]
    assert len(members) == len(RECIPE), [m['name'] for m in members]
    work = Path('/tmp/work'); work.mkdir(parents=True, exist_ok=True)
    wrapper = work / 'd15_fit.py'; wrapper.write_text(WRAPPER)
    with np.load(os.path.join(lcfs, 'table_training.npz'), allow_pickle=False) as t:
        features, stems, shot_of_row = t['features'], t['stems'].astype(str), t['shot_of_row']
        layout = str(t['feature_layout'])
    stem_fold = np.array([-1 if s in reserved else kr.fold_of(s, protocol['folds']) for s in stems])
    row_fold = stem_fold[shot_of_row]
    for k in FOLDS:
        started = time.time()
        held = sorted(s for s in splits['training'] if s not in reserved and kr.fold_of(s, protocol['folds']) == k)
        reserved_file = work / f'reserved_fold{k}.json'
        reserved_file.write_text(json.dumps({'stems': sorted(reserved | set(held))}))
        fitted = []
        for family in sorted({m['family'] for m in members}):
            tables = kr.family_tables(Path(lcfs), work, family, protocol['family_labels'][family])
            variants = {f"{m['name']}_tw5_k{k}": {s: v for s, v in m['settings'].items() if s != 'threads'}
                        for m in members if m['family'] == family}
            vfile = work / f'variants_{family}_fold{k}.json'; vfile.write_text(json.dumps(variants, indent=1))
            code = subprocess.call([sys.executable, str(wrapper), str(TAIL_WEIGHT), '--tables', str(tables),
                                    '--out-root', str(OUT / 'out'), '--reserved', str(reserved_file),
                                    '--variants', str(vfile)], env=env)
            print(json.dumps({'fold': k, 'family': family, 'exit': code, 'members': list(variants)}), flush=True)
            assert code == 0
            fitted += list(variants)
        networks = [Network.load(OUT / 'out' / name / 'network.npz') for name in fitted]
        reports = {name: json.loads((OUT / 'out' / name / 'fit_report.json').read_text()) for name in fitted}
        rows = np.flatnonzero(row_fold == k)
        _, labels = predict_together(networks, {layout: features[rows]})
        pred = np.column_stack([labels[name] for name in TARGETS]).astype(np.float32)
        np.savez(OUT / f'gbm_oof_fold{k}.npz', pred=pred, rows=rows, targets=np.asarray(TARGETS))
        (OUT / f'gbm_oof_fold{k}.json').write_text(json.dumps({
            'fold': k, 'members': fitted, 'tail_weight': TAIL_WEIGHT, 'seconds': round(time.time() - started),
            'validation': {name: r['validation'] for name, r in reports.items()},
            'calibration': {name: r['calibration'] for name, r in reports.items()}}, indent=1))
        print(json.dumps({'FOLD_DONE': k, 'rows': int(rows.size), 'seconds': round(time.time() - started)}), flush=True)
    print('ALL_D15_DONE', FOLDS, flush=True)


main()
