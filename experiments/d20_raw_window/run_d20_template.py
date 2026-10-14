# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D20-0 gate: networks that also read the raw 0.5 ms input windows
"""K-fold fits of four S16 members on the host processor (d20_raw_window/protocol_d20_0_gate.json).

Arm W reads the 367 table features, the 215 sub-frame features and the 840 raw-window samples; arm S (the control
kernel only) reads the first two, as arm S of D18-0 did on an accelerator. Same settings, seeds, rows and code;
fec_adapter.diii_d_accuracy.fit_on_accelerators unchanged, called with --allow-host. For each fold the arm then
answers on the frames of the fold's held-out training shots: the 13 heads, and the squared error of the 64 flux
coordinates per frame.
"""
import gc
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FOLDS = __FOLDS__
ARM = __ARM__
RECIPE = [16, 24, 30, 31]
TARGETS = ['q95', 'beta_n', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume',
           'x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
EXTRA = {'s': ('subframe',), 'w': ('subframe', 'rawwindow')}[ARM]
OUT = Path('/kaggle/working')


def sh(command):
    print('$', command, flush=True)
    return subprocess.run(command, shell=True).returncode


def find(pattern):
    return sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def widened(features, part, extra_of, rows=None):
    """The table features with the arm's extra columns appended, in single precision."""
    import numpy as np
    blocks = [features if rows is None else features[rows]]
    for kind in EXTRA:
        with np.load(extra_of[kind][part], allow_pickle=False) as z:
            block = z['features']  # read once
        blocks.append((block if rows is None else block[rows]).astype(np.float32))
        del block
    return np.concatenate(blocks, axis=1)


def extended(tables, work, family, extra_of):
    """Write the family's tables with the arm's extra columns appended to the features."""
    import numpy as np
    out = work / f'tables_{family}_{ARM}'
    out.mkdir(parents=True, exist_ok=True)
    for part in ('training', 'calibration', 'validation'):
        with np.load(tables / f'table_{part}.npz', allow_pickle=False) as t:
            arrays = {k: t[k] for k in t.files}
        for kind in EXTRA:
            with np.load(extra_of[kind][part], allow_pickle=False) as z:
                assert np.array_equal(z['shot_of_row'], arrays['shot_of_row']), (kind, part)
        arrays['features'] = widened(arrays['features'], part, extra_of)
        assert np.isfinite(arrays['features']).all(), part
        np.savez(out / f'table_{part}.npz', **arrays)
        del arrays
        gc.collect()
    (out / 'flux_basis.npz').write_bytes((tables / 'flux_basis.npz').read_bytes())
    return out


def main():
    assert sh('pip install -q "jax==0.8.2"') == 0
    kdir = os.path.dirname([p for p in find('kfold_run.py') if '/kfold/' in p][0])
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(find('fec_adapter/diii_d_accuracy/fit_on_accelerators.py')[0])))
    lcfs = os.path.dirname(find('table_training.npz')[0])
    sums = dict(reversed(line.split()) for line in open(os.path.join(lcfs, 'SHA256SUMS')).read().splitlines())
    for name in ('table_training.npz', 'table_calibration.npz', 'table_validation.npz'):
        assert digest(os.path.join(lcfs, name)) == sums[name], name
    extra_of = {kind: {part: find(f'{kind}_{part}.npz')[0] for part in ('training', 'calibration', 'validation')}
                for kind in EXTRA}
    print(json.dumps({kind: {part: digest(path)[:16] for part, path in parts.items()}
                      for kind, parts in extra_of.items()}), flush=True)
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
    for k in FOLDS:
        started = time.time()
        held = sorted(s for s in splits['training'] if s not in reserved and kr.fold_of(s, protocol['folds']) == k)
        reserved_file = work / f'reserved_fold{k}.json'
        reserved_file.write_text(json.dumps({'stems': sorted(reserved | set(held))}))
        fitted = []
        for family in sorted({m['family'] for m in members}):
            plain = kr.family_tables(Path(lcfs), work, family, protocol['family_labels'][family])
            tables = extended(plain, work, family, extra_of)
            variants = {f"{m['name']}_{ARM}_k{k}": {s: v for s, v in m['settings'].items() if s != 'threads'}
                        for m in members if m['family'] == family}
            vfile = work / f'variants_{family}_{ARM}_fold{k}.json'; vfile.write_text(json.dumps(variants, indent=1))
            code = subprocess.call([sys.executable, '-m', 'fec_adapter.diii_d_accuracy.fit_on_accelerators',
                                    '--tables', str(tables), '--out-root', str(OUT / 'out'),
                                    '--reserved', str(reserved_file), '--variants', str(vfile), '--allow-host'], env=env)
            print(json.dumps({'fold': k, 'family': family, 'arm': ARM, 'exit': code,
                              'seconds': round(time.time() - started)}), flush=True)
            assert code == 0
            fitted += list(variants)
            shutil.rmtree(tables)
        with np.load(os.path.join(lcfs, 'table_training.npz'), allow_pickle=False) as t:
            stems, shot_of_row, layout = t['stems'].astype(str), t['shot_of_row'], str(t['feature_layout'])
            stem_fold = np.array([-1 if s in reserved else kr.fold_of(s, protocol['folds']) for s in stems])
            rows = np.flatnonzero(stem_fold[shot_of_row] == k)
            base, truth = t['features'], t['coordinates'][rows]
        matrix = widened(base, 'training', extra_of, rows)
        del base
        networks = [Network.load(OUT / 'out' / name / 'network.npz') for name in fitted]
        coordinates, labels = predict_together(networks, {layout: matrix})
        result = {'rows': rows, 'targets': np.asarray(TARGETS),
                  f'pred_{ARM}': np.column_stack([labels[name] for name in TARGETS]).astype(np.float32),
                  f'coord_sse_{ARM}': np.sum((coordinates - truth[:, :coordinates.shape[1]]) ** 2, axis=1).astype(np.float32)}
        report = {name: {part: json.loads((OUT / 'out' / name / 'fit_report.json').read_text())[part]
                         for part in ('calibration', 'validation')} for name in fitted}
        np.savez(OUT / f'd20_{ARM}_fold{k}.npz', **result)
        (OUT / f'd20_{ARM}_fold{k}.json').write_text(json.dumps({'fold': k, 'arm': ARM, 'members': fitted, 'reports': report,
                                                                 'width': int(matrix.shape[1]),
                                                                 'seconds': round(time.time() - started)}, indent=1))
        print(json.dumps({'FOLD_DONE': k, 'rows': int(rows.size), 'seconds': round(time.time() - started)}), flush=True)
        del matrix, truth, result
        gc.collect()
    print('ALL_D20_DONE', ARM, FOLDS, flush=True)


main()
