# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D23-0 gate: q95 and betaN heads on a physics-normalised target, in the networks
"""K-fold fits of four S16 members with two more labels (d23_ratio_heads/protocol_d23_0_gate.json).

q95 and betaN both carry the factor B_t / I_p, which is where their extreme values come from. Besides the usual
heads the networks here get q95_s = q95 / ratio and beta_n_s = beta_n / ratio, ratio being table feature 60
(|I_TF| / 1e5 over max(|I_p|, floor) / 1e6, an input). The answers written are the direct heads and the ratio
heads multiplied back by the ratio. Tables: the 367 table features plus the 215 sub-frame features, as arm S of
D18-0. fec_adapter.diii_d_accuracy.fit_on_accelerators unchanged, called with --allow-host.
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
RECIPE = [16, 24, 30, 31]
RATIO_COLUMN = 60
EXTRA_LABELS = {'q95_s': 'q95', 'beta_n_s': 'beta_n'}
TARGETS = ['q95', 'beta_n', 'li_scored', 'r_axis', 'z_axis', 'kappa', 'tri_top', 'tri_bot', 'volume',
           'x_lower_r', 'x_lower_z', 'x_upper_r', 'x_upper_z']
ROOT = os.environ.get('D23_INPUT_ROOT', '/kaggle/input')
OUT = Path(os.environ.get('D23_OUT', '/kaggle/working'))
WORK = Path(os.environ.get('D23_WORK', '/tmp/work'))
LIMIT_SHOTS = int(os.environ.get('D23_LIMIT_SHOTS', '0'))
INSTALL = os.environ.get('D23_NO_INSTALL') != '1'


def sh(command):
    print('$', command, flush=True)
    return subprocess.run(command, shell=True).returncode


def find(pattern):
    return sorted(glob.glob(ROOT + '/**/' + pattern, recursive=True))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def augmented(lcfs, extra_of):
    """Write the lcfs tables with the sub-frame columns appended and the two ratio labels added, once."""
    import numpy as np
    out = WORK / 'lcfs_ratio_labels'
    if (out / 'DONE').is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for part in ('training', 'calibration', 'validation'):
        with np.load(Path(lcfs) / f'table_{part}.npz', allow_pickle=False) as t:
            arrays = {k: t[k] for k in t.files}
        with np.load(extra_of[part], allow_pickle=False) as z:
            assert np.array_equal(z['shot_of_row'], arrays['shot_of_row']), part
            extra = z['features']  # read once
        names = arrays['label_names'].astype(str).tolist()
        ratio = arrays['features'][:, RATIO_COLUMN].astype(np.float64)
        assert np.isfinite(ratio).all() and ratio.min() > 0.0, part
        added = [arrays['labels'][:, names.index(source)].astype(np.float64) / ratio for source in EXTRA_LABELS.values()]
        arrays['labels'] = np.concatenate([arrays['labels'], np.column_stack(added).astype(arrays['labels'].dtype)], axis=1)
        arrays['label_names'] = np.asarray(names + list(EXTRA_LABELS))
        arrays['features'] = np.concatenate([arrays['features'], extra], axis=1)
        del extra
        if LIMIT_SHOTS and part == 'training':  # smoke test only: a regular subsample of the shots
            n_rows, n_shots = len(arrays['shot_of_row']), len(arrays['stems'])
            keep_shots = np.arange(n_shots)[::max(1, n_shots // LIMIT_SHOTS)]
            rows = np.isin(arrays['shot_of_row'], keep_shots)
            renumber = np.full(n_shots, -1); renumber[keep_shots] = np.arange(len(keep_shots))
            for k in list(arrays):
                if k in ('label_names', 'feature_layout'):
                    continue
                if k == 'shot_of_row':
                    arrays[k] = renumber[arrays[k][rows]]
                elif len(arrays[k]) == n_rows:
                    arrays[k] = arrays[k][rows]
                elif len(arrays[k]) == n_shots:
                    arrays[k] = arrays[k][keep_shots]
        np.savez(out / f'table_{part}.npz', **arrays)
        del arrays
        gc.collect()
    (out / 'flux_basis.npz').write_bytes((Path(lcfs) / 'flux_basis.npz').read_bytes())
    (out / 'DONE').write_text('ok')
    return out


def main():
    if INSTALL:
        assert sh('pip install -q "jax==0.8.2"') == 0
    kdir = os.path.dirname([p for p in find('kfold_run.py') if '/kfold/' in p][0])
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(find('fec_adapter/diii_d_accuracy/fit_on_accelerators.py')[0])))
    lcfs = os.path.dirname(find('table_training.npz')[0])
    sums = dict(reversed(line.split()) for line in open(os.path.join(lcfs, 'SHA256SUMS')).read().splitlines())
    for name in ('table_training.npz', 'table_calibration.npz', 'table_validation.npz'):
        assert digest(os.path.join(lcfs, name)) == sums[name], name
    extra_of = {part: find(f'subframe_{part}.npz')[0] for part in ('training', 'calibration', 'validation')}
    print(json.dumps({part: digest(path)[:16] for part, path in extra_of.items()}), flush=True)
    env = dict(os.environ, PYTHONPATH=pkg, PYTHONUNBUFFERED='1', JAX_ENABLE_X64='1')
    sys.path.insert(0, kdir); sys.path.insert(0, pkg)
    import numpy as np
    import kfold_run as kr
    from fec_adapter.diii_d_accuracy.network import Network, predict_together
    protocol = json.loads(Path(kdir, 'protocol_kfold_oof_maps_20260929.json').read_text())
    reserved = set(json.loads(Path(kdir, 'reserved_for_evaluation.json').read_text())['stems'])
    members = [m for m in protocol['members'] if m['recipe_index'] in RECIPE]
    assert len(members) == len(RECIPE), [m['name'] for m in members]
    WORK.mkdir(parents=True, exist_ok=True)
    source = augmented(lcfs, extra_of)
    with np.load(source / 'table_training.npz', allow_pickle=False) as t:
        stems, shot_of_row, layout = t['stems'].astype(str), t['shot_of_row'], str(t['feature_layout'])
    stem_fold = np.array([-1 if s in reserved else kr.fold_of(s, protocol['folds']) for s in stems])
    for k in FOLDS:
        started = time.time()
        held = sorted(s for s, f in zip(stems, stem_fold) if f == k)
        reserved_file = WORK / f'reserved_fold{k}.json'
        reserved_file.write_text(json.dumps({'stems': sorted(reserved | set(held))}))
        fitted = []
        for family in sorted({m['family'] for m in members}):
            tables = kr.family_tables(source, WORK, family, protocol['family_labels'][family] + list(EXTRA_LABELS))
            variants = {f"{m['name']}_r_k{k}": {s: v for s, v in m['settings'].items() if s != 'threads'}
                        for m in members if m['family'] == family}
            vfile = WORK / f'variants_{family}_r_fold{k}.json'; vfile.write_text(json.dumps(variants, indent=1))
            code = subprocess.call([sys.executable, '-m', 'fec_adapter.diii_d_accuracy.fit_on_accelerators',
                                    '--tables', str(tables), '--out-root', str(OUT / 'out'),
                                    '--reserved', str(reserved_file), '--variants', str(vfile), '--allow-host'], env=env)
            print(json.dumps({'fold': k, 'family': family, 'exit': code, 'seconds': round(time.time() - started)}), flush=True)
            assert code == 0
            fitted += list(variants)
        rows = np.flatnonzero(stem_fold[shot_of_row] == k)
        with np.load(source / 'table_training.npz', allow_pickle=False) as t:
            features = t['features']  # read once
            matrix, truth = features[rows], t['coordinates'][rows]
            del features
        networks = [Network.load(OUT / 'out' / name / 'network.npz') for name in fitted]
        coordinates, labels = predict_together(networks, {layout: matrix})
        ratio = matrix[:, RATIO_COLUMN].astype(np.float64)
        result = {'rows': rows, 'targets': np.asarray(TARGETS), 'stems': stems,
                  'pred_r': np.column_stack([labels[name] for name in TARGETS]).astype(np.float32),
                  'coord_sse_r': np.sum((coordinates - truth[:, :coordinates.shape[1]]) ** 2, axis=1).astype(np.float32),
                  'ratio': ratio.astype(np.float32)}
        for name, source_name in EXTRA_LABELS.items():
            result[f'{source_name}_from_ratio'] = (np.asarray(labels[name], dtype=np.float64) * ratio).astype(np.float32)
        report = {name: {part: json.loads((OUT / 'out' / name / 'fit_report.json').read_text())[part]
                         for part in ('calibration', 'validation')} for name in fitted}
        np.savez(OUT / f'd23_fold{k}.npz', **result)
        (OUT / f'd23_fold{k}.json').write_text(json.dumps({'fold': k, 'members': fitted, 'reports': report,
                                                           'width': int(matrix.shape[1]), 'heads': sorted(labels),
                                                           'seconds': round(time.time() - started)}, indent=1))
        print(json.dumps({'FOLD_DONE': k, 'rows': int(rows.size), 'seconds': round(time.time() - started)}), flush=True)
        del matrix, truth, result
        gc.collect()
    print('ALL_D23_DONE', FOLDS, flush=True)


main()
