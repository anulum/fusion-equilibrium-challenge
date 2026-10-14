# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D22b diagnostic: which families of the 16-member K-fold ensemble to average, out of fold
"""The public scores of H0, H1a and H1 fall as the h and then the c members are added to the hx/hc set.

The same question off the public test, on the 16-member K-fold ensembles (4 c, 4 h, 3 hx, 5 hc per fold; table
features only; readout bank refitted), every fourth held-out frame: the squared error of the flux coordinates
and of every head for the hx+hc members, for those plus h, and for all sixteen, each family alone, and the
ratios to hx+hc with a shot bootstrap (2000, seed 990). Raw network output, no finisher. Diagnosis only.

    python d22b_family_subsets.py --pkg DIR --table table_training.npz --kfold DIR --lists DIR --nets DIR
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STRIDE = 4


def family(rel):
    name = Path(rel).parent.name
    for f in ('discharge_hx_', 'discharge_hc_', 'discharge_h_', 'discharge_c_'):
        if name.startswith(f):
            return f[len('discharge_'):-1]
    raise ValueError(rel)


def main():
    ap = argparse.ArgumentParser()
    for name in ('pkg', 'table', 'kfold', 'lists', 'nets'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    started = time.time()
    sys.path.insert(0, str(a.pkg))
    from fec_adapter.diii_d_accuracy.network import Network, predict_together
    reserved = set(json.loads((a.kfold / 'reserved_for_evaluation.json').read_text())['stems'])
    with np.load(a.table, allow_pickle=False) as t:
        names = t['label_names'].astype(str).tolist()
        stems, shot_of_row, layout = t['stems'].astype(str), t['shot_of_row'], str(t['feature_layout'])
        stem_fold = np.array([-1 if s in reserved else int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 for s in stems])
        row_fold = stem_fold[shot_of_row]
        keep = np.flatnonzero((row_fold >= 0) & (np.arange(len(row_fold)) % STRIDE == 0))
        features = t['features']  # read once
        base = features[keep]
        del features
        truth, labels = t['coordinates'][keep], t['labels'][keep]
    fold, shot = row_fold[keep], shot_of_row[keep]
    sets = ('c', 'h', 'hx', 'hc', 'hx+hc', 'h+hx+hc', 'all')
    csse = {s: np.full(len(keep), np.nan) for s in sets}
    head_se, sizes = {}, {}
    for k in range(5):
        nets = {}
        for line in (a.lists / f's16_fold{k}.sha').read_text().splitlines():
            rel = line.split(maxsplit=1)[1]
            nets.setdefault(family(rel), []).append(Network.load(a.nets / rel))
        groups = {'c': nets['c'], 'h': nets['h'], 'hx': nets['hx'], 'hc': nets['hc'], 'hx+hc': nets['hx'] + nets['hc'],
                  'h+hx+hc': nets['h'] + nets['hx'] + nets['hc'], 'all': nets['c'] + nets['h'] + nets['hx'] + nets['hc']}
        sizes[k] = {s: len(g) for s, g in groups.items()}
        rows = np.flatnonzero(fold == k)
        for s, group in groups.items():
            coordinates, heads = predict_together(group, {layout: base[rows]})
            csse[s][rows] = np.sum((coordinates - truth[rows][:, :coordinates.shape[1]]) ** 2, axis=1)
            for name, value in heads.items():
                if name in names:
                    head_se.setdefault(name, {}).setdefault(s, np.full(len(keep), np.nan))[rows] = \
                        (np.asarray(value) - labels[rows][:, names.index(name)]) ** 2
        print(json.dumps({'fold': k, 'rows': int(rows.size), 'sizes': sizes[k], 'seconds': round(time.time() - started)}), flush=True)
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)

    def ratio(new, old):
        ok = np.isfinite(new) & np.isfinite(old)
        pn, po = np.bincount(shot[ok], weights=new[ok], minlength=shots), np.bincount(shot[ok], weights=old[ok], minlength=shots)
        b = (counts @ pn) / (counts @ po)
        return {'ratio': float(pn.sum() / po.sum()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                'P_new_better': float(np.mean(b < 1.0))}

    out = {'frames': int(len(keep)), 'stride': STRIDE, 'sizes': sizes,
           'flux_coordinates': {'mean_sse': {s: float(np.nanmean(csse[s])) for s in sets},
                                **{f'{s}_over_hx+hc': ratio(csse[s], csse['hx+hc']) for s in sets if s != 'hx+hc'}},
           'heads': {}}
    for name, per in sorted(head_se.items()):
        entry = {'mean_se': {s: float(np.nanmean(v)) for s, v in per.items()}}
        for s in ('h+hx+hc', 'all', 'c', 'h'):
            if s in per and 'hx+hc' in per:
                entry[f'{s}_over_hx+hc'] = ratio(per[s], per['hx+hc'])
        out['heads'][name] = entry
    out['seconds'] = round(time.time() - started)
    (HERE / 'd22b_family_subsets_result.json').write_text(json.dumps(out, indent=1))
    f = out['flux_coordinates']
    print('flux mean sse', json.dumps(f['mean_sse']))
    for s in sets:
        if s != 'hx+hc':
            r = f[f'{s}_over_hx+hc']
            print(f"flux {s:8s} / hx+hc {r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
    for name, e in out['heads'].items():
        print(f"{name:12s} " + '  '.join(f"{s} {e[s + '_over_hx+hc']['ratio']:.4f} [{e[s + '_over_hx+hc']['ci95'][0]:.3f},{e[s + '_over_hx+hc']['ci95'][1]:.3f}]"
                                         for s in ('h+hx+hc', 'all', 'c', 'h') if s + '_over_hx+hc' in e))


main()
