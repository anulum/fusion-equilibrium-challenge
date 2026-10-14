# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D22 diagnostic: h and hx members with sub-frame inputs, alone and together, out of fold
"""Why did adding h members to the hx/hc set lower the public score (H1a against H0)?

Diagnosis off the public test. The D18-0 arm-S networks (two h and two hx members per fold, sub-frame inputs,
fitted without the fold's shots) answer on every fourth held-out frame: each alone, the h pair, the hx pair and
all four. Reported per set: the squared error of the 64 flux coordinates and of each head both families have,
and the ratio of all four to the hx pair with a shot bootstrap (2000, seed 990). Raw network output; no readout
refit, no finisher. Nothing is fitted or selected here.

    python d22_family_pairs.py --pkg DIR --table table_training.npz --subframe subframe_training.npz --kfold DIR
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D18 = HERE.parent / 'd18_subframe_nets'
STRIDE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', type=Path, required=True)
    ap.add_argument('--table', type=Path, required=True)
    ap.add_argument('--subframe', type=Path, required=True)
    ap.add_argument('--kfold', type=Path, required=True)
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
    with np.load(a.subframe, allow_pickle=False) as z:
        assert np.array_equal(z['shot_of_row'], shot_of_row)
        extra = z['features']  # read once
        wide = np.concatenate([base, extra[keep]], axis=1)
        del extra, base
    fold, shot = row_fold[keep], shot_of_row[keep]
    sets = ('h0', 'h1', 'x0', 'x1', 'h_pair', 'hx_pair', 'all_four')
    csse = {s: np.full(len(keep), np.nan) for s in sets}
    head_se = {}
    for k in range(5):
        dirs = sorted(D18.glob(f'kaggle_out_*/out/*_s_k{k}'))
        nets = {d.name: Network.load(d / 'network.npz') for d in dirs}
        h = [n for name, n in nets.items() if name.startswith('discharge_h_')]
        x = [n for name, n in nets.items() if name.startswith('discharge_hx_')]
        assert len(h) == 2 and len(x) == 2, sorted(nets)
        rows = np.flatnonzero(fold == k)
        groups = {'h0': h[:1], 'h1': h[1:], 'x0': x[:1], 'x1': x[1:], 'h_pair': h, 'hx_pair': x, 'all_four': h + x}
        for s, group in groups.items():
            coordinates, heads = predict_together(group, {layout: wide[rows]})
            csse[s][rows] = np.sum((coordinates - truth[rows][:, :coordinates.shape[1]]) ** 2, axis=1)
            for name, value in heads.items():
                if name in names:
                    head_se.setdefault(name, {}).setdefault(s, np.full(len(keep), np.nan))[rows] = \
                        (np.asarray(value) - labels[rows][:, names.index(name)]) ** 2
        print(json.dumps({'fold': k, 'rows': int(rows.size), 'seconds': round(time.time() - started)}), flush=True)
    shots = int(shot.max()) + 1
    rng = np.random.default_rng(990)
    counts = np.stack([np.bincount(rng.integers(0, shots, shots), minlength=shots) for _ in range(2000)]).astype(float)

    def ratio(new, old):
        ok = np.isfinite(new) & np.isfinite(old)
        pn, po = np.bincount(shot[ok], weights=new[ok], minlength=shots), np.bincount(shot[ok], weights=old[ok], minlength=shots)
        b = (counts @ pn) / (counts @ po)
        return {'ratio': float(pn.sum() / po.sum()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                'P_new_better': float(np.mean(b < 1.0)), 'frames': int(ok.sum())}

    out = {'frames': int(len(keep)), 'stride': STRIDE,
           'flux_coordinates': {'mean_sse': {s: float(np.nanmean(csse[s])) for s in sets},
                                'all_four_over_hx_pair': ratio(csse['all_four'], csse['hx_pair']),
                                'h_pair_over_hx_pair': ratio(csse['h_pair'], csse['hx_pair'])},
           'heads': {}}
    for name, per in sorted(head_se.items()):
        entry = {'mean_se': {s: float(np.nanmean(v)) for s, v in per.items()}}
        if 'all_four' in per and 'hx_pair' in per and 'h_pair' in per:
            entry['all_four_over_hx_pair'] = ratio(per['all_four'], per['hx_pair'])
            entry['h_pair_over_hx_pair'] = ratio(per['h_pair'], per['hx_pair'])
        out['heads'][name] = entry
    out['seconds'] = round(time.time() - started)
    (HERE / 'd22_family_pairs_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['flux_coordinates'], indent=1))
    for name, e in out['heads'].items():
        if 'all_four_over_hx_pair' in e:
            r, q = e['all_four_over_hx_pair'], e['h_pair_over_hx_pair']
            print(f"{name:12s} all4/hx {r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  h/hx {q['ratio']:.4f} [{q['ci95'][0]:.4f}, {q['ci95'][1]:.4f}]")
        else:
            print(f"{name:12s} only in {sorted(e['mean_se'])}")


main()
