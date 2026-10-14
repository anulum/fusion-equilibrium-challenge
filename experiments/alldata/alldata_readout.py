# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data original-geometry readout refit of the 49 bank members
"""As the v25 bank (ridge 0.001, exact validity groups, every original-map geometry target the network
outputs) on the all-data rows: combined training table (training + validation, reserved included) and
calibration for the calibration-trained members. Targets: second_agent cache (training minus reserved,
calibration) + targets_extra (validation, reserved training). CPU, streamed.

    python alldata_readout.py --nets DIR --out DIR [--members i,j,...]
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
W = Path('$FEC_WORK/second_agent-recovery-20260926')
sys.path.insert(0, os.environ.get('FEC_SCRATCH', str(W / 'scratch')))
from fec_adapter.diii_d_accuracy.network import Network  # noqa: E402
from geometry_readout_streaming import validity_groups, cpu_moments, corrected_from_moments  # noqa: E402
TABLES = Path(os.environ.get('FEC_TABLES', '$HOME/fusion-equilibrium-challenge/kaggle_upload/fec-tables-alldata'))
second_agent_ROOT = W / 'results/original_map_geometry_targets_20260927'
EXTRA_ROOT = Path('$FEC_WORK/first_agent/alldata/targets_extra')  # as written in the receipts
ROOTS = {str(second_agent_ROOT): Path(os.environ.get('FEC_TARGETS_ROOT', str(second_agent_ROOT))),
         str(EXTRA_ROOT): Path(os.environ.get('FEC_EXTRA_ROOT', str(EXTRA_ROOT)))}
RECEIPTS = [ROOTS[str(second_agent_ROOT)] / 'complete_receipt.json', ROOTS[str(EXTRA_ROOT)] / 'complete_receipt.json']
if os.environ.get('FEC_WITH_HOLDOUT') == '1':   # v28: the holdout part's targets (entrant go 2026-09-29)
    RECEIPTS.append(ROOTS[str(EXTRA_ROOT)] / 'holdout_receipt.json')
def local(path):
    for orig, here in ROOTS.items():
        if path.startswith(orig): return str(here) + path[len(orig):]
    return path
RIDGE, BLOCK = 0.001, 16384
def digest(p):
    with Path(p).open('rb') as s: return hashlib.file_digest(s, 'sha256').hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nets', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--members', default=''); a = ap.parse_args()
    proto = json.loads((HERE / 'protocol_alldata_refit_20260929.json').read_text())
    records = {}
    for rp in RECEIPTS:
        rc = json.loads(rp.read_text()); names = rc['label_names']
        for r in rc['records']: records[r['stem']] = r
    assert len(names) == 73
    tables = {}
    for part in ('training', 'calibration'):
        with np.load(TABLES / f'table_{part}.npz', allow_pickle=False) as t:
            stems = t['stems'].astype(str); sor = t['shot_of_row']; feats = t['features']
        counts = np.bincount(sor, minlength=len(stems)); b = np.concatenate(([0], np.cumsum(counts)))
        truth = np.full((len(feats), 73), np.nan, dtype=np.float32)
        for i, s in enumerate(stems):
            r = records[s]; assert r['frames'] == b[i + 1] - b[i], s
            with np.load(local(r['output_path']), allow_pickle=False) as tg:
                assert tg['stem'].item() == s; truth[b[i]:b[i + 1]] = tg['labels']
        tables[part] = (feats, truth, np.arange(len(feats)))
        print(json.dumps({'part': part, 'shots': int(len(stems)), 'rows': int(len(feats))}), flush=True)
    keep = {int(x) for x in a.members.split(',')} if a.members else None
    a.out.mkdir(parents=True, exist_ok=True); done = []
    for m in proto['members']:
        if not m['readout_bank_refit'] or (keep and m['recipe_index'] not in keep): continue
        t0 = time.time(); src = a.nets / f"{m['name']}_all" / 'network.npz'; old = Network.load(src)
        nm = [n for n in names if n in old.label_names]
        assert len(nm) == {'h': 5, 'hx': 9, 'hc': 73}[m['family']]
        sel = [names.index(n) for n in nm]
        parts = ('training', 'calibration') if m['train_on_calibration'] else ('training',)
        groups = validity_groups(np.concatenate([tables[p][1][:, sel] for p in parts]))
        total = None
        for p in parts:
            f, tr, rows = tables[p]
            for o in range(0, len(rows), BLOCK):
                blk = rows[o:o + BLOCK]
                mom = cpu_moments(old, f[blk, :old.feature_mean.size].astype(np.float64), tr[blk][:, sel], nm, groups)
                total = mom if total is None else tuple(tuple(x + y for x, y in zip(u, v)) for u, v in zip(total, mom))
        upd, reps = corrected_from_moments(old, total, nm, groups, RIDGE)
        d = a.out / f"{m['name']}_all"; d.mkdir(parents=True, exist_ok=True); upd.save(d / 'network.npz')
        (d / 'fit_report.json').write_text(json.dumps({'member': m['name'], 'recipe_index': m['recipe_index'], 'labels': nm,
            'fit_parts': list(parts), 'ridge': RIDGE, 'group_reports': reps, 'initial_sha256': digest(src),
            'network_sha256': digest(d / 'network.npz'), 'seconds': time.time() - t0}, indent=1))
        done.append(m['recipe_index']); print(json.dumps({'refit': m['name'], 'seconds': round(time.time() - t0)}), flush=True)
    (a.out / f'readout_receipt_{min(done) if done else "none"}.json').write_text(json.dumps({'members': done}, indent=1))
if __name__ == '__main__':
    main()
