# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D7: out-of-fold maps with the five train-on-calibration refits in place of their V0 members
"""Finish one fold's held-out shots exactly as kfold/oof_fold{K} was finished (same shots, order, v25 finisher), with
the V0 networks of the five changed members replaced by their D7 refits; the other eleven are the V0 networks.

Fit acceptance (protocol_d7_train_on_calibration.json): calibration flux R2 >= 0.999 for the first fit, else the
seed+1000 retry, else the gradient_limit 0.1 refit, else the member is dropped. The choice is written to the receipt.

    python d7_build.py --fold K --fits DIR --readouts DIR --out DIR [--workers 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAT = HERE.parent
sys.path.insert(0, str(SEAT / 'd4c'))
import kfold_build as kb  # noqa: E402

THRESHOLD = 0.999
CHANGED = {'discharge_c_seed0': False, 'discharge_c_seed1': False, 'discharge_h_labels02_s76': True,
           'discharge_h_seed8': True, 'discharge_hx_seed1': True}
"""V0 member name -> whether it is a readout-bank member."""


def calibration_r2(directory):
    report = directory / 'fit_report.json'
    return json.loads(report.read_text())['calibration']['flux_r2'] if report.is_file() else None


def accepted(fits, name, k):
    base = f'{name}_tcall_k{k}'
    tried = []
    for suffix in ('', '_retry', '_gl01'):
        r2 = calibration_r2(fits / (base + suffix))
        tried.append((base + suffix, r2))
        if r2 is not None and r2 >= THRESHOLD:
            return base + suffix, tried
    return None, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--fits', type=Path, required=True)
    ap.add_argument('--readouts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    started = time.time()
    k = a.fold
    base = json.loads((SEAT / 'kfold' / f'oof_fold{k}' / 'receipt.json').read_text())
    finish, values = kb.v25_finish()
    assert {key: str(v) for key, v in values.items()} == base['finish']
    networks = [Path(p) for p in base['networks']]
    choices = {}
    for name, bank in CHANGED.items():
        index = [i for i, p in enumerate(networks)
                 if p.parent.name == f'{name}_k{k}' or p.parent.name.startswith(f'{name}_k{k}_')]
        assert len(index) == 1, (name, index)
        fitted, tried = accepted(a.fits, name, k)
        choices[name] = {'tried': tried, 'accepted': fitted}
        if fitted is None:
            networks.pop(index[0])
            continue
        if bank:
            assert fitted == f'{name}_tcall_k{k}', 'a bank member needed a retry; readout naming not covered'
            networks[index[0]] = a.readouts / fitted / 'network.npz'
        else:
            networks[index[0]] = a.fits / fitted / 'network.npz'
    a.out.mkdir(parents=True, exist_ok=False)
    paths = [kb.DATA / f'{s}.parquet' for s in base['stems']]
    archive = kb.vbuild.predicted_archive(paths, networks, kb.BASIS, a.workers, finish)
    archive.write(a.out / 'diii_d_train.npz')
    receipt = {'status': 'd7_oof_maps_built', 'fold': k, 'shots': archive.shots, 'frames': archive.frames,
               'replaced': archive.replaced, 'stems': base['stems'], 'networks': [str(p) for p in networks],
               'network_sha256': [kb.digest(p) for p in networks], 'choices': choices, 'finish': base['finish'],
               'this_script_sha256': kb.digest(__file__), 'archive_sha256': kb.digest(a.out / 'diii_d_train.npz'),
               'seconds': time.time() - started, 'validation_targets_used': False, 'MAST_targets_used': False}
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({key: v for key, v in receipt.items() if key not in ('stems', 'networks', 'network_sha256')}), flush=True)


if __name__ == '__main__':
    main()
