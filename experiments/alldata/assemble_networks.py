# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data refit: the 65-network list in recipe order
"""Direct members from the fits dir under the ledger's fitted_name (amendments 1 and 2); bank members from the
readout dir. Every network must load; writes one path per line and a sha manifest.

    python assemble_networks.py --version 27|28
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--version', required=True); a = ap.parse_args(); v = a.version
    proto = json.loads((HERE / 'protocol_alldata_refit_20260929.json').read_text())
    ledger = json.loads((HERE / f'alldata_fit_ledger_v{v}.json').read_text())
    nets = []
    for m in sorted(proto['members'], key=lambda m: m['recipe_index']):
        e = ledger[m['name']]
        assert e['status'] in ('ok', 'ok_after_retry', 'ok_amendment2'), (m['name'], e['status'])
        p = (HERE / f'readout_v{v}' / f"{m['name']}_all" / 'network.npz') if m['readout_bank_refit'] \
            else (HERE / f'fits_v{v}' / e['fitted_name'] / 'network.npz')
        z = np.load(p); [z[k] for k in z.files]
        nets.append(p)
    assert len(nets) == 65
    (HERE / f'networks_v{v}.txt').write_text(''.join(f'{p}\n' for p in nets))
    (HERE / f'networks_v{v}.sha').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n' for p in nets))
    print(v, len(nets), 'networks, all load')
if __name__ == '__main__':
    main()
