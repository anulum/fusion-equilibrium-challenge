# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — shared X initialization on three ROCm cards
"""Verify every paired seed retains exactly the same float32 GPU parameters."""
import json
from pathlib import Path
import time

import numpy as np
import shared_initial_parameters as initial
import jax

HERE = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    protocol = json.loads((HERE / 'shared_initial_GPU_protocol.json').read_text())
    for path, expected in protocol['sources_sha256'].items():
        assert initial.digest(path) == expected, path
    devices = jax.devices()
    assert len(devices) == 3 and all(d.platform == 'gpu' for d in devices)
    receipt = json.loads((initial.OUT / 'receipt.json').read_text())
    reports = []
    for item in receipt['models']:
        assert initial.digest(item['file']) == item['sha256']
        cpu = initial.load(Path(item['file']))
        for device in devices:
            uploaded = jax.tree.map(lambda a: jax.device_put(a, device), cpu)
            assert all(np.array_equal(a, np.asarray(b)) for a, b in zip(jax.tree.leaves(cpu), jax.tree.leaves(uploaded), strict=True))
            reports.append({'seed': item['seed'], 'device': str(device), 'device_kind': device.device_kind,
                            'initial_arrays_bitexact_CPU_GPU': True})
    assert len(reports) == 24
    for path, expected in protocol['sources_sha256'].items():
        assert initial.digest(path) == expected, path
    with (HERE / 'shared_initial_GPU_qualification.json').open('x') as stream:
        json.dump({'status': 'all8_shared_initializations_bitexact_on_all3_ROCm_cards', 'reports': reports,
                   'no_optimizer_updates_or_data_or_labels_read': True,
                   'not_optimizer_checkpoint_resume_qualification': True,
                   'seconds': time.monotonic()-started}, stream, indent=2)
    print(json.dumps({'shared_initial_GPU_checks': 24, 'bitexact': True}), flush=True)


if __name__ == '__main__':
    main()
