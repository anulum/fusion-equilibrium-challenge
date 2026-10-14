# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — synthetic whole-driver recovery fixture
"""Exercise the real fixed60 fit routine without creating a candidate activation."""
import argparse
import json
from pathlib import Path

import numpy as np
import train_X_specialists_v3 as trainer
import qualify_X_optimizer_resume_v2 as fixture
import jax

HERE = Path(__file__).resolve().parent
OUT = HERE / 'driver_recovery_qualification'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('continuous', 'interrupted'), required=True)
    case = parser.parse_args().case
    protocol = json.loads((OUT / 'scope.json').read_text())
    assert protocol['status'] == 'synthetic_driver_qualification_not_D40_activation'
    for path, expected in protocol['sources_sha256'].items():
        assert trainer.kernel.initial.digest(path) == expected, path
    card = next(i for i, tasks in enumerate(trainer.assignments()) if ('static', 500) in tasks)
    device = jax.devices()[card]
    with jax.default_device(device):
        fixture_protocol = json.loads((fixture.OUT / 'protocol.json').read_text())
        _, _, _, reference, data, _ = fixture.inputs(fixture_protocol)
        normalization = {'feature_mean': reference.network.feature_mean[:582],
                         'feature_scale': reference.network.feature_scale[:582],
                         'output_mean': np.zeros(4), 'output_scale': np.ones(4),
                         'temporal_mean': reference.mean, 'temporal_scale': reference.scale}
        root = OUT / case
        root.mkdir(exist_ok=True)
        report = trainer.fit_one('static', 500, data, normalization, device, root, OUT / 'scope.json')
    print(json.dumps({'synthetic_driver_case_complete': case, 'operational_resumes': report['operational_resumes']}), flush=True)


if __name__ == '__main__':
    main()
