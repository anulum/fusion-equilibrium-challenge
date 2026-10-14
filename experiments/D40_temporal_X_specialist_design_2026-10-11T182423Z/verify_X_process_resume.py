# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — new-process X optimizer checkpoint verification
"""Resume a prefix in a new GPU process and compare every saved optimizer leaf."""
import argparse
import json
import time

import numpy as np
import qualify_X_optimizer_resume as qualification
import jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('static', 'mean', 'ordered'))
    mode = parser.parse_args().mode
    started = time.monotonic()
    kernel = qualification.kernel
    qualification.verify()
    protocol = json.loads((qualification.OUT / 'protocol.json').read_text())
    _, _, _, _, data, normalization = qualification.inputs(protocol)
    device = jax.devices()[0]
    assert device.platform == 'gpu'
    template, _ = kernel.initial_state(500, device)
    frozen = {'scope': 'synthetic_optimizer_qualification_only', 'mode': mode,
              'recipe_sha256': kernel.initial.digest(qualification.HERE / 'training_recipe_frozen_v1.json'),
              'protocol_sha256': kernel.initial.digest(qualification.OUT / 'protocol.json')}
    folder = qualification.OUT / mode
    restored, key, history = kernel.resume(folder / 'epoch1.npz', template, frozen)
    step = kernel.epoch_step(data, normalization, mode, kernel.settings(500))
    actual, actual_key, actual_history = kernel.advance(step, restored, key, history, 2)
    expected, expected_key, expected_history = kernel.resume(folder / 'epoch3.npz', template, frozen)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))
    assert np.array_equal(jax.random.key_data(actual_key), jax.random.key_data(expected_key))
    assert actual_history == expected_history and int(actual[-1]) == 6
    qualification.verify()
    qualification.write(folder / 'new_process_resume.json', {'status': 'new_process_full_optimizer_RNG_history_resume_bitexact',
             'mode': mode, 'synthetic_labels_only': True, 'device': str(device),
             'checkpoint_source_sha256': kernel.initial.digest(folder / 'epoch1.npz'),
             'continuous_reference_sha256': kernel.initial.digest(folder / 'epoch3.npz'),
             'seconds': time.monotonic()-started})
    print(json.dumps({'new_process_resume_bitexact': mode}), flush=True)


if __name__ == '__main__':
    main()
