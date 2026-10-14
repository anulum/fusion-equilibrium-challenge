# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — guarded paired four-position specialist training
"""Run all24 fixed-final fits only with a source-bound post-D39 activation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import threading
import time

import numpy as np
import x_training_kernel as kernel
from campaign_common import base_maps, metadata, TABLES
from window_cache import mapped_array
from training_data import put_block
from fec_adapter.diii_d_accuracy.fit_network import _Scaling
import jax
import jaxlib
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
TRANSFER, COMPILE = threading.Lock(), threading.Lock()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify(activation_path):
    activation = json.loads(activation_path.read_text())
    assert activation['status'] == 'D40_training_activated_after_D39_terminal'
    assert activation['D39_terminal_outcome_source'] in activation['sources_sha256']
    assert activation['peer_trainer_review_source'] in activation['sources_sha256']
    for path, expected in activation['sources_sha256'].items():
        assert kernel.initial.digest(path) == expected, path
    receipt = json.loads((HERE / 'optimizer_resume_qualification/process_chain_receipt.json').read_text())
    assert receipt['status'] == 'all3_actual_new_process_optimizer_RNG_history_resume_bitexact'
    assert activation['recipe_sha256'] == kernel.initial.digest(HERE / 'training_recipe_frozen_v1.json')
    plan = json.loads(Path(activation['fit_row_plan']).read_text())
    cohort = json.loads((kernel.model.D38 / 'cohort.json').read_text())
    excluded = set(plan['excluded_stems'])
    if activation['branch'] == 'unused357_original6326':
        assert not activation['D39_opened357'] and not excluded
    else:
        assert activation['branch'] == 'D39_opened357_new400' and activation['D39_opened357']
        conditional = json.loads((HERE / 'conditional400_manifest_frozen.json').read_text())
        assert excluded == set(conditional['stems']) and len(excluded) == 400
    for part in ('training', 'calibration'):
        assert {s['stem'] for s in plan['parts'][part]} == set(cohort[part])-excluded
        assert len({s['stem'] for s in plan['parts'][part]}) == len(plan['parts'][part])
        assert all(s['teacher_window_span'][1]-s['teacher_window_span'][0] ==
                   s['feature_span'][1]-s['feature_span'][0] for s in plan['parts'][part])
        alignment = json.loads((kernel.model.D38 / 'temporal' / (part+'_alignment.json')).read_text())
        teacher = {s: alignment['offsets'][i:i+2] for i, s in enumerate(alignment['stems'])}
        meta = metadata(part)
        static = {s: meta['bounds'][i:i+2] for i, s in enumerate(meta['stems'])}
        assert all(s['teacher_window_span'] == teacher[s['stem']] and s['feature_span'] == static[s['stem']]
                   for s in plan['parts'][part])
    required = [str(Path(__file__)), str(HERE / 'x_training_kernel.py'),
                str(HERE / 'shared_initial_parameters.py'), str(HERE / 'x_temporal_model.py'),
                str(HERE / 'training_recipe_frozen_v1.json'), activation['fit_row_plan'],
                activation['normalization'], str(HERE / 'optimizer_resume_qualification/process_chain_receipt.json')]
    required += [str(HERE / 'shared_initial_parameters/receipt.json'), str(kernel.model.D38 / 'cohort.json'),
                 str(kernel.model.D38 / 'supervision/receipt.json')]
    required += [str(HERE / 'shared_initial_parameters' / ('seed'+str(s)+'.npz')) for s in range(500, 508)]
    for part in ('training', 'calibration'):
        required += [str(TABLES / ('table_'+part+'.npz')),
                     str(kernel.model.D38 / 'supervision' / (part+'_geometry.npy')),
                     str(kernel.model.D38 / 'temporal/windows' / ('rawwindow_'+part+'.npz')),
                     str(kernel.model.D38 / 'temporal' / (part+'_alignment.json'))]
    assert all(p in activation['sources_sha256'] for p in required)
    assert activation['runtime_versions'] == {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__}
    return activation, plan


def upload(plan, normalization, device):
    total = sum(s['teacher_window_span'][1]-s['teacher_window_span'][0]
                for part in ('training', 'calibration') for s in plan['parts'][part])
    base = base_maps()
    names = json.loads((kernel.model.D38 / 'supervision/receipt.json').read_text())['teacher_labels']
    columns = [names.index(n) for n in kernel.model.LABELS]
    with TRANSFER, jax.default_device(device):
        x = jnp.zeros((total, 582), jnp.float32)
        y = jnp.zeros((total, 4), jnp.float32)
        raw = jnp.zeros((total, 840), jnp.float16)
        at = 0
        for part in ('training', 'calibration'):
            windows = mapped_array(kernel.model.D38 / 'temporal/windows' / ('rawwindow_'+part+'.npz'), 'features')
            targets = np.load(kernel.model.D38 / 'supervision' / (part+'_geometry.npy'), mmap_mode='r', allow_pickle=False)
            for shot in plan['parts'][part]:
                a, b = shot['teacher_window_span']
                c, _ = shot['feature_span']
                for offset in range(0, b-a, 4096):
                    size = min(4096, b-a-offset)
                    features = base[part]['features'][c+offset:c+offset+size].astype(float)
                    features = np.clip((features-normalization['feature_mean'])/normalization['feature_scale'], -8., 8.).astype(np.float32)
                    wanted = targets[a+offset:a+offset+size, columns].astype(float)
                    wanted = ((wanted-normalization['output_mean'])/normalization['output_scale']).astype(np.float32)
                    sequence = np.asarray(windows[a+offset:a+offset+size], dtype=np.float16)
                    assert np.isfinite(features).all() and np.isfinite(sequence).all() and not np.isinf(wanted).any()
                    x = put_block(x, jax.device_put(features, device), at)
                    y = put_block(y, jax.device_put(wanted, device), at)
                    raw = put_block(raw, jax.device_put(sequence, device), at)
                    x.block_until_ready(); y.block_until_ready(); raw.block_until_ready()
                    at += size
        assert at == total
        return x, y, raw


def fit_one(mode, seed, data, normalization, device, root, activation_path):
    recipe = kernel.settings(seed)
    out = root / (mode+'_'+str(seed))
    out.mkdir(exist_ok=True)
    frozen = {'mode': mode, 'seed': seed, 'activation_sha256': kernel.initial.digest(activation_path),
              'recipe_sha256': kernel.initial.digest(HERE / 'training_recipe_frozen_v1.json')}
    if (out / 'fit_report.json').exists():
        receipt = json.loads((out / 'fit_report.json').read_text())
        assert receipt['frozen'] == frozen
        assert kernel.initial.digest(out / 'export/network.npz') == receipt['network_sha256']
        assert kernel.initial.digest(out / 'export/encoder.npz') == receipt['encoder_sha256']
        return receipt
    state, key = kernel.initial_state(seed, device)
    history = []
    checkpoints = sorted((out / 'checkpoints').glob('epoch*.json'), key=lambda p: int(p.stem[5:]))
    if checkpoints:
        state, key, history = kernel.resume(checkpoints[-1].with_suffix('.npz'), state, frozen)
    assert int(state[-1]) == len(history)*(len(data[0])//recipe.batch)
    start = len(history)
    mean, scale = [jnp.asarray(normalization[n], dtype=jnp.float32) for n in ('temporal_mean', 'temporal_scale')]
    step = kernel.epoch_step(data, (mean, scale), mode, recipe)
    started = time.monotonic()
    for epoch in range(start, 60):
        if epoch == start:
            with COMPILE:
                state, key, history = kernel.advance(step, state, key, history, 1)
        else:
            state, key, history = kernel.advance(step, state, key, history, 1)
        with (out / 'passes.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch+1, 'loss': history[-1], 'resumed_from': start,
                                      'seconds': time.monotonic()-started})+'\n')
        if (epoch+1)%10 == 0:
            kernel.checkpoint(out / 'checkpoints', state, key, history, frozen)
        print(json.dumps({'X_fit': out.name, 'epoch': epoch+1, 'loss': history[-1]}), flush=True)
    quality = kernel.quality(history, state[0])
    assert int(state[-1]) == 60*(len(data[0])//recipe.batch)
    assert quality['final_parameters_finite'], 'nonfinite final parameters; no numerical retry'
    scaling = _Scaling(np.r_[normalization['feature_mean'], np.zeros(32)],
                       np.r_[normalization['feature_scale'], np.ones(32)],
                       normalization['output_mean'], normalization['output_scale'])
    kernel.model.save(out / 'export', state[0], scaling, recipe,
                      normalization['temporal_mean'], normalization['temporal_scale'], mode)
    receipt = {'status': 'fit_complete', 'mode': mode, 'seed': seed, 'frozen': frozen,
               'epochs': len(history), 'resumed_from': start, 'quality': quality,
               'network_sha256': kernel.initial.digest(out / 'export/network.npz'),
               'encoder_sha256': kernel.initial.digest(out / 'export/encoder.npz'),
               'training_frames': len(data[0]), 'held_out_scores_read': False,
               'seconds': time.monotonic()-started}
    write(out / 'fit_report.json', receipt)
    return receipt


def worker(card, tasks, plan, normalization, out, activation_path):
    device = jax.devices()[card]
    receipts = []
    with jax.default_device(device):
        data = upload(plan, normalization, device)
        for mode, seed in tasks:
            failure = out / (mode+'_'+str(seed)+'_failure.json')
            if failure.exists():
                receipts.append(json.loads(failure.read_text()))
                continue
            try:
                receipts.append(fit_one(mode, seed, data, normalization, device, out, activation_path))
            except Exception as error:
                receipt = {'status': 'failed_no_numeric_retry', 'mode': mode, 'seed': seed,
                           'type': type(error).__name__, 'error': str(error)}
                write(failure, receipt)
                receipts.append(receipt)
            gc.collect()
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    activation, plan = verify(args.activation)
    with np.load(activation['normalization'], allow_pickle=False) as saved:
        normalization = {n: saved[n].copy() for n in ('feature_mean', 'feature_scale', 'output_mean', 'output_scale', 'temporal_mean', 'temporal_scale')}
    assert all(np.isfinite(a).all() for a in normalization.values())
    assert all(normalization[n].shape == (582,) for n in ('feature_mean', 'feature_scale'))
    assert all(normalization[n].shape == (4,) for n in ('output_mean', 'output_scale'))
    assert all(normalization[n].shape == (21,) for n in ('temporal_mean', 'temporal_scale'))
    assert all((normalization[n] > 0).all() for n in ('feature_scale', 'output_scale', 'temporal_scale'))
    assert len(jax.devices()) == 3 and all(d.platform == 'gpu' for d in jax.devices())
    assert args.out.resolve().is_relative_to(HERE)
    args.out.mkdir(exist_ok=True)
    tasks = [(mode, seed) for seed in range(500, 508) for mode in kernel.model.MODES]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(worker, i, tasks[i::3], plan, normalization, args.out, args.activation) for i in range(3)]
        receipts = [r for job in jobs for r in job.result()]
    verify(args.activation)
    decisions = {mode: all(r['status'] == 'fit_complete' and r['quality']['seed_quality_pass']
                          for r in receipts if r['mode'] == mode) for mode in kernel.model.MODES}
    assert len(receipts) == 24 and all(sum(r['mode'] == m for r in receipts) == 8 for m in kernel.model.MODES)
    write(args.out / 'receipt.json', {'status': 'all24_real_X_fits_terminal', 'fits': receipts,
             'family_training_quality_pass': decisions, 'no_reduced_seed_subset': True,
             'held_out_scores_computed': False, 'activation_sha256': kernel.initial.digest(args.activation)})


if __name__ == '__main__':
    main()
