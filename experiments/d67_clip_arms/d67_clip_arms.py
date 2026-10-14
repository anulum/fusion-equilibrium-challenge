# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D67 (TODO FEC-R07): matched fits with the input clip at 8 (incumbent) and at 50
"""Fits the first four 78-label recipe members (hc seeds 200-203) twice, clip 8 and clip 50, interleaved, on the
prepared fit-excluded tables, and stores every held-frame prediction of the 358 development shots."""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--members', type=int, default=4)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_clip as chain  # noqa: E402  (FEC_PACKAGE names the unchanged package)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from fec_adapter.diii_d_accuracy import network as netmod  # noqa: E402
from fec_adapter.diii_d_accuracy import fit_network as fitmod  # noqa: E402
from fec_adapter.diii_d_accuracy.fit_network import Table, FitSettings  # noqa: E402

CLIPS = (8.0, 50.0)


def load(part):
    d = args.prepared / part
    meta = json.loads((d / 'metadata.json').read_text())
    v = {k: np.load(d / (k + '.npy'), mmap_mode='r') for k in ('features', 'coordinates', 'labels', 'shot_of_row')}
    return Table(v['features'], v['coordinates'], v['labels'], tuple(meta['label_names']), {}, 'discharge', v['shot_of_row'])


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    train, cal, held = load('training'), load('calibration'), load('held')
    all_data = chain.merge([train, cal])
    device = jax.devices('gpu')[0]
    recipe = json.loads((Path(os.environ['FEC_PACKAGE']) / 'training_recipe.json').read_text())['members']
    members = [m for m in recipe if len(m['label_names']) == 78][:args.members]
    for m in members:
        for clip in CLIPS:
            tag = f"{Path(m['model_path']).parent.name}_clip{int(clip)}"
            path = args.out / tag / 'network.npz'
            if (path.parent / 'held_predictions.npz').exists():
                continue
            chain.CLIP = clip
            netmod.INPUT_CLIP = clip  # inference of the exported network
            fitmod.INPUT_CLIP = clip  # calibration inputs of the checkpoint selection
            settings = FitSettings(**m['fit_settings'])
            names = tuple(m['label_names'])
            source = all_data if m['train_on_calibration'] else train
            started = time.monotonic()
            network = chain.train_network(chain.view(source, names, settings.coordinates), chain.view(cal, names, settings.coordinates),
                                          settings, 'maps', path, device, False)
            coords, labs = [], {n: [] for n in names}
            for s in range(0, len(held.features), 4096):
                c, l = network.predict(np.asarray(held.features[s:s + 4096]))
                coords.append(c.astype(np.float32))
                for n in names:
                    labs[n].append(np.asarray(l[n], dtype=np.float32))
            np.savez(path.parent / 'held_predictions.npz', coordinates=np.concatenate(coords),
                     **{'label_' + n: np.concatenate(labs[n]) for n in names})
            chain.record(path.parent / 'arm.json', {'member': m['model_path'], 'seed': settings.seed, 'clip': clip,
                         'fit_and_predict_seconds': time.monotonic() - started, 'network_sha256': chain.digest(path)})
            print(json.dumps({'done': tag, 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
