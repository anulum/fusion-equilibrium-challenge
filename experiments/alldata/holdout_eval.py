# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — the holdout part opened for its purpose: one honest estimate per candidate
"""Build and score one candidate network list on the 358 holdout shots with the exact v25 finisher,
using the official local scorer (fec_harness/run_parallel_score.py) in-process.

Entrant 2026-09-29 opened the holdout ("neblokujme sa"). v25 and v27 never saw it, so their holdout
scores are an honest paired comparison; v28 trains on it and is not evaluated here. The package guard
is lifted in this process only (the scorer refuses holdout shots in chosen_files).

    python holdout_eval.py --tag v25 --networks-from-recipe
    python holdout_eval.py --tag v27 --networks FILE   (one network path per line, recipe order)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import fec_adapter.split as split

split.refuse_holdout = lambda stem: None          # entrant-approved opening of the holdout part

HERE = Path(__file__).resolve().parent
REPO = Path('$HOME/fusion-equilibrium-challenge')
sys.path.insert(0, str(HERE.parent / 'kfold'))
sys.path.insert(0, str(REPO / 'fec_harness'))
import kfold_build as kb  # noqa: E402
import run_parallel_score as scoring  # noqa: E402

REFS = Path('$DISK/_offroot_cache/fec_holdout_reference_cache')
OUT = HERE / 'holdout_eval'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--networks', type=Path)
    ap.add_argument('--networks-from-recipe', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.networks_from_recipe:
        nets = [Path(line.split(maxsplit=1)[1]) for line in
                (kb.W / 'results/fixed_axis_volume_full357_56cb/network_recipe.sha').read_text().splitlines()]
    else:
        nets = [Path(line.strip()) for line in a.networks.read_text().splitlines() if line.strip()]
    assert all(n.is_file() for n in nets)
    paths = scoring.chosen_files('holdout', 0, 0)
    assert len(paths) == 358 and all(split.bucket_of(p.stem) == 'holdout' for p in paths)
    d = OUT / a.tag
    d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    finish, values = kb.v25_finish()
    arc = d / 'diii_d_train.npz'
    if not arc.is_file():
        kb.vbuild.predicted_archive(paths, nets, kb.BASIS, a.workers, finish).write(arc)
    built = time.time() - started
    sys.argv = ['run_parallel_score.py', '--pred', str(arc), '--references', str(REFS), '--workers', str(a.workers),
                '--part', 'holdout', '--per-shot', str(d / 'per_shot.json')]
    code = scoring.main()
    (d / 'receipt.json').write_text(json.dumps({
        'tag': a.tag, 'networks': [str(n) for n in nets],
        'network_sha256': [hashlib.sha256(n.read_bytes()).hexdigest() for n in nets],
        'finish': {k: v for k, v in values.items()}, 'shots': len(paths), 'archive': str(arc),
        'build_seconds': built, 'total_seconds': time.time() - started, 'scorer_exit': code}, indent=1, default=str))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
