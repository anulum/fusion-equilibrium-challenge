# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — DIII-D re-evaluation R4: learning curve (100 % vs 80 % of the data)
"""Same S16 sub-ensemble, same v25 finisher, on the 357 validation shots: once with the original
members (fitted on all training [+ calibration] shots) and once with fold 0's members (fitted without
a fifth of the training shots). Official local scorer on both, paired per shot. Diagnosis of how
much more data is worth; nothing is selected by it."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = HERE.parent / 'kfold'
sys.path.insert(0, str(K))
import kfold_build as kb  # noqa: E402

REPO = Path('$HOME/fusion-equilibrium-challenge')
REFS = kb.W / 'results/local_validation_reference_cache_20260927'
OUT = HERE / 'results' / 'r4_learning_curve'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / 'fec_harness'))
    import run_parallel_score as scoring
    paths = scoring.chosen_files('validation', 0, 357)
    protocol = json.loads((K / 'protocol_kfold_oof_maps_20260929.json').read_text())
    recipe = [line.split(maxsplit=1)[1] for line in
              (kb.W / 'results/fixed_axis_volume_full357_56cb/network_recipe.sha').read_text().splitlines()]
    full = [Path(recipe[m['recipe_index']]) for m in protocol['members']]
    fold0 = [Path(p) for p in json.loads((K / 'oof_fold0/receipt.json').read_text())['networks']]
    finish, _ = kb.v25_finish()
    result = {}
    for tag, nets in (('S16_full_data', full), ('S16_fold0_80pct', fold0)):
        d = OUT / tag
        d.mkdir(exist_ok=True)
        arc = d / 'diii_d_train.npz'
        if not arc.is_file():
            kb.vbuild.predicted_archive(paths, nets, kb.BASIS, 6, finish).write(arc)
        subprocess.run([sys.executable, str(REPO / 'fec_harness/run_parallel_score.py'), '--pred', str(arc),
                        '--references', str(REFS), '--workers', '6', '--part', 'validation',
                        '--per-shot', str(d / 'per_shot.json')], stdout=open(d / 'score.log', 'w'),
                       stderr=subprocess.STDOUT, check=True)
        result[tag] = (d / 'score.log').read_text().splitlines()[-12:]
        print(tag, '\n'.join(result[tag]), flush=True)
    (OUT / 'summary.json').write_text(json.dumps({'networks': {'full': [str(p) for p in full],
                                                              'fold0': [str(p) for p in fold0]},
                                                  'score_tail': result, 'selection_or_fit': False}, indent=1))


if __name__ == '__main__':
    main()
