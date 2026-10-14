# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — controlled wider-network learning rate training
"""Describe every completed paired training trajectory after common scoring."""

import json
from campaign_common import ARMS, D27, D30, HERE, digest, write_json


def trajectory(path):
    # An operational resume can repeat unsaved epochs; retain original log and
    # describe the latest checkpoint-consistent attempt for each epoch.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_epoch = {row['epoch']: row for row in rows}
    return [by_epoch[k] for k in sorted(by_epoch)]


def main():
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    assert summary['status'] == 'all_three_learning_rate_comparisons_processed'
    tasks = json.loads((HERE / 'fits/receipt.json').read_text())['tasks']
    rows = []
    for task in tasks:
        name = task['name']
        arm, suffix = name.split('_seed')
        assert arm in ARMS
        value = {'name': name, 'fit_status': task['status']}
        if task['status'] == 'fixed_final_epoch_fit_complete':
            own = HERE / 'fits' / name / 'passes.jsonl'
            wide = D30 / 'fits' / ('capacity_seed' + suffix) / 'passes.jsonl'
            narrow = D27 / 'fits' / ('control_seed' + suffix) / 'passes.jsonl'
            tracks = {key: trajectory(path) for key, path in [('new', own), ('wide002', wide), ('narrow002', narrow)]}
            assert all(len(track) == 60 and track[-1]['epoch'] == 60 for track in tracks.values())
            last = {key: track[-1]['training_loss'] for key, track in tracks.items()}
            value.update(final_loss=last, ratio_to_wide=last['new']/last['wide002'],
                         ratio_to_narrow=last['new']/last['narrow002'],
                         max_loss={key:max(r['training_loss'] for r in track) for key,track in tracks.items()},
                         trajectories=tracks, log_sha256={str(p):digest(p) for p in (own,wide,narrow)})
        rows.append(value)
    assert len(rows) == 30
    write_json(HERE / 'training_diagnostic.json', {'status':'all_30_training_trajectories_processed',
        'tasks':rows, 'score_comparison_sha256':digest(HERE/'evaluation/summary.json'),
        'seed_omission':False, 'checkpoint_selection':False,
        'scope':'Descriptive optimization evidence; admission remains original full-score gate.'})


if __name__ == '__main__':
    main()
