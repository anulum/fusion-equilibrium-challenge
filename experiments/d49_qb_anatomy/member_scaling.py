# SPDX-License-Identifier: AGPL-3.0-or-later
# SCPN Reactor Systems — D49b: q95 / betaN by member on the 358 development shots (measurement only)
"""Per-member answers of the 69 fit-disjoint map networks and the 8 fit-disjoint joint scalar networks.

Reports each family's pooled R2 alone, the mean R2 of one member, the error correlation between members, and the
expected R2 of an equal mean of N joint members (all subsets up to 8), with the variance split into the part common
to the members and the part that averages out. Nothing is fitted and nothing is selected.
"""
import itertools
import json

import numpy as np

import d39_common as c

MEANS = {'q95': 5.262893061875164, 'beta_n': 1.2688498003107427}


def main():
    import predict_diii_d as pd
    tasks = json.loads((c.X / 'd38/tasks.json').read_text())['tasks']
    maps = [(t['name'], t['family'], pd.Network.load(c.X / 'd38/readouts' / t['name'] / 'network.npz')) for t in tasks if t['family'] != 'joint']
    joint = [(t['name'], 'joint', pd.Network.load(c.X / 'd38/fits' / t['name'] / 'network.npz')) for t in tasks if t['family'] == 'joint']
    z = np.load(c.X / 'reports/d49_qb_frames_358.npz')
    stems = [str(s) for s in z['stems']]
    features = np.concatenate([c.load_cache(s).features.astype(float) for s in stems])
    truth = {'q95': z['q95_truth'].astype(float), 'beta_n': z['betaN_truth'].astype(float)}
    assert len(features) == len(truth['q95'])
    answers = {'q95': {}, 'beta_n': {}}
    for name, family, net in maps:
        _, labels = pd.predict_together([net], {'discharge': features})
        for key in answers:
            answers[key][name] = (family, np.asarray(labels[key], dtype=float))
    padded = np.pad(features, ((0, 0), (0, 183)))
    for name, family, net in joint:
        _, labels = pd.predict_together([net], {'discharge': padded})
        for key in answers:
            answers[key][name] = (family, np.asarray(labels[key], dtype=float))
    report = {}
    for key, table in answers.items():
        t = truth[key]
        total = np.sum((t - MEANS[key]) ** 2)
        r2 = lambda p: float(1 - np.sum((p - t) ** 2) / total)
        families = sorted({f for f, _ in table.values()})
        block = {}
        errors = {}
        for family in families:
            members = np.stack([p for f, p in table.values() if f == family])
            e = members - t
            errors[family] = e
            cov = e @ e.T / e.shape[1]
            off = cov[~np.eye(len(cov), dtype=bool)]
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            block[family] = {'members': int(len(members)), 'r2_family_mean': r2(members.mean(axis=0)),
                             'r2_one_member_mean': float(np.mean([r2(m) for m in members])),
                             'r2_one_member_range': [float(min(r2(m) for m in members)), float(max(r2(m) for m in members))],
                             'mean_error_correlation': float(corr[~np.eye(len(cov), dtype=bool)].mean()),
                             'common_mse': float(off.mean()), 'member_mse': float(np.diag(cov).mean()),
                             'r2_limit_infinite_members': float(1 - off.mean() * e.shape[1] / total)}
        e = errors['joint']
        block['joint_scaling'] = {str(n): float(np.mean([r2(t + e[list(s)].mean(axis=0)) for s in itertools.combinations(range(len(e)), n)]))
                                  for n in range(1, len(e) + 1)}
        allmap = np.concatenate([errors[f] for f in families if f != 'joint'])
        block['map69'] = {'r2': r2(t + allmap.mean(axis=0)),
                          'error_correlation_with_joint8': float(np.corrcoef(allmap.mean(axis=0), e.mean(axis=0))[0, 1])}
        for share in (.25, .5, .75):
            block['map69'][f'r2_blend_joint_{share}'] = r2(t + (1 - share) * allmap.mean(axis=0) + share * e.mean(axis=0))
        report[key] = block
    (c.X / 'reports/d49b_member_scaling_358.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
