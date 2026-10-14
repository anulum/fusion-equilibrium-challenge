# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — all-data refit of the v25 recipe (protocol_alldata_refit_20260929)
"""Fit the named recipe members on the all-data tables with their original settings, seeds and
calibration flag; reserved list empty. Family tables are label subsets of the all-data lcfs table.

    python alldata_run.py --tables DIR --work DIR --out DIR --members i,j,...
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
HERE = Path('$FEC_WORK/first_agent/alldata')
sys.path.insert(0, str(HERE))
from kfold_run import family_tables  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=Path, required=True); p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--members', required=True)
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args()
    proto = json.loads((HERE / 'protocol_alldata_refit_20260929.json').read_text())
    assert proto['status'] == 'frozen_before_any_alldata_fit'
    keep = {int(x) for x in a.members.split(',')}
    members = [m for m in proto['members'] if m['recipe_index'] in keep]
    a.work.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True)
    reserved = a.work / 'reserved_none.json'; reserved.write_text(json.dumps({'stems': [], 'note': 'all-data refit'}))
    groups = {}
    for m in members:
        groups.setdefault((m['family'], m['train_on_calibration']), []).append(m)
    log = []
    for (fam, tc), group in sorted(groups.items()):
        tables = family_tables(a.tables, a.work, fam, proto['family_labels'][fam])
        variants = {f"{m['name']}_all": {s: v for s, v in m['settings'].items() if s != 'threads'} for m in group}
        vf = a.work / f'variants_{fam}_{"tc" if tc else "plain"}_all.json'; vf.write_text(json.dumps(variants, indent=1))
        cmd = [a.python, '-m', 'fec_adapter.diii_d_accuracy.fit_on_accelerators', '--tables', str(tables), '--out-root', str(a.out),
               '--reserved', str(reserved), '--variants', str(vf)] + (['--train-on-calibration'] if tc else [])
        t0 = time.time(); code = subprocess.call(cmd, env=os.environ.copy())
        log.append({'family': fam, 'tc': tc, 'members': list(variants), 'exit': code, 'seconds': round(time.time() - t0)})
        (a.out / f'alldata_log_{min(keep)}.json').write_text(json.dumps(log, indent=1))
        if code != 0:
            raise SystemExit(f'group {fam} tc={tc} failed {code}')
    print(json.dumps({'ALL_DONE': sorted(keep)}), flush=True)


if __name__ == '__main__':
    main()
