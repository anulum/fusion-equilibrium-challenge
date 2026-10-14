# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — independent complete-grid peer recomputation
"""Use the accepted peer selector only after the entire 32-by-120 grid exists."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PEER = Path('$FEC_RUNS/ml350/runs/d39_finisher_20261011')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(manifest):
    for source, expected in manifest.items():
        assert digest(source) == expected, source


def main():
    started = time.time()
    try:
        manifest = json.loads((HERE / 'peer_screen_recomputation_sources.json').read_text())
        verify(manifest)
        variants = HERE / 'peer_amendment4_sources/d39_variants_a2.json'
        selector = HERE / 'peer_amendment4_sources/d39_amendment_4.json'
        declared = json.loads(variants.read_text())['variants']
        assert len(declared) == 32
        cases = json.loads((PEER / 'd32/manifest.json').read_text())['cases']
        cohort = json.loads((HERE.parent / 'D38_DIIID_boundary_spatial_temporal_2026-10-11T001358Z/cohort.json').read_text())
        assert len(cases) == 358 and [c['stem'] for c in cases] == sorted(cohort['development'])
        selected = [c for c in cases if c['index'] % 3 == 0]
        assert len(selected) == 120
        records = {(name, c['stem']): PEER / 'variants' / name / (c['stem'] + '.json')
                   for name in declared for c in selected}
        deadline = time.monotonic() + 30 * 3600
        while True:
            missing = sum(not p.exists() for p in records.values())
            (HERE / 'peer_screen_recomputation_state.json').write_text(json.dumps({
                'phase': 'waiting_for_complete_declared_grid', 'completed': len(records) - missing,
                'total': len(records), 'UTC_seconds': time.time()}))
            if not missing:
                break
            assert time.monotonic() < deadline, 'bounded complete-grid wait expired'
            time.sleep(60)
        verify(manifest)
        hashes = {name: {stem: digest(p) for (n, stem), p in records.items() if n == name} for name in declared}
        common = load('d39_common', HERE / 'peer_common_frozen.py')
        common.X = PEER  # Runtime path mapping only; frozen peer arithmetic unchanged.
        report = load('d39_independent_report', HERE / 'peer_amendment5_sources/d39_report_v3.py')
        out = HERE / 'peer_screen_independent_report.json'
        sys.argv = ['d39_report_v3.py', '--variants', str(variants), '--stride', '3', '--offsets', '0',
                    '--base', 'base', '--seed', '3901', '--replicates', '10000', '--select', str(selector),
                    '--out', str(out)]
        report.main()
        verify(manifest)
        for (name, stem), path in records.items():
            assert digest(path) == hashes[name][stem], path
        receipt = {'status': 'complete32x120_independent_frozen_peer_selector_recomputed',
                   'record_sha256': hashes, 'report_sha256': digest(out),
                   'whole_grid_required_before_any_metric_read': True, 'runtime_path_mapping_only': True,
                   'peer_sources_and_results_not_modified': True, 'not238_validation_or357_confirmation': True,
                   'seconds': time.time() - started, 'publication': False}
        with (HERE / 'peer_screen_recomputation_terminal.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
    except Exception as error:
        with (HERE / 'peer_screen_recomputation_failure.json').open('x') as stream:
            json.dump({'type': type(error).__name__, 'error': str(error), 'seconds': time.time() - started,
                       'candidate_admitted': False}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
