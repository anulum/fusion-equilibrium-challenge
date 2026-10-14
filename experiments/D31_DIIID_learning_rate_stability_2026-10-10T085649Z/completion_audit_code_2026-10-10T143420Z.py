# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — controlled wider-network learning rate training
"""Verify completed campaign evidence before removing redundant own arrays."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STAMP = '2026-10-10T143420Z'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    terminal = json.loads((HERE / 'terminal_receipt.json').read_text())
    assert terminal['status'] == 'D31_workflow_complete' and terminal['successful_new_fits'] == 30
    unit = subprocess.run(['systemctl', '--user', 'show', 'fec-d31-diiid-stability-v1', '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True)
    assert unit.stdout.strip() == 'inactive'
    frozen = json.loads((HERE / 'frozen_sources.json').read_text())
    for path, expected in frozen.items():
        assert digest(path) == expected, path
    fit = json.loads((HERE / 'fits/receipt.json').read_text())
    readout = json.loads((HERE / 'readouts/receipt.json').read_text())
    for group, receipt in [('fits', fit), ('readouts', readout)]:
        assert len(receipt['tasks']) == 30
        for task in receipt['tasks']:
            assert task['status'] == ('fixed_final_epoch_fit_complete' if group == 'fits' else 'geometry_readout_complete')
            if group == 'fits':
                assert task['epochs_completed'] == 60
            assert digest(HERE / group / task['name'] / 'network.npz') == task['network_sha256']
    manifest = json.loads((HERE / 'evaluation/manifest.json').read_text())
    assert len(manifest['cases']) == 400 and not manifest['incomplete_arms']
    expected = {f'shot_{i:04d}_{field}.npy' for i in range(400) for field in ('psirz', 'q95', 'betaN')}
    archives = {}
    for arm in manifest['complete_arms']:
        path = HERE / 'evaluation' / (arm + '.npz')
        with zipfile.ZipFile(path) as zipped:
            assert len(zipped.namelist()) == 1200 and set(zipped.namelist()) == expected
        archives[arm] = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
    opened = {arm: np.load(record['path'], allow_pickle=False) for arm, record in archives.items()}
    cleanup = []
    frames = 0
    try:
        for case in manifest['cases']:
            path = HERE / 'evaluation/shots' / (case['stem'] + '.npz')
            record = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == record['sha256']
            assert record['frames'] == case['span'][1] - case['span'][0]
            frames += record['frames']
            with np.load(path, allow_pickle=False) as checkpoint:
                assert len(checkpoint.files) == 9
                for arm, archived in opened.items():
                    for field in ('psirz', 'q95', 'betaN'):
                        a = checkpoint[arm + '_' + field]
                        b = archived[f"shot_{case['index']:04d}_{field}"]
                        assert a.dtype == b.dtype == np.float32
                        assert len(a) == record['frames'] and np.isfinite(a).all() and np.isfinite(b).all()
                        assert np.array_equal(a, b), (case['stem'], arm, field)
            cleanup.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': record['sha256']})
        assert frames == 89584
    finally:
        for archived in opened.values():
            archived.close()
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    assert summary['shots'] == 400 and summary['frames'] == 89584 and not summary['incomplete_arms']
    diagnostic = json.loads((HERE / 'training_diagnostic.json').read_text())
    assert len(diagnostic['tasks']) == 30 and not diagnostic['seed_omission'] and not diagnostic['checkpoint_selection']
    audit = {'status': 'full_completion_integrity_verified', 'frozen_sources_unchanged': len(frozen),
             'fits_verified': 30, 'readouts_verified': 30, 'shots': 400, 'frames': frames,
             'archives': archives, 'all_checkpoint_arrays_bitidentical_to_retained_archives': True,
             'all_float32_arrays_finite': True, 'summary_sha256': digest(HERE / 'evaluation/summary.json'),
             'diagnostic_sha256': digest(HERE / 'training_diagnostic.json'), 'redundant_case_arrays': cleanup}
    audit_path = HERE / ('completion_audit_' + STAMP + '.json')
    with audit_path.open('x') as stream:
        json.dump(audit, stream, indent=2)
    # Delete only the fully verified duplicate arrays after a final inactivity check.
    assert subprocess.run(['systemctl', '--user', 'show', 'fec-d31-diiid-stability-v1', '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip() == 'inactive'
    for record in cleanup:
        Path(record['path']).unlink()
    with (HERE / ('cleanup_' + STAMP + '.json')).open('x') as stream:
        json.dump({'status': 'only_bitidentical_duplicate_case_arrays_removed',
                   'audit': str(audit_path), 'files_removed': len(cleanup),
                   'bytes_freed': sum(r['bytes'] for r in cleanup), 'retained_archives': archives,
                   'case_json_hash_receipts_and_models_checkpoints_logs_sources_retained': True}, stream, indent=2)
    print(json.dumps({'audit': str(audit_path), 'redundant_files_removed': len(cleanup),
                      'GiB_freed': sum(r['bytes'] for r in cleanup)/1024**3}), flush=True)


if __name__ == '__main__':
    main()
