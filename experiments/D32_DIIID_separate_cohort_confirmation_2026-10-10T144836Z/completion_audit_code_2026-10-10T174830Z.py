# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed separate-cohort confirmation of D31
"""Verify every retained confirmation array and exact replacement before cleanup."""

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STAMP = '2026-10-10T174830Z'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inactive():
    return subprocess.run(['systemctl', '--user', 'show', 'fec-d32-confirmation-v3', '-p', 'ActiveState', '--value'], capture_output=True, text=True, check=True).stdout.strip() == 'inactive'


def main():
    assert inactive()
    terminal = json.loads((HERE / 'terminal_receipt.json').read_text())
    assert terminal['status'] == 'D32_workflow_complete' and terminal['shots'] == 358
    sources = json.loads((HERE / 'frozen_sources.json').read_text())
    for path, wanted in sources.items():
        assert digest(path) == wanted, path
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert len(manifest['cases']) == 358 and len(manifest['models']) == 40
    for model in manifest['models']:
        assert digest(model['path']) == model['sha256']
    expected = {f'shot_{i:04d}_{field}.npy' for i in range(358) for field in ('psirz', 'q95', 'betaN')}
    archives = {}
    for arm, record in terminal['archives'].items():
        assert digest(record['path']) == record['sha256']
        with zipfile.ZipFile(record['path']) as stored:
            assert len(stored.namelist()) == 1074 and set(stored.namelist()) == expected
        archives[arm] = record
    opened = {arm: np.load(row['path'], allow_pickle=False) for arm, row in archives.items()}
    removal = []
    frames = 0
    try:
        for case in manifest['cases']:
            path = HERE / 'shots' / (case['stem'] + '.npz')
            record = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == record['sha256']
            assert record['index'] == case['index'] and record['frames'] == case['span'][1] - case['span'][0]
            values = {}
            with np.load(path, allow_pickle=False) as checkpoint:
                assert len(checkpoint.files) == 9
                for arm in manifest['arms']:
                    for field in ('psirz', 'q95', 'betaN'):
                        a = checkpoint[arm + '_' + field]
                        b = opened[arm][f"shot_{case['index']:04d}_{field}"]
                        assert a.dtype == b.dtype == np.float32 and len(a) == record['frames']
                        assert np.isfinite(a).all() and np.isfinite(b).all()
                        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
                        values[arm + '_' + field] = b
            rebuilt = io.BytesIO()
            np.savez(rebuilt, **values)
            assert hashlib.sha256(rebuilt.getbuffer()).hexdigest() == record['sha256']
            frames += record['frames']
            removal.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': record['sha256']})
        assert frames == 78875
    finally:
        for stored in opened.values():
            stored.close()
    assert len(list((HERE / 'references').glob('*.pkl'))) == 358
    assert len(json.loads((HERE / 'reference_statistics.json').read_text())) == 358
    for arm in manifest['arms']:
        assert len(json.loads((HERE / 'evaluation' / (arm + '_parts.json')).read_text())) == 358
    summary = json.loads((HERE / 'evaluation/summary.json').read_text())
    assert summary['confirmation_passes'] is False and summary['shots'] == 358 and summary['frames'] == 78875
    audit_path = HERE / ('completion_audit_' + STAMP + '.json')
    audit = {'status': 'full_confirmation_integrity_and_exact_replacement_verified',
             'source_files_verified': len(sources), 'models_verified': 40,
             'shots': 358, 'frames': frames, 'archives': archives,
             'all_float32_predictions_finite': True, 'all_original_checkpoint_NPZ_bytes_reconstructible_SHA_exact': True,
             'references_and_official_parts_complete': True, 'summary_sha256': digest(HERE / 'evaluation/summary.json'),
             'redundant_case_arrays': removal}
    with audit_path.open('x') as stream:
        json.dump(audit, stream, indent=2)
    assert inactive()
    for row in removal:
        Path(row['path']).unlink()
    with (HERE / ('cleanup_' + STAMP + '.json')).open('x') as stream:
        json.dump({'status': 'only_exact_duplicate_case_arrays_removed', 'audit': str(audit_path),
                   'files_removed': len(removal), 'bytes_freed': sum(r['bytes'] for r in removal),
                   'archives': archives, 'case_JSON_receipts_models_lookups_refs_parts_logs_sources_retained': True}, stream, indent=2)
    print(json.dumps({'audit': str(audit_path), 'redundant_files_removed': len(removal),
                      'GiB_freed': sum(r['bytes'] for r in removal) / 1024**3}), flush=True)


if __name__ == '__main__':
    main()
