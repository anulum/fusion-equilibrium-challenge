# SPDX-License-Identifier: AGPL-3.0-or-later
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Reactor Systems — the official validator's own code, fed from the local copy of the public test inputs
"""fec_harness/validate_submission.py unchanged; only `datasets.load_dataset` is replaced by a stream of the local
parquet files in sorted order (the Hub's stream order; the builds read the same local copy). Only efit_times is read."""
import sys, types, glob
import pyarrow.parquet as pq
LOCAL = '$DISK/_offroot_cache/fec_data_local'
def load_dataset(repo, config, split=None, streaming=True):
    for f in sorted(glob.glob(f'{LOCAL}/{config}/*.parquet')):
        yield {'efit_times': pq.read_table(f, columns=['efit_times'])['efit_times'][0].as_py()}
sys.modules['datasets'] = types.SimpleNamespace(load_dataset=load_dataset)
sys.path.insert(0, '$HOME/fusion-equilibrium-challenge/fec_harness')
import validate_submission as v
sys.argv = ['validate_submission.py', *sys.argv[1:]]
raise SystemExit(v.main())
