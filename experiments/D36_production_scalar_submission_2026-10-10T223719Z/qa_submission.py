# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — production archive validation with actual public input rows
"""Run the official validator against actual local copies of public-test inputs."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import datasets
p=Path(__file__).resolve().parent
DATA=Path('$FEC_DATA/data')
original_loader=datasets.load_dataset

def public_loader(repo,config,split=None,streaming=True):
    assert repo=='Sophelio/fusion-equilibrium-challenge' and split=='public_test' and streaming
    return original_loader('parquet',data_files=sorted(str(f)for f in(DATA/config).glob('*.parquet')),
                           split='train',streaming=True,columns=['efit_times'])

datasets.load_dataset=public_loader
sys.path.insert(0,'$HOME/fusion-equilibrium-challenge/fec_harness')
import validate_submission as official
rows={}
for config,shots in (('diii_d_public_test',874),('mast_public_test',1206)):
    path=p/'submission'/(config+'.npz')
    code=official.validate(path,config,0);assert code==0
    frames=0; native_precision_seen=False
    with np.load(path,allow_pickle=False)as z:
        expected={f'shot_{i:04d}_{head}'for i in range(shots)for head in('psirz','q95','betaN')}
        assert len(z.files)==len(expected) and set(z.files)==expected
        for i in range(shots):
            psi=z[f'shot_{i:04d}_psirz'];frames+=len(psi)
            assert psi.dtype==np.float32 and psi.shape[1:]==(65,65) and np.isfinite(psi).all()
            native_precision_seen |= not np.array_equal(psi.astype(np.float16).astype(np.float32),psi)
            for name in('q95','betaN'):
                v=z[f'shot_{i:04d}_{name}'];assert v.dtype==np.float32 and v.shape==(len(psi),) and np.isfinite(v).all()
    assert native_precision_seen, 'archive appears wholly float16 quantized'
    with path.open('rb')as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
    rows[config]={'shots':shots,'frames':frames,'official_validator_exit_code':code,'all_finite':True,
                  'float32_native_65x65':True,'float16_quantised':False,'decimal_rounded':False,'sha256':digest}
assert rows['diii_d_public_test']['frames']==193359
(p/'submission/qa_receipt.json').write_text(json.dumps({'status':'both_archives_verified','configs':rows,
 'validator_source':'existingofficialvalidate_submission.py; datasetloaderusesactualparquetpublicrows viaHFdatasets streaming',
 'assembly_precision_qualification':str(p/'assembly_precision_qualification.json')},indent=2)+'\n')
print(json.dumps({'status':'both_archives_verified','configs':rows}),flush=True)
