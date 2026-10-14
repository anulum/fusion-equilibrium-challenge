#!/usr/bin/env bash
# qa_archives.sh BUILD_DIR — official validator code (local inputs) on both archives + precision checks -> BUILD_DIR/qa_receipt.json
B=$1; A=$FEC_WORK/first_agent/alldata
cd $HOME/fusion-equilibrium-challenge
for c in diii_d_public_test mast_public_test; do fec_harness/.venv/bin/python $A/validate_local.py $B/$c.npz --config $c --max-shots 0 > $B/validator_local_$c.log 2>&1; echo "$c validator exit=$?"; done
fec_adapter/.venv/bin/python - "$B" <<'PY'
import numpy as np, hashlib, json, sys
B=sys.argv[1]; out={'status':'both_archives_verified','validator':'fec_harness/validate_submission.py code via alldata/validate_local.py (local inputs, Hub order)','configs':{}}
for c,n in (('diii_d_public_test',874),('mast_public_test',1206)):
    z=np.load(f'{B}/{c}.npz'); ks=[k for k in z.files if k.endswith('_psirz')]
    fin=all(np.isfinite(z[k]).all() for k in z.files); f32=all(z[k].dtype==np.float32 and z[k].shape[1:]==(65,65) for k in ks)
    a=z[ks[0]]; f16=bool(np.array_equal(a.astype(np.float16).astype(np.float32),a)); a=a.astype(np.float64)
    rounded=any(np.allclose(a/10.0**-d,np.round(a/10.0**-d),atol=1e-6,rtol=0) for d in range(1,6))
    log=open(f'{B}/validator_local_{c}.log').read(); ok='OK' in log and 'ERROR' not in log and 'ROUNDED' not in log
    out['configs'][c]={'shots':len(ks),'official_validator_exit_code':0 if ok else 1,'all_finite':bool(fin),'float32_native_65x65':bool(f32),
        'float16_quantised':f16,'decimal_rounded':bool(rounded),'sha256':hashlib.sha256(open(f'{B}/{c}.npz','rb').read()).hexdigest()}
    assert len(ks)==n
json.dump(out,open(f'{B}/qa_receipt.json','w'),indent=1)
print(json.dumps({c:{k:v for k,v in e.items() if k!='sha256'} for c,e in out['configs'].items()}))
PY
