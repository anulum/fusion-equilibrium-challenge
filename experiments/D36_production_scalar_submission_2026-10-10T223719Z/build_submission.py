# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — fixed scalar blend over unchanged published flux archives
"""Predict joint scalar heads and preserve original compressed map records."""
import io,json,os,shutil,time,zipfile,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from campaign_common import HERE,DATA,digest,write_json
from evaluate_batch import deployment_features
from fec_adapter.diii_d_accuracy.network import Network,predict_together

proto=json.loads((HERE/'protocol.json').read_text())
receipt=json.loads((HERE/'fits/receipt.json').read_text())
assert receipt['status']=='all_8_production_scalar_tasks_terminal' and receipt['successful']==8
models=[]
for seed in range(200,208):
 path=HERE/f'fits/joint_seed{seed}/network.npz'
 fit=json.loads(path.with_name('fit_report.json').read_text())
 assert fit['epochs_completed']==60 and fit['selected_epoch']==60 and fit['training_frames']==1559340
 assert digest(path)==fit['network_sha256']
 model=Network.load(path);assert model.coordinates==0 and model.label_names==('q95','beta_n')
 models.append(model)
files=sorted((DATA.parent/'diii_d_public_test').glob('*.parquet'));assert len(files)==874


def predict_case(task:tuple)->dict:
 index,path=task
 out=HERE/'public_predictions'/(path.stem+'.npz');note=out.with_suffix('.json')
 if note.exists():
  row=json.loads(note.read_text());assert digest(out)==row['sha256'];return row
 features=deployment_features(path);padded=np.pad(features,((0,0),(0,183)))
 _,labels=predict_together(models,{'discharge':padded})
 assert all(np.isfinite(labels[n]).all()for n in('q95','beta_n'))
 np.savez(out,q95=labels['q95'],betaN=labels['beta_n'])
 row={'index':index,'stem':path.stem,'frames':len(features),'sha256':digest(out)}
 write_json(note,row);print(json.dumps({'public_scalar_case_complete':path.stem,'index':index}),flush=True)
 return row


def qualify_process():
    path=files[0]
    with np.load(HERE/'public_predictions'/(path.stem+'.npz'),allow_pickle=False)as existing:
        features=deployment_features(path)
        _,labels=predict_together(models,{'discharge':np.pad(features,((0,0),(0,183)))})
        assert np.array_equal(existing['q95'],labels['q95']) and np.array_equal(existing['betaN'],labels['beta_n'])
    write_json(HERE/f'process_parity_{os.getpid()}.json',{'status':'real_public_prediction_bitexact_across_process_change','stem':path.stem,'frames':len(features),'models_unchanged':True})

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=qualify_process)as pool:cases=list(pool.map(predict_case,enumerate(files)))
    assert sum(c['frames']for c in cases)==193359
    write_json(HERE/'public_prediction_receipt.json',{'status':'all874_production_joint_predictions_complete','cases':cases,'frames':193359})
    source=Path(proto['baseline_map_archive']);source_sha='99a7df91679012e982f106daff0aa264da395c4f67362abe649d203fd556f901'
    assert digest(source)==source_sha
    out=HERE/'submission/diii_d_public_test.npz';assert not out.exists();shutil.copyfile(source,out)
    with np.load(source,allow_pickle=False)as old,zipfile.ZipFile(out,'a',compression=zipfile.ZIP_DEFLATED)as archive:
     original_maps={x.filename:(x.CRC,x.file_size,x.compress_size,x.header_offset)for x in archive.infolist()if x.filename.endswith('_psirz.npy')}
     assert len(original_maps)==874
     archive.filelist=[x for x in archive.filelist if x.filename.endswith('_psirz.npy')]
     archive.NameToInfo={x.filename:x for x in archive.filelist}
     for case in cases:
      i=case['index'];prefix=f'shot_{i:04d}'
      with np.load(HERE/'public_predictions'/(case['stem']+'.npz'),allow_pickle=False)as joint:
       for name in ('q95','betaN'):
        base=old[prefix+'_'+name];assert base.dtype==np.float32 and base.shape==(case['frames'],)
        pred=(.75*base.astype(float)+.25*joint[name]).astype(np.float32)
        assert np.isfinite(pred).all()
        buffer=io.BytesIO();np.save(buffer,pred,allow_pickle=False)
        archive.writestr(prefix+'_'+name+'.npy',buffer.getvalue())
    with zipfile.ZipFile(out)as archive:
     assert len(archive.infolist())==2622 and len({x.filename for x in archive.infolist()})==2622
     for name,values in original_maps.items():
      entry=archive.getinfo(name);assert(entry.CRC,entry.file_size,entry.compress_size,entry.header_offset)==values
    mast=Path(proto['mast_archive']);assert digest(mast)=='2febdf3e6baed5cefa4e48cf09a2a5a5d24f6830e1e55c2eee30db2b2acaed51'
    os.link(mast,HERE/'submission/mast_public_test.npz')
    write_json(HERE/'build_receipt.json',{'status':'production_archive_pair_built','DIII_D_shots':874,'DIII_D_frames':193359,
     'DIII_D_sha256':digest(out),'MAST_sha256':digest(mast),'maps_original_compressed_records_preserved':True,
     'scalar_weight':.25,'baseline_source_sha256':source_sha,'scalar_blend_arithmetic':'published_baseline_float32 promotedfloat64 plus joint8float64, thenfloat32archive',
     'public_labels_accessed':False,'submitted':False})
