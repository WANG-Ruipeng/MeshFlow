"""Execute the locked handoff AE/FM budgets after a separate real-CUDA gate."""
from argparse import ArgumentParser,Namespace
from pathlib import Path
import json,datetime,time
import numpy as np
import torch
from .. import train_ours
from ..io_utils import file_hash,new_dir,write_json,save_npz,model_hash
from ..models import ShapeAE,fit_pca_train_only
from .official import configure

def main(data,out,gate):
    data=Path(data).resolve();gate=Path(gate).resolve();root=Path(__file__).resolve().parents[1]
    g=json.loads((gate/'GATE.json').read_text());assert g['status']=='PASS'
    assert g['source_and_data_unchanged'] and g['actual_optimizer_updates']==16
    assert g['original_trainer_sha256']==file_hash(root/'train_ours.py')
    assert g['data_hashes']=={n:file_hash(data/n) for n in g['data_hashes']}
    out=new_dir(out);sources=['train_ours.py','models.py','io_utils.py','sample_ours.py','export_ours.py','exporter.py','system_probe/ours.py','system_probe/ours_run.py','system_probe/official.py']
    protocol={'study':'AE-FM-CLOSED16-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'all16 development parents; no held-out generalization and no qualified external ranking',
        'data':str(data),'data_hashes':g['data_hashes'],'gate_sha256':file_hash(gate/'GATE.json'),
        'seeds':[10,11],'ae':{'steps':3200,'batch':32,'lr':6e-4,'checkpoint_every':400,'parameters':5383234},
        'fm':{'steps':8000,'batch':128,'lr':1e-3,'checkpoint_every':1000,'parameters':50240},
        'weight_decay':1e-4,'scheduler':'CosineAnnealingLR eta_min=lr*.1','gradient_clip':2.,
        'selection':'fixed final steps only','initialization':'fresh seed-specific rank<=15 PCA plus original zero-head AE; fresh FM',
        'sampling':{'samples_per_seed':64,'noise_seed_by_training_seed':{'10':73510,'11':73511},'euler_steps':32},
        'outlets':['fixed','original_archived_a_same_seed','vtk_qem'],'target_faces':512,'shared_mainfield_per_seed_sample':True,
        'PCA_reference':'recorded before optimization, not a selected learned checkpoint; PCA already interpolates train16',
        'external_baseline':'full-network gate passed; fixed50k formal training remains separate and pending',
        'training_source_sha256':{n:file_hash(root/n) for n in sources},
        'deterministic_algorithms':True,'tf32':False,'threads':1,'data_rotation_or_expansion':False}
    write_json(out/'RUN_PROTOCOL.json',protocol)
    with np.load(data/'radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr']).cuda();uids=z['uids'].copy()
    records=[];started=time.perf_counter()
    for seed in (10,11):
        configure(seed);pca=ShapeAE().cuda();info=fit_pca_train_only(pca,y)
        with torch.no_grad():recon,codes=pca(y)
        save_npz(out/f'pca_initial_seed{seed}.npz',uids=uids,nodes=recon.cpu().numpy(),codes=codes.cpu().numpy())
        write_json(out/f'PCA_INITIAL_{seed}.json',{'pca':info,'model_hash':model_hash(pca),
            'mean_node_mse':float((recon-y).square().mean()),'per_parent_node_mse':(recon-y).square().mean(-1).cpu().tolist()})
        del pca,recon,codes
        ae=None
        for stage in ('ae','fm'):
            configure(seed);settings=protocol[stage];target=out/f'seed{seed}'/stage
            args=Namespace(stage=stage,data=str(data),out=str(target),ae=ae,resume=None,seed=seed,
                device='cuda',threads=1,steps=settings['steps'],batch=settings['batch'],lr=settings['lr'],
                checkpoint_every=settings['checkpoint_every'],log_every=100)
            before=time.perf_counter();train_ours.run(args)
            final=target/f'step{settings["steps"]:08d}.pt';state=torch.load(final,map_location='cpu',weights_only=True)
            assert state['step']==settings['steps'] and state['scheduler']['last_epoch']==settings['steps']
            assert all(float(v['step'])==settings['steps'] for v in state['optimizer']['state'].values())
            records.append({'seed':seed,'stage':stage,'final_checkpoint':str(final.relative_to(out)),
                'final_sha256':file_hash(final),'elapsed_seconds':time.perf_counter()-before,
                'parent_exposures_each':settings['steps']*settings['batch']//16,'fixed_final':True})
            write_json(out/'PROGRESS.json',{'completed_stages':records,'scientific_fit_qualification':False},replace=(out/'PROGRESS.json').exists())
            if stage=='ae':ae=str(final)
    assert protocol['training_source_sha256']=={n:file_hash(root/n) for n in sources}
    assert protocol['data_hashes']=={n:file_hash(data/n) for n in protocol['data_hashes']}
    write_json(out/'TRAINING_COMPLETE.json',{'status':'COMPLETE','stages':records,'actual_optimizer_updates':22400,
        'elapsed_seconds':time.perf_counter()-started,'fixed_final_selection':True,'scientific_quality_pass':False})

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--gate',required=True)
    a=p.parse_args();main(a.data,a.out,a.gate)