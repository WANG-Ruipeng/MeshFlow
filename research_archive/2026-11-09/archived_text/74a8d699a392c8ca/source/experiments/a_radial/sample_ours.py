"""Sample ONCE, persist noise/standardized z/z/mainfield; outlets never resample."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from .models import ShapeAE,ShapeFM
from .io_utils import new_dir,write_json,save_npz,file_hash,array_hash,model_hash

def run(a):
    dev=torch.device(a.device);ae=ShapeAE().to(dev);fm=ShapeFM().to(dev)
    ae_hash=file_hash(a.ae);fm_hash=file_hash(a.fm)
    ac=torch.load(a.ae,map_location='cpu',weights_only=True);fc=torch.load(a.fm,map_location='cpu',weights_only=True)
    if fc['config']['ae_sha256']!=ae_hash:raise ValueError('FM is bound to a different AE checkpoint')
    if ac['config']['dataset_sha256']!=fc['config']['dataset_sha256']:raise ValueError('Data provenance mismatch')
    ae.load_state_dict(ac['model']);fm.load_state_dict(fc['model']);ae.eval().requires_grad_(False);fm.eval().requires_grad_(False)
    out=new_dir(a.out);gen=torch.Generator().manual_seed(a.noise_seed);noise=torch.randn((a.samples,64),generator=gen);records=[]
    for i in range(a.samples):
        eps=noise[i:i+1].to(dev)
        if dev.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter()
        try:
            with torch.no_grad():zs=fm.sample_standardized(eps,a.steps);z=fm.unstandardize(zs);nodes=ae.decode_nodes(z)
            if not torch.isfinite(nodes).all():raise FloatingPointError('Nonfinite decoded nodes')
            if dev.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            arrs={'noise':eps.cpu().numpy()[0],'z_standardized':zs.cpu().numpy()[0],'z':z.cpu().numpy()[0],'nodes':nodes.cpu().numpy()[0]}
            path=out/f'sample_{i:05d}.npz';save_npz(path,**arrs)
            records.append({'sample_id':f'{i:05d}','status':'ok','path':path.name,'array_hash':array_hash(*arrs.values()),'nodes_hash':array_hash(arrs['nodes']),'seconds_noise_to_nodes':elapsed})
        except Exception as e:records.append({'sample_id':f'{i:05d}','status':'failed','error':repr(e)})
        write_json(out/'SAMPLES.json',{'schema':1,'noise_seed':a.noise_seed,'requested':a.samples,'fm_steps':a.steps,'fm_nfe':a.steps,'ae_sha256':ae_hash,'fm_sha256':fm_hash,'dataset_sha256':ac['config']['dataset_sha256'],'dataset_manifest_sha256':ac['config']['dataset_manifest_sha256'],'sampling_device':str(dev),'records':records},replace=(out/'SAMPLES.json').exists())
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ae',required=True);p.add_argument('--fm',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--noise-seed',type=int,default=73510);p.add_argument('--samples',type=int,default=64);p.add_argument('--steps',type=int,default=32);run(p.parse_args())
