"""Free sampling of the single-target direct FM; reference used only after rollout."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,argparse,traceback
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import ROOT,dump,sha
from fm_model import DeepSetsVelocity,total_velocity
from single_state_common import arrays,semantic

def assignment(source,target):
    cost=((source[:,None,:]-target[None,:,:])**2).sum(-1)
    row,col=linear_sum_assignment(cost);assert np.array_equal(row,np.arange(len(source)))
    return col

def sample_model(nfe):
    data=arrays(ROOT/'data.npz');target=data['states'][0]
    noise=arrays(ROOT/'sample_noise.npz')['noise'];checkpoint=ROOT/'training'/'direct'/'final.pt'
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
    assert ck['status']=='COMPLETED' and ck['completed_steps']==3000
    model=DeepSetsVelocity(128).cuda();model.load_state_dict(ck['model_state_dict']);model.eval()
    mean=torch.tensor(data['mean'],dtype=torch.float32,device='cuda');scale=torch.tensor(data['scale'],dtype=torch.float32,device='cuda')
    out=ROOT/f'sampling_{nfe}';out.mkdir(exist_ok=False)
    endpoints=[];perms=[];rows=[];snapshots=[]
    marks=sorted(set([0,nfe//4,nfe//2,3*nfe//4,round(.9*nfe),round(.99*nfe),nfe]))
    diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    torch.cuda.synchronize();started=time.perf_counter()
    for i in range(32):
        z=torch.tensor(noise[i:i+1],device='cuda');actual=0;failure=None;history={}
        try:
            with torch.no_grad():
                history[0]=(z*scale+mean)[0].cpu().numpy()
                for k in range(nfe):
                    actual=k+1;t=torch.full((1,),k/nfe,device='cuda')
                    velocity,_=total_velocity(model,z,t,mean,scale,'direct');z=z+velocity/nfe
                    if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite integration state')
                    if actual in marks:history[actual]=(z*scale+mean)[0].cpu().numpy()
            physical=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
            # This target-based matching is strictly after the complete free rollout.
            perm=assignment(physical[:,:3],target[:,:3])
            initial=assignment(noise[i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3],target[:,:3])
            row=dict(sample_id=i,nfe=actual,status='COMPLETE',endpoint_semantic=semantic(physical,target[perm],diag),
                initial_coupling_endpoint_semantic=semantic(physical,target[initial],diag))
            row['free_snapshots_semantic']=[dict(step=k,time=k/nfe,semantic_to_initial_assigned_target=semantic(v,target[initial],diag)) for k,v in history.items()]
        except Exception as exc:
            failure=dict(error=repr(exc),traceback=traceback.format_exc());physical=np.full((256,22),np.nan);perm=np.full(256,-1,dtype=int)
            row=dict(sample_id=i,nfe=actual,status='ROLLOUT_FAILED',**failure)
        row['checkpoint_sha256']=sha(checkpoint);rows.append(row);endpoints.append(physical);perms.append(perm)
        snapshots.append(np.stack([history.get(k,np.full((256,22),np.nan)) for k in marks]))
        print('ROLLOUT',nfe,i,row['status'],flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    np.savez_compressed(out/'endpoints.npz',states=np.stack(endpoints),target_permutations=np.stack(perms),
        snapshots=np.stack(snapshots),snapshot_steps=np.array(marks))
    dump(out/'ROLLOUT.json',dict(nfe_requested=nfe,rows=rows,seconds=elapsed,timing_scope='rollout with trajectory snapshots plus posthoc matching and semantic diagnostics; not pure GPU integration',
        noise_sha256=sha(ROOT/'sample_noise.npz'),checkpoint_sha256=sha(checkpoint),
        note='Hungarian target access occurs after free integration only. Snapshot errors use one chosen initial coupling; these are not true off-path ODE velocity errors.'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nfe',type=int,default=64);sample_model(ap.parse_args().nfe)