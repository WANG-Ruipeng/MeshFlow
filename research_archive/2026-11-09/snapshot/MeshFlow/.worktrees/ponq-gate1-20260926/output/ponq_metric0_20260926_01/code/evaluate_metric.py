"""Two frozen models, exactly32 fresh noises each; no privileged field mixing."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,time,traceback
import numpy as np
import torch
from common import ROOT,dump,sha
from fm_model import total_velocity
from interaction_model import InteractionVelocity
from interaction_semantics import extended_semantic,METRIC_DEFINITIONS
from single_state_common import arrays,geometry,target_reference
from sample_single_state import assignment

def main(arm):
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=arrays(ROOT/'data.npz');target=data['states'][0]
    noise=arrays(ROOT/'sample_noise.npz')['noise'];assert noise.shape==(32,256,22)
    checkpoint=ROOT/'control/final.pt' if arm=='B0' else ROOT/'training/M0/final.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['status']=='COMPLETED' and ck['completed_steps']==9000 and ck['new_updates']==6000
    model=InteractionVelocity(arm='B0',width=128).cuda();model.load_state_dict(ck['model_state_dict']);model.eval()
    for key in ['mean','scale']:assert torch.equal(ck[key].cpu(),torch.tensor(data[key],dtype=torch.float32))
    mean=torch.tensor(data['mean'],dtype=torch.float32,device='cuda');scale=torch.tensor(data['scale'],dtype=torch.float32,device='cuda')
    out=ROOT/'evaluation'/arm;out.mkdir(parents=True,exist_ok=False)
    checkpoint_hash=sha(checkpoint);noise_hash=sha(ROOT/'sample_noise.npz')
    diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    endpoints=[];perms=[];rollrows=[];pure_seconds=[];post_seconds=[]
    started=time.perf_counter()
    for i in range(32):
        z=torch.tensor(noise[i:i+1],device='cuda');actual=0;physical=None;perm=np.full(256,-1,dtype=np.int64);duration=None
        torch.cuda.synchronize();begin=time.perf_counter()
        try:
            with torch.no_grad():
                for k in range(64):
                    t=torch.full((1,),k/64,device='cuda')
                    velocity,_=total_velocity(model,z,t,mean,scale,'direct')
                    # The training loss weights are NEVER read or applied here.
                    z=z+velocity/64;actual=k+1
                    if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite integration state')
            torch.cuda.synchronize();duration=time.perf_counter()-begin
            physical=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
            post_begin=time.perf_counter()
            perm=assignment(physical[:,:3],target[:,:3])
            sem=extended_semantic(physical,target[perm],diag)
            row=dict(status='COMPLETE',endpoint_semantic=sem)
            post_duration=time.perf_counter()-post_begin
        except Exception as exc:
            torch.cuda.synchronize();duration=time.perf_counter()-begin if duration is None else duration;post_duration=0.
            physical=np.full((256,22),np.nan) if physical is None else physical
            row=dict(status='ROLLOUT_OR_DIAGNOSTIC_FAILED',error=repr(exc),traceback=traceback.format_exc())
        row.update(arm=arm,sample_id=i,split='fresh',nfe=actual,
            generated=True,oracle_assisted=False,checkpoint_sha256=checkpoint_hash,
            integration_seconds=duration,posthoc_seconds=post_duration)
        endpoints.append(physical);perms.append(perm);rollrows.append(row);pure_seconds.append(duration);post_seconds.append(post_duration)
        if i%8==0:print('ROLLOUT',arm,i,row['status'],flush=True)
    np.savez_compressed(out/'endpoints.npz',states=np.stack(endpoints),target_permutations=np.stack(perms))
    dump(out/'ROLLOUT.json',dict(arm=arm,nfe=64,rows=rollrows,noise_sha256=noise_hash,checkpoint_sha256=checkpoint_hash,
        integration_seconds=sum(pure_seconds),posthoc_seconds=sum(post_seconds),wall_seconds=time.perf_counter()-started,
        timing_scope='Synchronized64Euler integration includes unchanged direct L diagnostics/finite checks; excludes initialization and endpoint transfer/matching/geometry.',
        training_weights_applied_to_velocity=False,metric_definitions=METRIC_DEFINITIONS))
    reference=target_reference();ledger=[];eval_start=time.perf_counter()
    for i,(pred,perm) in enumerate(zip(endpoints,perms)):
        available=bool(np.all(perm>=0) and len(np.unique(perm))==256)
        meta=dict(arm=arm,sample_id=i,split='fresh',kind='free_generation',combination='PPP',generated=True,oracle_assisted=False,
            nfe=64,checkpoint_sha256=checkpoint_hash,noise_sha256=noise_hash,matching_available=available,
            target_permutation=perm.tolist(),endpoint_semantic=rollrows[i].get('endpoint_semantic'),
            integration_seconds=pure_seconds[i],posthoc_seconds=post_seconds[i])
        row=geometry(pred,'fixed',out/f'{i:03d}_PPP_fixed',reference,408000+i,meta,perm if available else None)
        row['quality_pass']=bool(row['valid'] and row.get('rms_pct_bboxdiag',float('inf'))<=.20)
        row['quality_definition']='original geometry validity AND sampled symmetric surface RMS <=0.20% target bbox diagonal'
        dump(out/f'{i:03d}_PPP_fixed/result.json',row);ledger.append(row)
        print('FREE_GEOMETRY',arm,i,row['valid'],row['quality_pass'],flush=True);dump(out/'FREE_LEDGER_PARTIAL.json',ledger)
    dump(out/'FREE_LEDGER.json',ledger)
    summary=dict(arm=arm,count=32,extraction_count=32,
        valid=sum(r['valid'] for r in ledger),quality_pass=sum(r['quality_pass'] for r in ledger),
        extracted=sum(r['extraction_completed'] for r in ledger),
        extraction_evaluation_seconds=time.perf_counter()-eval_start)
    summary['advance']=summary['quality_pass']>=24
    dump(out/'SUMMARY.json',summary);print('EVALUATION_COMPLETE',summary,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['B0','M0'],required=True);main(ap.parse_args().arm)