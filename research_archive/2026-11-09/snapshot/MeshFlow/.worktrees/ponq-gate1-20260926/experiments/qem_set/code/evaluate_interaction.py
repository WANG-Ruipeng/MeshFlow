"""Frozen 64-noise free generation and 24 oracle field diagnostics per arm."""
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

ARMS=['B0','B1','B2']
IDS=[0,4,8,12,16,20,24,28]
COMBINATIONS=['POO','OPO','OOP']
def main(arm):
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=arrays(ROOT/'data.npz');target=data['states'][0];frozen=arrays(ROOT/'sample_noise.npz')
    noise=frozen['noise'];assert noise.shape==(64,256,22)
    checkpoint=ROOT/'training'/arm/'final.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['status']=='COMPLETED' and ck['completed_steps']==9000 and ck['new_updates']==6000
    model=InteractionVelocity(arm=arm,width=128).cuda();model.load_state_dict(ck['model_state_dict']);model.eval()
    mean=torch.tensor(data['mean'],dtype=torch.float32,device='cuda');scale=torch.tensor(data['scale'],dtype=torch.float32,device='cuda')
    out=ROOT/'evaluation'/arm;out.mkdir(parents=True,exist_ok=False)
    checkpoint_hash=sha(checkpoint);noise_hash=sha(ROOT/'sample_noise.npz')
    diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    endpoints=[];perms=[];rollrows=[];pure_seconds=[];post_seconds=[]
    started=time.perf_counter()
    for i in range(64):
        z=torch.tensor(noise[i:i+1],device='cuda');actual=0;physical=None;perm=np.full(256,-1,dtype=np.int64);duration=None
        torch.cuda.synchronize();begin=time.perf_counter()
        try:
            with torch.no_grad():
                for k in range(64):
                    t=torch.full((1,),k/64,device='cuda');velocity,_=total_velocity(model,z,t,mean,scale,'direct')
                    z=z+velocity/64;actual=k+1
                    if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite integration state')
            torch.cuda.synchronize();duration=time.perf_counter()-begin
            physical=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
            post_begin=time.perf_counter()
            # Post-hoc target matching after complete unconstrained rollout. This
            # single permutation also binds all later field interventions.
            perm=assignment(physical[:,:3],target[:,:3])
            sem=extended_semantic(physical,target[perm],diag)
            row=dict(status='COMPLETE',endpoint_semantic=sem)
            post_duration=time.perf_counter()-post_begin
        except Exception as exc:
            torch.cuda.synchronize();duration=time.perf_counter()-begin if duration is None else duration;post_duration=0.
            physical=np.full((256,22),np.nan) if physical is None else physical
            row=dict(status='ROLLOUT_OR_DIAGNOSTIC_FAILED',error=repr(exc),traceback=traceback.format_exc())
        row.update(arm=arm,sample_id=i,split='old' if i<32 else 'new',split_id=i%32,nfe=actual,
            generated=True,oracle_assisted=False,checkpoint_sha256=checkpoint_hash,
            integration_seconds=duration,posthoc_seconds=post_duration)
        endpoints.append(physical);perms.append(perm);rollrows.append(row);pure_seconds.append(duration);post_seconds.append(post_duration)
        if i%8==0:print('ROLLOUT',arm,i,row['status'],flush=True)
    np.savez_compressed(out/'endpoints.npz',states=np.stack(endpoints),target_permutations=np.stack(perms))
    dump(out/'ROLLOUT.json',dict(arm=arm,nfe=64,rows=rollrows,noise_sha256=noise_hash,checkpoint_sha256=checkpoint_hash,
        integration_seconds=sum(pure_seconds),posthoc_seconds=sum(post_seconds),wall_seconds=time.perf_counter()-started,
        timing_scope='Integration sum: GPU synchronized before/after each 64-step rollout; includes unchanged direct total_velocity L diagnostics and finite checks; excludes model load, initial noise transfer, endpoint transfer, posthoc matching and geometry.',
        metric_definitions=METRIC_DEFINITIONS))
    reference=target_reference();ledger=[];interventions=[];eval_start=time.perf_counter()
    for i,(pred,perm) in enumerate(zip(endpoints,perms)):
        available=bool(np.all(perm>=0) and len(np.unique(perm))==256)
        meta=dict(arm=arm,sample_id=i,split='old' if i<32 else 'new',split_id=i%32,
            kind='free_generation',combination='PPP',generated=True,oracle_assisted=False,
            nfe=64,checkpoint_sha256=checkpoint_hash,noise_sha256=noise_hash,matching_available=available,
            target_permutation=perm.tolist(),endpoint_semantic=rollrows[i].get('endpoint_semantic'),
            integration_seconds=pure_seconds[i],posthoc_seconds=post_seconds[i])
        row=geometry(pred,'fixed',out/f'{i:03d}_PPP_fixed',reference,406000+i,meta,perm if available else None)
        row['quality_pass']=bool(row['valid'] and row.get('rms_pct_bboxdiag',float('inf'))<=.20)
        row['quality_definition']='original geometry validity AND sampled symmetric surface RMS <=0.20% target bbox diagonal'
        dump(out/f'{i:03d}_PPP_fixed/result.json',row);ledger.append(row)
        print('FREE_GEOMETRY',arm,i,row['valid'],row['quality_pass'],flush=True)
        dump(out/'FREE_LEDGER_PARTIAL.json',ledger)
    for i in IDS:
        pred,perm=endpoints[i],perms[i];available=bool(np.all(perm>=0) and len(np.unique(perm))==256)
        aligned=target[perm] if available else None
        for combo in COMBINATIONS:
            state=np.concatenate([pred[:,a:b] if c=='P' else aligned[:,a:b] for c,(a,b) in zip(combo,[(0,3),(3,6),(6,22)])],axis=1) if available else np.full((256,22),np.nan)
            meta=dict(arm=arm,sample_id=i,split='old',split_id=i,kind='oracle_field_intervention',
                combination=combo,generated=False,oracle_assisted=True,nfe=64,
                checkpoint_sha256=checkpoint_hash,noise_sha256=noise_hash,matching_available=available,
                target_permutation=perm.tolist(),field_definition='P=actual extractor v/raw n/centered F; decode world Q at selected P')
            row=geometry(state,'fixed',out/f'{i:03d}_{combo}_fixed',reference,406000+i,meta,perm if available else None)
            interventions.append(row);print('FIELD',arm,i,combo,row['valid'],flush=True)
    dump(out/'FREE_LEDGER.json',ledger);dump(out/'FIELD_LEDGER.json',interventions)
    summary=dict(arm=arm,free_count=64,intervention_count=24,extraction_evaluation_seconds=time.perf_counter()-eval_start,
        splits={s:dict(count=32,valid=sum(r['valid'] for r in ledger if r['split']==s),
            quality_pass=sum(r['quality_pass'] for r in ledger if r['split']==s),
            extracted=sum(r['extraction_completed'] for r in ledger if r['split']==s)) for s in ['old','new']},
        field_valid={c:sum(r['valid'] for r in interventions if r['combination']==c) for c in COMBINATIONS})
    summary['advance']=summary['splits']['new']['quality_pass']>=24
    dump(out/'SUMMARY.json',summary);print('EVALUATION_COMPLETE',summary,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);main(ap.parse_args().arm)