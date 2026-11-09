"""The registered 16 fixed FM fixtures; no sampling or teacher is involved."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import training_data as data
from meshflow_control.data.io import read_json, digest, json_hash, freeze_json, atomic_json, save_npz

def freeze(out, manifest):
    out=data.checked_out(out); dataset=data.Dataset(manifest); m=dataset.manifest
    large=next(k for k in m['pools'] if k.startswith('USER_LARGE_'))
    sets={'SMALL_SEEN_LARGE_SEEN':sorted(m['pools']['USER_SMALL']),
          'SMALL_UNSEEN_LARGE_SEEN':sorted(set(m['pools'][large])-set(m['pools']['USER_SMALL']))}
    rows=[]
    for group,parents in sets.items():
        ranked=sorted(parents,key=lambda uid:(data.seed('diagnostic_parent',group,uid),uid))[:8]
        for uid in ranked:
            options=sorted([r for r in m['tasks'].values() if r['uid']==uid],
                key=lambda r:(data.seed('diagnostic_condition',r['task_id']),r['task_id']))
            i=len(rows); family=sorted(data.TYPES)[i%4]
            task=next((r for r in options if r['condition_type']==family),options[0]); ts=data.seed('diagnostic_time',i)
            alpha=1. if i%2==0 else float(np.float32(np.random.RandomState(data.seed('diagnostic_alpha',i)).uniform(.9,1.1)))
            rows.append(dict(sample_index=i,step=0,global_step=0,arm='FIXTURE',uid=uid,
                task_id=task['task_id'],N=task['N'],K=task['K'],ratio=task['K']/task['N'],alpha=alpha,
                clean=alpha==1.,condition_type=task['condition_type'],region_type=task['condition_type'].split('_')[0].lower(),
                t_seed=ts,t=float(data.training_time(ts)),group=group,
                gaussian_seed=data.seed('diagnostic_Gaussian',i),epsilon_seed=data.seed('diagnostic_epsilon',i),
                permutation_seed=data.seed('diagnostic_permutation',i)))
    plan=dict(schema='r2_scaling_fixed_diagnostics_v1',manifest_sha256=dataset.manifest_sha256,records=rows,
        count=16,steps=[0,1000,3000,'fixed_endpoint'],groups=sets,uses_DEV=False)
    folder=data.writable(out,out/'diagnostics');folder.mkdir(parents=True,exist_ok=True)
    ph=freeze_json(folder/'fixtures.json',plan)
    receipts=[]
    for offset in (0,8):
        samples=data.prepare_batch(out,dataset,rows[offset:offset+8],'diagnostic','FIXTURE',offset//8,ph)
        for s in samples:
            ident=s['sample_index'];path=folder/f'fixture_{ident:02d}.npz';rp=path.with_suffix('.json')
            if not rp.exists():
                if path.exists():raise RuntimeError('Incomplete diagnostic fixture packaging')
                arrays={k:v for k,v in s.items() if isinstance(v,np.ndarray)}
                scalars={k:v for k,v in s.items() if not isinstance(v,np.ndarray)}
                save_npz(path,**arrays)
                atomic_json(rp,dict(path=str(path),sha256=digest(path),metadata=scalars,record=rows[ident],plan_sha256=ph))
            r=read_json(rp)
            if r['sha256']!=digest(path) or r['plan_sha256']!=ph:raise ValueError('Frozen diagnostic changed')
            receipts.append(r)
    return dict(plan_sha256=ph,fixtures=receipts,actual_fixed_OT=16,model_forwards=0)

def load(out):
    out=data.checked_out(out);folder=out/'diagnostics';plan=read_json(folder/'fixtures.json');ph=digest(folder/'fixtures.json')
    result=[]
    for i,row in enumerate(plan['records']):
        receipt=read_json(folder/f'fixture_{i:02d}.json');path=Path(receipt['path'])
        if receipt['plan_sha256']!=ph or digest(path)!=receipt['sha256']:raise ValueError('Diagnostic identity differs')
        with np.load(path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        s.update(receipt['metadata']);data.validate_sample(s,row);s['diagnostic_group']=row['group'];result.append(s)
    return result

def measure(model,out,arm,step,ledger):
    import torch
    from meshflow_control.data.dataset import collate,model_inputs
    from meshflow_control.models.native import native_execution
    from meshflow_control.training.losses import masked_fm_loss
    from meshflow_control.checkpoints import state_hash
    out=data.checked_out(out);path=data.writable(out,out/'diagnostics'/arm/f'step_{step:04d}.json')
    ident=dict(arm=arm,step=step,generator_state_sha256=state_hash(model.state_dict()),
        fixtures_sha256=digest(out/'diagnostics/fixtures.json'))
    if path.exists():
        r=read_json(path)
        if r['identity']!=ident or r['status']!='PASS':raise ValueError('Completed diagnostic identity differs')
        return r
    values=[];device=next(model.parameters()).device
    with torch.no_grad():
        for s in load(out):
            batch=collate([s],device)
            with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
                velocity,_=ledger.call('diagnostic_forward',lambda:model(*model_inputs(batch)).float(),
                    dict(arm=arm,step=step,fixture=s['sample_index'],input_bytehash=s['input_bytehash']))
                loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite fixed FM diagnostic')
            values.append(dict(fixture=s['sample_index'],group=s['diagnostic_group'],uid=s['object_id'],
                condition_id=s['task_id'],N=int(s['y']),K=s['K'],free_scalar_count=s['free_coordinate_count'],FM=float(loss)))
    result=dict(status='PASS',identity=ident,rows=values,actual_forward=16,backward=0)
    atomic_json(path,result);return result
