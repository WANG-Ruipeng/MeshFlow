"""Finite CPU gates: source restoration, exact old arithmetic, masks and N range.
No neural model construction/forward, GPU, or optimizer update.
"""
from __future__ import annotations
import ast, inspect, json, textwrap, time
from pathlib import Path
import numpy as np
from .common import *


def main():
    import torch
    from . import runtime as R, training as T, stream as Q
    from meshflow_control.seam import model_adapter as old_adapter
    from r2.pilots.bench_parts_convergence_v1 import data as D
    from meshflow_control.checkpoints import state_hash
    started=time.perf_counter()
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only process required')
    torch.set_num_threads(1)
    rng=torch.get_rng_state().clone()
    source_functions=(('seam_forward',old_adapter.seam_forward,R.seam_forward),)
    for name,a,b in source_functions:
        original=textwrap.dedent(inspect.getsource(a)).replace('(y > 256)','(y > 799)').replace('N128..256','N128..799')
        actual=textwrap.dedent(inspect.getsource(b))
        if ast.dump(ast.parse(original))!=ast.dump(ast.parse(actual)):
            raise ValueError('SEAM arithmetic changed beyond explicit N guard')
    m=read(OUT/'data/manifest.json');ev=read(OUT/'data/eval_manifest.json')
    schedule=read(OUT/'data/schedule_SEAM1500.json');Q.validate(schedule,m['train'])
    source_meta=dict(SOURCE_IDENTITIES,source_checkpoint=str(SOURCE))
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('B source bytes changed')
    payload=T._source_payload(source_meta)
    states=payload['optimizer']['state'];groups=payload['optimizer']['param_groups']
    if len(groups)!=1 or len(states)!=len(payload['generator_state']):raise ValueError('Original AdamW groups/states differ')
    if any(int(s['step'])!=22500 for s in states.values()):raise ValueError('AdamW does not resume exactly22500')
    g=groups[0]
    if (g['lr'],tuple(g['betas']),g['weight_decay'],g['eps'])!=(1e-5,(.9,.95),0,1e-8):raise ValueError('AdamW hyperparameters differ')
    if payload['head_state_sha256']!=T.HEAD_STATE_SHA256:raise ValueError('Wrong retained historical head')
    restore=dict(generator_state_sha256=payload['generator_state_sha256'],optimizer_state_sha256=payload['optimizer_state_sha256'],
        rng_sha256=payload['rng_sha256'],AdamW_parameter_steps=22500,AdamW_states=len(states),head_state_sha256=payload['head_state_sha256'],
        source_schema=payload['schema'],head_optimizer=None,head_training_forwards=0)
    del payload,states
    row_lookup={r['condition_id']:r for r in m['train']}
    samples=[];ot_calls=0
    for cid,row in row_lookup.items():
        source=D.load_case(row)
        first=next(s for entry in schedule['steps'] for s in entry['samples'] if s['condition_id']==cid and s['augmentation']=='CLEAN')
        pair=next(entry['samples'][i:i+2] for entry in schedule['steps'] for i in (0,2,4,6) if entry['samples'][i]['condition_id']==cid)
        for item in pair:
            sample=R.sample_for_training(source,item,'SEAM');ot_calls+=1
            facts=R.verify_sample(sample,'SEAM')
            scaled=source['full_target'].copy();scaled[...,0]*=np.float32(item['alpha'])
            roundtrip=scaled[sample['source_face_ids'][:,None],sample['source_corner_permutations']]
            if not np.array_equal(sample['x1'],roundtrip):raise ValueError('Source full target changed or dropped')
            if not np.array_equal(np.sort(sample['source_face_ids']),np.arange(len(scaled))):raise ValueError('Nonbijective source faces')
            anchor=sample['anchor_ids'];fixed=anchor>=0
            if not np.array_equal(sample['x0'][fixed],sample['x1'][fixed]) or sample['u'][fixed].any():raise ValueError('Anchor target moved')
            if not np.allclose(sample['xt']+(1-float(sample['t']))*sample['u'],sample['x1'],rtol=2e-6,atol=2e-6):raise ValueError('FM endpoint identity')
            compared=False
            if int(row['N'])<=256:
                old=D.sample_for_training(source,item,'SEAM');ot_calls+=1
                if set(old)!=set(sample) or any(not np.array_equal(old[k],sample[k]) for k in old):raise ValueError('Original-range packing changed: '+cid)
                compared=True
            samples.append(dict(condition_id=cid,augmentation=item['augmentation'],old_exact=compared,**facts))
    layouts=[]
    for q in {q['condition_id']:q for q in ev['requests']}.values():
        a=D.arrays(q['inference_npz']);n=int(q['N'])
        e,f=R.edge_features(a['C_vertices'],a['C_faces_original'],n)
        for occupancy in (np.zeros(len(e),bool),np.ones(len(e),bool)):
            layout=R.inference_layout(a['C_vertices'],a['C_faces_original'],a['C_faces_model'],n,occupancy)
            fixed=layout['anchor_ids']>=0
            if int((~fixed).sum())*3!=layout['free_coordinate_count']:raise ValueError('Scalar mask count')
            if not np.array_equal(layout['C_vertices'][layout['anchor_ids'][:q['K']]],a['C_vertices'][a['C_faces_model']]):raise ValueError('C preservation')
            if n<=256:
                old=D.inference_layout(a['C_vertices'],a['C_faces_original'],a['C_faces_model'],n,occupancy)
                if set(old)!=set(layout) or any(not np.array_equal(old[k],layout[k]) for k in old):raise ValueError('Original C-only layout changed')
        layouts.append(dict(condition_id=q['condition_id'],N=n,K=q['K'],boundary_edges=len(e),closed_C=len(e)==0,
            predicted_head_calls_when_sampled=int(len(e)>0),all_off_and_all_on_compatible=True,source_Y_read=False))
    # Actual tiny-tensor autodiff verifies fixed corners never enter loss/denominator.
    p=torch.arange(54,dtype=torch.float32).reshape(1,6,9).clone().requires_grad_()
    target=torch.zeros_like(p);valid=torch.ones((1,6),dtype=torch.bool)
    fixed=torch.zeros((1,6,3),dtype=torch.bool);fixed[:,:2]=True;fixed[:,2,:2]=True
    mask=(~fixed)[...,None].expand(1,6,3,3).reshape_as(p)
    den=int(mask.sum());loss=R.corner_masked_fm_loss(p,target,valid,fixed,denominator=den)
    loss.backward()
    if not torch.equal(p.grad[~mask],torch.zeros_like(p.grad[~mask])) or not torch.allclose(p.grad[mask],2*p.detach()[mask]/den):raise ValueError('Corner gradients differ')
    if not torch.equal(rng,torch.get_rng_state()) or torch.cuda.is_initialized():raise ValueError('CPU gate consumed global RNG/initialized CUDA')
    source_paths=[Path(__file__),Path(R.__file__),Path(T.__file__),Path(Q.__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('prepare.py')]
    result=dict(status='PASS',source_identities=SOURCE_IDENTITIES,source_hashes={str(p):sha(p) for p in source_paths},
        manifest_sha256=sha(OUT/'data/manifest.json'),eval_manifest_sha256=sha(OUT/'data/eval_manifest.json'),
        source_restore=restore,actual_CPU_samples=samples,actual_CPU_OT_calls=ot_calls,eval_layouts=layouts,
        model_constructions=0,model_forwards=0,model_backwards=0,optimizer_updates=0,head_forwards=0,
        tiny_tensor_autograd=True,N_range=[128,799],same_SEAM_forward_except_N_guard=True,
        new_parameters=0,source_owner_train_C_only_inference=True,closed_C_no_head_applicability='NA_NO_OPEN_BOUNDARY',
        seconds=time.perf_counter()-started)
    write_new(OUT/'training_cpu_gate.json',result)
    write_new(OUT/'runtime_cpu_gate.json',dict(status='PASS',source_hashes=result['source_hashes'],
        contract='Same original SEAM math except N128..799 guard; literal inherited Native model/runtime',model_forwards=0))
    print(json.dumps(dict(status='PASS',training_parents=len(row_lookup),CPU_samples=len(samples),CPU_OT_calls=ot_calls,
        evaluation_conditions=len(layouts),closed_C_conditions=sum(r['closed_C'] for r in layouts),
        maximum_evaluation_N=max(r['N'] for r in layouts),seconds=result['seconds'])))
    return result

if __name__=='__main__':main()
