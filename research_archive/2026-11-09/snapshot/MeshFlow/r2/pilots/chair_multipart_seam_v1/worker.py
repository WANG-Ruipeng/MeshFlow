"""Fixed final mixed-part generator evaluation. Original 44 noises, one D0."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'r2/src'))
import numpy as np
from .eval_prepare import ROOT,OUT,LABEL,G
from r2.pilots.chair_back_transfer_v1 import worker as BASE

GEN=OUT/'inference'/f'{LABEL}_generator.pt'
HEAD=OUT/'inference'/f'{LABEL}_occupancy.pt'
D0=BASE.D0


def rows(shard=0,shards=1):
    m=G.read(OUT/'eval_manifest.json');G.verify_files(m['protected'])
    ids=sorted({q['condition_id'] for q in m['requests']});owned=set(ids[shard::shards])
    return [q for q in m['requests'] if q['condition_id'] in owned]


def assets(q):
    for field in ('inference_npz','noise_npz','metadata_npz','source_npz'):
        if field in q and G.sha(q[field])!=q[field+'_sha256']:
            raise ValueError('Changed input '+field)
    bundle=G.read_bundle(q);source=G.source_from_bundle(q,bundle);G.read_noise(q)
    return bundle,source


def load_generator():
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    receipt=G.read(GEN.with_suffix('.json'));assert G.sha(GEN)==receipt['sha256']
    p=torch.load(GEN,map_location='cpu',weights_only=True,mmap=True)
    assert (p['global_step'],p['local_step'],p['EMA'])==(15500,1500,False)
    assert state_hash(p['generator_state'])==p['generator_state_sha256']
    model=build_model(p['model_config'],readout_mode='none',trainable=False)
    model.load_state_dict(p['generator_state'],strict=True)
    model.to('cuda').set_trainable(False).eval()
    return model,p['generator_state_sha256'],dict(path=str(GEN),sha256=receipt['sha256'],generator_state_sha256=p['generator_state_sha256'],global_step=15500,local_step=1500,EMA=False)


def layout(head,q,hid):
    from .head import predict_layout
    from meshflow_control.seam.slot_data import inference_layout
    import torch
    bundle,source=assets(q);dest=OUT/'evaluation'/LABEL/'layouts'/q['condition_id']
    if (dest/'receipt.json').exists():
        r=G.checked_receipt(dest/'receipt.json')
        assert r['head']==hid and r['C_bundle_sha256']==G.sha(q['inference_npz'])
        a=G.load_npz(dest/'layout.npz');lay=inference_layout(**bundle,occupied=a['occupied'])
        for k in ('anchor_ids','edge_records','seam_slots','bulk_slots'):assert G.same(a[k],lay[k])
        return lay,a['occupancy_logits'],source,dest
    if dest.exists():raise RuntimeError('Unresolved head call; do not silently repeat '+str(dest))
    dest.mkdir(parents=True);G.write(dest/'attempt.json',dict(condition_id=q['condition_id'],head=hid,source_labels_read=False))
    G.event(dest/'calls.jsonl','HEAD_FORWARD_STARTED',budget=1)
    with torch.autocast('cuda',dtype=torch.bfloat16):lay,logits=predict_layout(head,bundle)
    torch.cuda.synchronize();G.event(dest/'calls.jsonl','HEAD_FORWARD_RETURNED')
    G.save_npz(dest/'layout.npz',{**lay,'occupancy_logits':logits});G.save_npz(dest/'source_condition.npz',source)
    G.write(dest/'receipt.json',dict(status='COMPLETE',condition_id=q['condition_id'],head=hid,
        C_bundle_sha256=G.sha(q['inference_npz']),occupancy_forward_calls=1,threshold_logit=0.,
        active=int(lay['occupied'].sum()),boundary_edges=len(lay['occupied']),source_labels_read=False,
        files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay,logits,source,dest


def raw_run(qs,attempt):
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    from .head import MultipartOccupancyHead
    pending=[]
    for q in qs:
        assets(q);d=OUT/'evaluation'/LABEL/q['request_id']/'raw'
        if (d/'receipt.json').exists():G.checked_receipt(d/'receipt.json')
        elif d.exists():raise RuntimeError('Unresolved RAW call '+str(d))
        else:pending.append(q)
    if not pending:return dict(outputs=len(qs),new_rollouts=0,new_generator_forwards=0)
    runtime.configure_stable_runtime();model,expected,identity=load_generator()
    hr=G.read(HEAD.with_suffix('.json'));assert G.sha(HEAD)==hr['sha256']
    hp=torch.load(HEAD,map_location='cpu',weights_only=True,mmap=True)
    assert state_hash(hp['head_state'])==hp['head_state_sha256'] and hp['threshold_logit']==0.
    head=MultipartOccupancyHead(**hp['head_config']).eval().requires_grad_(False)
    head.load_state_dict(hp['head_state'],strict=True);head.to('cuda')
    hid=dict(path=str(HEAD),sha256=hr['sha256'],head_state_sha256=hp['head_state_sha256'])
    G.write(attempt/'model_identity.json',dict(generator=identity,head=hid));G.write(attempt/'resources_loaded.json',G.resources())
    layouts={}
    for i,q in enumerate(pending):
        cid=q['condition_id']
        if cid not in layouts:layouts[cid]=layout(head,q,hid)
        lay,logits,source,lp=layouts[cid];dest=OUT/'evaluation'/LABEL/q['request_id']/'raw';dest.mkdir(parents=True)
        G.write(dest/'attempt.json',dict(request=q,checkpoint=identity,forward_budget=50))
        raw,traj,audit=G.rollout(model,lay,G.read_noise(q),dest/'calls.jsonl')
        arrays=G.raw_arrays(raw,lay,logits,source)
        G.save_npz(dest/'raw.npz',arrays);G.save_npz(dest/'trajectory.npz',dict(state=traj));G.save_npz(dest/'source_condition.npz',source)
        write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces']);G.write(dest/'audit.json',audit)
        G.write(dest/'receipt.json',dict(status='COMPLETE',arm=LABEL,label=LABEL,request_id=q['request_id'],condition_id=cid,
            checkpoint=identity,raw_path=str(dest/'raw.npz'),raw_sha256=G.sha(dest/'raw.npz'),noise_sha256=G.sha(q['noise_npz']),
            input_bundle_sha256=G.sha(q['inference_npz']),layout_receipt=str(lp/'receipt.json'),generator_forwards=50,D0_forwards=0,
            source_labels_read=False,training_updates=0,audit=audit,files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}))
        G.write(attempt/f'resources_output{i+1}.json',G.resources());print('RAW',q['request_id'],i+1,len(pending),flush=True)
    assert state_hash(model.state_dict())==expected and state_hash(head.state_dict())==hp['head_state_sha256']
    return dict(outputs=len(qs),new_rollouts=len(pending),new_generator_forwards=50*len(pending))


def run(stage,shard=0,shards=1):
    reg=G.read(OUT/'inference_registration.json')
    assert reg['status']=='FROZEN' and G.sha(OUT/'eval_manifest.json')==reg['eval_manifest_sha256']
    G.verify_files(reg['protected']);G.verify_files(reg['source_hashes'])
    qs=rows(shard,shards)
    # Reuse the proven D0 path, but all its writes and barrier are redirected
    # to this experiment; original module/source and assets remain unchanged.
    BASE.OUT=OUT;BASE.LABEL=LABEL
    dest=OUT/'dispatch'/f'{stage}_shard{shard}_attempt{time.time_ns()}';dest.mkdir(parents=True)
    snap=G.resources();G.write(dest/'resources_before.json',snap)
    G.dispatch_guard(snap,2600 if stage=='raw' else 3500)
    G.write(dest/'dispatch.json',dict(stage=stage,shard=shard,shards=shards,requests=[q['request_id'] for q in qs],
        estimate_mib=2600 if stage=='raw' else 3500,no_automatic_neural_retry=True))
    tick=time.perf_counter()
    try:
        r=raw_run(qs,dest) if stage=='raw' else BASE.post_run(qs,dest)
        G.write(dest/'complete.json',dict(status='COMPLETE',wall_seconds=time.perf_counter()-tick,**r))
    except BaseException as e:
        G.write(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True));raise
    finally:G.write(dest/'resources_finished.json',G.resources())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('raw','post'),required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();run(a.stage,a.shard,a.shards)
