"""Frozen checkpoint sampling on the pre-registered Bench multi-part panel.

No source completion or source owner is read during layout prediction. Closed C
uses zero seam edges and makes zero occupancy-head calls for that condition.
GPU execution occurs only under explicit CLI run; the controller owns waves.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import panel as P


def requests(root,subset):
    manifest=P.read(P.resolve(root)/'evaluation_inputs/manifest.json')
    process=set(manifest['process_request_ids'])
    return [q for q in manifest['requests'] if subset=='full' or q['request_id'] in process]


def checked_request(q):
    for key,digest in [('inference_npz','input_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_sha256'),('noise_npz','noise_sha256')]:
        if P.sha(q[key])!=q[digest]:raise ValueError('Frozen request changed: '+key)


def run(root,checkpoint,label,mode='SEAM',subset='full',estimated_peak_mib=3500):
    if Path(label).name!=label or label in ('.','..'):raise ValueError('Unsafe label')
    if mode not in ('PLAIN','SEAM'):raise ValueError('Explicit PLAIN/SEAM required')
    root=P.resolve(root);checkpoint=P.resolve(checkpoint);qs=requests(root,subset)
    out=root/'evaluation'/label;done_path=out/('raw_complete_'+subset+'.json')
    checkpoint_sha=P.sha(checkpoint)
    if done_path.exists():
        done=P.read(done_path)
        if done['checkpoint_sha256']!=checkpoint_sha or done['mode']!=mode:raise ValueError('Frozen checkpoint differs')
        for path,digest in done['files'].items():
            if P.sha(path)!=digest:raise ValueError('Completed RAW changed')
        return done
    # Model imports are deliberately deferred until an explicit sampling run.
    import torch
    from meshflow_control import runtime
    from meshflow_control.config import MODEL_CONFIG
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    from meshflow_control.seam.occupancy import OccupancyHead
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import train as T, sample as OLD
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays, dispatch_guard
    from . import data as D
    qs=[dict(q) for q in qs]
    for q in qs:checked_request(q)
    out.mkdir(parents=True,exist_ok=True)
    registration=dict(label=label,mode=mode,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        evaluation_manifest_sha256=P.sha(root/'evaluation_inputs/manifest.json'),code_sha256=P.sha(__file__),
        original_rollout_sha256=P.sha(OLD.__file__),data_adapter_sha256=P.sha(D.__file__),
        solver='Euler50 standard Gaussian FP32 integration',source_owner_read=False)
    P.immutable(out/'sampling_identity.json',registration)
    pending=[]
    for q in qs:
        dest=out/q['request_id']/'raw'
        if (dest/'receipt.json').exists():
            r=P.read(dest/'receipt.json')
            if r['checkpoint_sha256']!=checkpoint_sha or r['raw_sha256']!=P.sha(dest/'raw.npz'):raise ValueError('Saved request differs')
        elif dest.exists():raise RuntimeError('Unresolved RAW attempt; no automatic resampling: '+str(dest))
        else:pending.append(q)
    if not pending:
        receipts=[out/q['request_id']/'raw/receipt.json' for q in qs]
        result=dict(status='COMPLETE',label=label,mode=mode,subset=subset,outputs=len(qs),new_rollouts=0,new_forwards=0,
            checkpoint_sha256=checkpoint_sha,files={str(p):P.sha(p) for p in receipts})
        P.immutable(done_path,result);return result
    attempt=out/('sampling_attempt_'+str(time.time_ns()));attempt.mkdir()
    snapshot=T.resources();T.write(attempt/'resources_before.json',snapshot)
    dispatch_guard(snapshot,estimated_peak_mib)
    T.write(attempt/'dispatch.json',dict(estimated_process_peak_mib=estimated_peak_mib,
        headroom_rule='max(2GiB,10%total) plus 1.2x comparable peak',workers=1,controller_owns_concurrency=True))
    model=head=None;count=head_calls=0;started=time.perf_counter()
    try:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        state=payload.get('generator_state',payload.get('model_state'))
        if state is None:raise ValueError('Checkpoint lacks generator_state')
        expected=payload.get('generator_state_sha256') or state_hash(state)
        if state_hash(state)!=expected:raise ValueError('Generator state hash differs')
        config=payload.get('model_config',MODEL_CONFIG)
        if config!=MODEL_CONFIG:raise ValueError('Model architecture differs from category protocol')
        runtime.configure_stable_runtime()
        model=build_model(copy.deepcopy(config),readout_mode='none',trainable=False).eval()
        model.load_state_dict(state,strict=True);model.to('cuda').set_trainable(False)
        if mode=='SEAM':
            hstate=payload.get('head_state')
            if hstate is None:raise ValueError('SEAM checkpoint lacks C-only occupancy head')
            head=OccupancyHead(24).eval().requires_grad_(False);head.load_state_dict(hstate,strict=True);head.to('cuda')
        T.write(attempt/'resources_loaded.json',T.resources());layouts={}
        for q in pending:
            cid=q['condition_id'];bundle=T.arrays(q['inference_npz']);source=T.arrays(q['source_npz'])
            core={k:bundle[k] for k in ('C_vertices','C_faces_original','C_faces_model')};core['total_N']=int(bundle['N'])
            if int(bundle['N'])!=q['N'] or int(bundle['K'])!=q['K']:raise ValueError('N/K mismatch')
            if cid not in layouts:
                dest=out/'layouts'/cid
                if (dest/'receipt.json').exists():
                    receipt=P.read(dest/'receipt.json');a=T.arrays(dest/'layout.npz')
                    if receipt['checkpoint_sha256']!=checkpoint_sha or receipt['layout_sha256']!=P.sha(dest/'layout.npz'):raise ValueError('Saved layout differs')
                    logits=a['occupancy_logits'];layout=D.inference_layout(**core,occupied=a['occupied'])
                else:
                    if dest.exists():raise RuntimeError('Unresolved layout forward; no silent retry')
                    dest.mkdir(parents=True)
                    T.write(dest/'attempt.json',dict(condition_id=cid,source_labels_read=False,checkpoint_sha256=checkpoint_sha))
                    edges,features=D.edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
                    if len(edges)!=q['boundary_edges']:raise ValueError('C-only boundary changed')
                    forward=int(mode=='SEAM' and len(edges)>0)
                    if forward:
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=head(torch.from_numpy(features).cuda(),torch.ones(len(edges),dtype=torch.bool,device='cuda')).float().cpu().numpy()
                        head_calls+=1;occupied=logits>=0
                    else:logits=np.zeros(len(edges),np.float32);occupied=np.zeros(len(edges),bool)
                    layout=D.inference_layout(**core,occupied=occupied)
                    P.array_file(dest/'layout.npz',{**layout,'occupancy_logits':logits})
                    P.array_file(dest/'source_condition.npz',source)
                    T.write(dest/'receipt.json',dict(status='COMPLETE',condition_id=cid,checkpoint_sha256=checkpoint_sha,
                        occupancy_forward=forward,active=int(occupied.sum()),edges=len(edges),threshold=0.,closed_C=len(edges)==0,
                        layout_sha256=P.sha(dest/'layout.npz'),source_labels_read=False))
                layouts[cid]=(layout,logits)
            layout,logits=layouts[cid];noise=T.arrays(q['noise_npz'])['z'];dest=out/q['request_id']/'raw';dest.mkdir(parents=True)
            T.write(dest/'attempt.json',dict(label=label,mode=mode,request=q,checkpoint_sha256=checkpoint_sha,forward_budget=50))
            raw,traj,audit=OLD.rollout(model,layout,noise,mode,dest/'calls.jsonl')
            arrays=raw_arrays(raw,layout,logits,source)
            P.array_file(dest/'raw.npz',arrays);P.array_file(dest/'trajectory.npz',dict(state=traj));P.array_file(dest/'source_condition.npz',source)
            write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces'])
            T.write(dest/'receipt.json',dict(status='COMPLETE',label=label,arm=label,mode=mode,request_id=q['request_id'],
                condition_id=cid,checkpoint_sha256=checkpoint_sha,raw_sha256=P.sha(dest/'raw.npz'),noise_sha256=q['noise_sha256'],
                source_labels_read=False,D0_forwards=0,**audit))
            count+=1
            if count in (1,2) or count%8==0:T.write(attempt/('resources_output'+str(count)+'.json'),T.resources())
            print(json.dumps(dict(stage='RAW',label=label,completed=count,total=len(pending),request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=expected:raise RuntimeError('Frozen model changed')
        receipts=[out/q['request_id']/'raw/receipt.json' for q in qs]
        result=dict(status='COMPLETE',label=label,mode=mode,subset=subset,outputs=len(qs),new_rollouts=count,new_forwards=50*count,
            new_head_forwards=head_calls,checkpoint_sha256=checkpoint_sha,seconds=time.perf_counter()-started,
            files={str(p):P.sha(p) for p in receipts})
        P.immutable(done_path,result);return result
    except BaseException as exc:
        T.write(attempt/'failure.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),completed_rollouts=count,
            completed_head_calls=head_calls,no_automatic_retry=True));raise
    finally:
        T.write(attempt/'resources_finished.json',T.resources())
        del model,head
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(P.OUT));p.add_argument('--checkpoint',required=True)
    p.add_argument('--label',required=True);p.add_argument('--mode',choices=('PLAIN','SEAM'),required=True)
    p.add_argument('--panel',choices=('full','process'),default='full');p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args();print(json.dumps(run(a.root,a.checkpoint,a.label,a.mode,a.panel,a.estimated_peak_mib)))
