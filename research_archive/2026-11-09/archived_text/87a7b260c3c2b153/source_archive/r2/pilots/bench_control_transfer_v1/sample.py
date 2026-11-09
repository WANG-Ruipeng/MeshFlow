"""Frozen final Bench PLAIN/SEAM, same saved Gaussian arrays, Euler50 only.

The inference input deliberately contains C and N only; source labels and the
unknown complete target are never consulted by a model or layout predictor.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, copy, json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
from . import train as T
from meshflow_control import runtime
import torch
from meshflow_control.checkpoints import state_hash
from meshflow_control.models.native import build_model,native_execution
from meshflow_control.seam.occupancy import OccupancyHead
from meshflow_control.seam.slot_data import edge_features,inference_layout
from meshflow_control.seam.model_adapter import seam_forward
from meshflow_control.source_winding import capture_source,validate_source
from meshflow_control.geo_io import write_indexed_ply
from r2.pilots.seam_slot_c200_v1.generate import raw_arrays

def save_npz(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def requests(manifest):
    if 'requests' in manifest:return manifest['requests']
    if 'evaluation_requests' in manifest:return manifest['evaluation_requests']
    if 'evaluation' in manifest:return manifest['evaluation']['requests']
    raise KeyError('Frozen manifest requires requests')

def input_only(q):
    """Read a separate C-only NPZ if supplied; else select just C fields."""
    p=T.resolve(next(q[k] for k in ('inference_path','inference_npz','case_path') if k in q))
    with np.load(p,allow_pickle=False) as z:
        bundle={k:z[k].copy() for k in ('C_vertices','C_faces_original','C_faces_model')}
        n=int(z['N']) if 'N' in z else int(q['N'])
        table=z['C_source_vertex_ids'].copy() if 'C_source_vertex_ids' in z else np.arange(len(bundle['C_vertices']),dtype=np.int64)
        faceids=z['known_ids'].copy() if 'known_ids' in z else np.arange(len(bundle['C_faces_original']),dtype=np.int64)
    bundle['total_N']=n
    original,canonical=bundle['C_faces_original'],bundle['C_faces_model']
    permutation=np.asarray([[np.flatnonzero(original[i]==v).item() for v in row] for i,row in enumerate(canonical)],np.int64)
    source=capture_source(bundle['C_vertices'][original],source_vertex_ids=table[original],corner_permutation=permutation,source_face_ids=faceids)
    validate_source(source,bundle['C_vertices'][canonical])
    return bundle,source

def freeze_inputs(root):
    """CPU-only create shared fixed-noise inputs before either model samples."""
    root=T.resolve(root);reg=T.checked(root);manifest=T.read(T.resolve(reg['manifest']));qs=requests(manifest)
    dest=root/'evaluation_inputs';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():return T.read(dest/'manifest.json')
    rows=[]
    for q0 in qs:
        q=dict(q0);bundle,source=input_only(q);rid=q['request_id'];folder=dest/rid
        if folder.exists():raise RuntimeError('Partial input freeze requires explicit audit, not silent overwrite')
        folder.mkdir();n=bundle['total_N'];k=len(bundle['C_faces_model'])
        save_npz(folder/'condition.npz',{**bundle,'N':np.asarray(n)})
        save_npz(folder/'source_condition.npz',source)
        seed=int(q.get('noise_seed',q.get('seed')))
        noise=np.random.RandomState(seed).standard_normal((n-k,3,3)).astype(np.float32)
        save_npz(folder/'noise.npz',dict(z=noise,seed=np.asarray(seed,np.int64)))
        q.update(N=n,K=k,seed=seed,inference_npz=str(folder/'condition.npz'),source_npz=str(folder/'source_condition.npz'),noise_npz=str(folder/'noise.npz'),
            input_sha256=T.sha(folder/'condition.npz'),source_sha256=T.sha(folder/'source_condition.npz'),noise_sha256=T.sha(folder/'noise.npz'))
        q.pop('case_path',None);q.pop('inference_path',None);rows.append(q)
    r=dict(schema='bench_frozen_C_only_evaluation_v1',status='FROZEN',arms=list(T.ARMS),requests=rows,outputs_per_arm=len(rows),solver='Euler50',
        registration_sha256=T.sha(root/'registration.json'),noise='same stored standard Gaussian per request shared across PLAIN and SEAM',source_labels_available_to_models=False)
    T.write(dest/'manifest.json',r);return r

def rollout(model,layout,noise,arm,ledger):
    n,k=layout['N'],layout['K'];fixed=layout['anchor_ids']>=0
    initial=np.zeros((n,3,3),np.float32);initial[k:]=noise;initial[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
    state=torch.from_numpy(initial.reshape(1,n,9)).cuda();anchor=torch.from_numpy(fixed[None]).cuda();free=~anchor
    condition=torch.from_numpy(initial[None]).cuda();known=torch.from_numpy(layout['known_mask'][None]).cuda();valid=torch.ones_like(known)
    y=torch.tensor([n],dtype=torch.int64,device='cuda');trajectory=[initial.copy()]
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.no_grad(),native_execution(model,coordinates=state):
        for step in range(50):
            t=torch.full((1,),step/50,dtype=torch.float32,device='cuda');T.event(ledger,'FORWARD_STARTED',step=step)
            v=seam_forward(model,state,t,y,valid,known,anchor) if arm=='SEAM' else model(state,t,y,valid_mask=valid,known_mask=known)
            torch.cuda.synchronize();T.event(ledger,'FORWARD_COMPLETE',step=step)
            if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite velocity')
            cur=state.reshape(1,n,3,3);nxt=cur.clone();nxt[free]=cur[free]+v.float().reshape_as(cur)[free]*(1/50);nxt[anchor]=condition[anchor]
            if not bool(torch.isfinite(nxt).all()):raise FloatingPointError('Nonfinite rollout')
            state=nxt.reshape(1,n,9);trajectory.append(nxt[0].cpu().numpy().copy())
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):raise RuntimeError('Unexpected sampling RNG')
    traj=np.stack(trajectory)
    if any(not np.array_equal(s[fixed],initial[fixed]) for s in traj):raise ValueError('Anchor moved')
    return traj[-1],traj,dict(generator_forwards=50,seconds=time.perf_counter()-tick,anchored_all51_exact=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def run(root,arm):
    root=T.resolve(root);reg=T.checked(root);manifest=T.read(root/'evaluation_inputs/manifest.json')
    if manifest['registration_sha256']!=T.sha(root/'registration.json'):raise ValueError('Evaluation registration differs')
    out=root/'evaluation'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'raw_complete.json').exists():return T.read(out/'raw_complete.json')
    checkpoint=root/'training'/arm/f'step{reg["updates"]}.pt';receipt=T.read(checkpoint.with_suffix('.json'))
    if T.sha(checkpoint)!=receipt['sha256']:raise ValueError('Final checkpoint changed')
    p=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if p['arm']!=arm or p['step']!=reg['updates'] or p['registration_sha256']!=T.sha(root/'registration.json'):raise ValueError('Wrong final checkpoint')
    runtime.configure_stable_runtime();T.write(out/'resources_before.json',T.resources())
    model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False).eval();model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False)
    head=None
    if arm=='SEAM':head=OccupancyHead(24).eval().requires_grad_(False);head.load_state_dict(p['head_state'],strict=True);head.to('cuda')
    layouts={};count=0;tick=time.perf_counter()
    try:
        for q in manifest['requests']:
            dest=out/q['request_id']/'raw'
            if (dest/'receipt.json').exists():
                r=T.read(dest/'receipt.json')
                if r['status']!='COMPLETE' or T.sha(dest/'raw.npz')!=r['raw_sha256']:raise ValueError('Saved RAW changed')
                continue
            if dest.exists():raise RuntimeError('Unresolved sampling attempt: no implicit replay')
            for key,hashkey in [('inference_npz','input_sha256'),('source_npz','source_sha256'),('noise_npz','noise_sha256')]:
                if T.sha(T.resolve(q[key]))!=q[hashkey]:raise ValueError('Frozen evaluation input changed')
            cid=q['condition_id'];bundle=T.arrays(q['inference_npz']);source=T.arrays(q['source_npz'])
            core={k:bundle[k] for k in ('C_vertices','C_faces_original','C_faces_model')};core['total_N']=int(bundle['N'])
            if cid not in layouts:
                lp=out/'layouts'/cid
                if (lp/'receipt.json').exists():
                    a=T.arrays(lp/'layout.npz');logits=a['occupancy_logits'];layout=inference_layout(**core,occupied=a['occupied'])
                else:
                    if lp.exists():raise RuntimeError('Unresolved layout inference')
                    lp.mkdir(parents=True);T.write(lp/'attempt.json',dict(condition_id=cid,source_labels_read=False))
                    edges,features=edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
                    if head:
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=head(torch.from_numpy(features).cuda(),torch.ones(len(edges),dtype=torch.bool,device='cuda')).float().cpu().numpy()
                        occupied=logits>=0
                    else:logits=np.zeros(len(edges),np.float32);occupied=np.zeros(len(edges),bool)
                    layout=inference_layout(**core,occupied=occupied);save_npz(lp/'layout.npz',{**layout,'occupancy_logits':logits})
                    save_npz(lp/'source_condition.npz',source);T.write(lp/'receipt.json',dict(status='COMPLETE',condition_id=cid,occupancy_forward=int(head is not None),active=int(occupied.sum()),edges=len(edges),threshold=0.,source_labels_read=False))
                layouts[cid]=(layout,logits)
            layout,logits=layouts[cid];noise=T.arrays(q['noise_npz'])['z'];dest.mkdir(parents=True)
            T.write(dest/'attempt.json',dict(arm=arm,request=q,checkpoint=receipt,forward_budget=50))
            raw,traj,audit=rollout(model,layout,noise,arm,dest/'calls.jsonl');a=raw_arrays(raw,layout,logits,source)
            save_npz(dest/'raw.npz',a);save_npz(dest/'trajectory.npz',dict(state=traj));save_npz(dest/'source_condition.npz',source)
            write_indexed_ply(dest/'raw.ply',a['vertices'],a['faces'])
            T.write(dest/'receipt.json',dict(status='COMPLETE',arm=arm,request_id=q['request_id'],condition_id=cid,checkpoint=receipt,
                raw_sha256=T.sha(dest/'raw.npz'),noise_sha256=q['noise_sha256'],source_labels_read=False,D0_forwards=0,**audit))
            count+=1;print(json.dumps(dict(stage='RAW',arm=arm,request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=p['generator_state_sha256']:raise RuntimeError('Frozen generator mutated')
        rows=[dict(q,arm=arm,label=arm,raw_path=str(out/q['request_id']/'raw/raw.npz')) for q in manifest['requests']]
        result=dict(status='COMPLETE',arm=arm,outputs=len(rows),new_rollouts=count,new_forwards=50*count,rows=rows,wall_seconds=time.perf_counter()-tick)
        T.write(out/'raw_complete.json',result);T.write(out/'resources_finished.json',T.resources());return result
    except BaseException as e:T.write(out/'failure.json',dict(status='BLOCKED',error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(T.DEFAULT_ROOT));ap.add_argument('--arm',choices=T.ARMS);ap.add_argument('--freeze-inputs',action='store_true');a=ap.parse_args()
    if a.freeze_inputs:freeze_inputs(a.root)
    elif a.arm:run(a.root,a.arm)
    else:ap.error('--arm or --freeze-inputs required')
if __name__=='__main__':main()
