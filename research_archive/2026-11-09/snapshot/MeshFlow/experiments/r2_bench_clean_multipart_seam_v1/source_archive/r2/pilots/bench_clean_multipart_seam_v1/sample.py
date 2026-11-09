"""Final1500 generator, two frozen C-only layouts, exactly96 Euler50 outputs."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
from .eval_prepare import P,OUT,OLD,MODEL,LABELS,requests,verify

GEN=OUT/'inference'/f'{MODEL}_generator.pt'
HEAD=OUT/'inference'/f'{MODEL}_occupancy.pt'

def freeze_inference():
    paths=[GEN,GEN.with_suffix('.json'),HEAD,HEAD.with_suffix('.json'),OUT/'eval_manifest.json',OUT/'registration.json']
    sourcepaths=[Path(__file__),Path(__file__).with_name('head.py'),
        Path(__file__).with_name('post_d0.py'),
        P.ROOT/'r2/pilots/bench_control_transfer_v1/sample.py',
        P.ROOT/'r2/pilots/bench_parts_convergence_v1/data.py',
        P.ROOT/'r2/pilots/bench_parts_convergence_v1/postprocess.py',
        P.ROOT/'r2/pilots/seam_slot_c200_v1/generate.py',
        P.ROOT/'r2/pilots/seam_slot_c200_v1/postprocess.py']
    value=dict(status='FROZEN',version='bench_clean_two_layout96_inference_v1',
        labels=list(LABELS),model=MODEL,outputs_per_label=48,global_step=13500,local_step=1500,
        generator_forwards=4800,D0_forwards=96,head_layout_calls=20,
        barrier='ALL96RAW_before_any_D0',source_labels_read=False,solver='Euler50',
        protected={str(p):P.sha(p) for p in paths},source_hashes={str(p):P.sha(p) for p in sourcepaths})
    P.immutable(OUT/'inference_registration.json',value);return value

def arrays(path):
    with np.load(P.resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def source_assets(q):
    for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
        assert P.sha(q[key])==q[key+'_sha256']
    a=arrays(q['inference_npz'])
    bundle={k:a[k] for k in ('C_vertices','C_faces_original','C_faces_model')}
    bundle['total_N']=int(a['N'])
    return bundle,arrays(q['source_npz'])

def layout(q,label,head,hid):
    from r2.pilots.bench_parts_convergence_v1.data import inference_layout
    from .head import predict_layout
    import torch
    bundle,source=source_assets(q);dest=OUT/'evaluation'/label/'layouts'/q['condition_id']
    if (dest/'receipt.json').exists():
        r=P.read(dest/'receipt.json');verify(r['files']);assert r['head']==hid
        a=arrays(dest/'layout.npz');lay=inference_layout(**bundle,occupied=a['occupied'])
        for key in ('anchor_ids','edge_records','seam_slots','bulk_slots'):assert np.array_equal(a[key],lay[key])
        return lay,a['occupancy_logits'],source,dest,0
    if dest.exists():raise RuntimeError('Unresolved layout attempt; do not retry silently '+str(dest))
    dest.mkdir(parents=True)
    P.immutable(dest/'attempt.json',dict(condition_id=q['condition_id'],label=label,head=hid,source_labels_read=False))
    calls=0
    if label=='OLD_S15000_LAYOUT':
        assert P.sha(q['old_layout_npz'])==q['old_layout_sha256']
        old=arrays(q['old_layout_npz']);logits=old['occupancy_logits']
        lay=inference_layout(**bundle,occupied=old['occupied'])
        for key in ('anchor_ids','edge_records','seam_slots','bulk_slots'):assert np.array_equal(old[key],lay[key])
    else:
        if not q['closed_C']:
            from r2.pilots.bench_control_transfer_v1.train import event
            event(dest/'calls.jsonl','HEAD_FORWARD_STARTED',budget=1)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):lay,logits=predict_layout(head,bundle)
            torch.cuda.synchronize();event(dest/'calls.jsonl','HEAD_FORWARD_RETURNED')
            calls=1
        else:lay=inference_layout(**bundle,occupied=np.zeros(0,bool));logits=np.zeros(0,np.float32)
    assert len(lay['occupied'])==int(q['boundary_edges'])
    P.array_file(dest/'layout.npz',{**lay,'occupancy_logits':logits})
    P.array_file(dest/'source_condition.npz',source)
    P.immutable(dest/'receipt.json',dict(status='COMPLETE',condition_id=q['condition_id'],label=label,head=hid,
        C_bundle_sha256=P.sha(q['inference_npz']),occupancy_forward_calls=calls,threshold_logit=0.,
        active=int(lay['occupied'].sum()),boundary_edges=len(lay['occupied']),source_labels_read=False,
        old_layout_sha256=q['old_layout_sha256'] if label=='OLD_S15000_LAYOUT' else None,
        files={str(p):P.sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay,logits,source,dest,calls

def run(label,estimated_peak_mib=2800):
    if label not in LABELS:raise ValueError('Only two registered layout arms')
    reg=freeze_inference()
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import train as T,sample as S
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays,dispatch_guard
    from .head import load_head
    qq=requests();out=OUT/'evaluation'/label;done=out/'raw_complete_full.json'
    if done.exists():
        result=P.read(done);verify(result['files'])
        for q in qq:verify(P.read(out/q['request_id']/'raw/receipt.json')['files'])
        return result
    pending=[]
    for q in qq:
        source_assets(q);dest=out/q['request_id']/'raw'
        if (dest/'receipt.json').exists():
            prior=P.read(dest/'receipt.json');verify(prior['files'])
            assert prior['generator']['sha256']==P.sha(GEN) and prior['noise_sha256']==q['noise_sha256']
        elif dest.exists():raise RuntimeError('Unresolved raw attempt; no automatic neural repeat '+str(dest))
        else:pending.append(q)
    if not pending:raise RuntimeError('All raws exist but missing completion; explicit ledger reconciliation required')
    attempt=out/('sampling_attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    snap=T.resources();T.write(attempt/'resources_before.json',snap);dispatch_guard(snap,estimated_peak_mib)
    T.write(attempt/'dispatch.json',dict(workers=1,estimated_process_peak_mib=estimated_peak_mib,headroom='max(2GiB,10%total)+1.2xpeak'))
    model=head=None;count=headcalls=0;tick=time.perf_counter()
    try:
        gr=P.read(GEN.with_suffix('.json'));assert P.sha(GEN)==gr['sha256']
        p=torch.load(GEN,map_location='cpu',weights_only=True,mmap=True)
        assert p['global_step']==13500 and p['local_step']==1500 and not p['EMA']
        assert state_hash(p['generator_state'])==p['generator_state_sha256']
        runtime.configure_stable_runtime()
        model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
        model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
        hid=dict(kind='FROZEN_OLD_S15000_LAYOUT',model_calls=0)
        hp=None
        if label=='NEW_HEAD':
            hr=P.read(HEAD.with_suffix('.json'));assert P.sha(HEAD)==hr['sha256']
            head,hp=load_head(HEAD);head.to('cuda').eval().requires_grad_(False)
            hid=dict(path=str(HEAD),sha256=P.sha(HEAD),head_state_sha256=state_hash(head.state_dict()))
        identity=dict(path=str(GEN),sha256=P.sha(GEN),state_sha256=p['generator_state_sha256'],local_step=1500,global_step=13500,EMA=False)
        T.write(attempt/'model_identity.json',dict(generator=identity,layout=hid))
        T.write(attempt/'resources_loaded.json',T.resources());layouts={}
        for q in pending:
            cid=q['condition_id']
            if cid not in layouts:
                lay,logits,source,lp,calls=layout(q,label,head,hid);headcalls+=calls
                layouts[cid]=(lay,logits,source,lp)
            lay,logits,source,lp=layouts[cid];dest=out/q['request_id']/'raw';dest.mkdir(parents=True)
            T.write(dest/'attempt.json',dict(request=q,generator=identity,forward_budget=50,label=label))
            raw,traj,audit=S.rollout(model,lay,arrays(q['noise_npz'])['z'],'SEAM',dest/'calls.jsonl')
            a=raw_arrays(raw,lay,logits,source)
            P.array_file(dest/'raw.npz',a);P.array_file(dest/'trajectory.npz',dict(state=traj));P.array_file(dest/'source_condition.npz',source)
            write_indexed_ply(dest/'raw.ply',a['vertices'],a['faces'])
            T.write(dest/'audit.json',audit)
            T.write(dest/'receipt.json',dict(status='COMPLETE',label=label,arm=label,request_id=q['request_id'],condition_id=cid,
                generator=identity,checkpoint_sha256=identity['sha256'],generator_forwards=50,D0_forwards=0,
                raw_sha256=P.sha(dest/'raw.npz'),noise_sha256=q['noise_sha256'],source_labels_read=False,
                layout_receipt=str(lp/'receipt.json'),audit=audit,files={str(f):P.sha(f) for f in dest.iterdir() if f.is_file()}))
            count+=1
            if count<=2 or count%8==0:T.write(attempt/f'resources_output{count}.json',T.resources())
            print(json.dumps(dict(stage='RAW',label=label,completed=count,total=len(pending),request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
        assert state_hash(model.state_dict())==p['generator_state_sha256']
        if head is not None:assert state_hash(head.state_dict())==hid['head_state_sha256']
        verify(reg['protected']);verify(reg['source_hashes'])
        result=dict(status='COMPLETE',label=label,mode='SEAM',subset='full',outputs=48,new_rollouts=count,new_forwards=count*50,
            new_head_forwards=headcalls,checkpoint_sha256=P.sha(GEN),seconds=time.perf_counter()-tick,
            files={str(out/q['request_id']/'raw/receipt.json'):P.sha(out/q['request_id']/'raw/receipt.json') for q in qq})
        P.immutable(done,result);return result
    except BaseException as e:
        T.write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_rollouts=count,head_calls=headcalls,no_automatic_retry=True));raise
    finally:
        T.write(attempt/'resources_finished.json',T.resources())
        del model,head;torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',choices=LABELS,required=True)
    parser.add_argument('--estimated-peak-mib',type=float,default=2800);a=parser.parse_args();print(json.dumps(run(a.label,a.estimated_peak_mib)),flush=True)
