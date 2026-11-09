"""Single official Bench D0 + unchanged GEO_C/source orientation.

An explicit cross-label RAW barrier is checked before loading D0. This module
does not choose models, thresholds or extra denoising iterations.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import time
import traceback
from . import panel as P
from .sample import requests
from r2.pilots.bench_control_transfer_v1 import postprocess as OLD

VERSION='bench_parts_single_official_D0_GEO_C_source_v1'


def verify_raw_barrier(root,labels,subset):
    root=P.resolve(root);qs=requests(root,subset);receipts=[]
    if not labels or len(set(labels))!=len(labels):raise ValueError('Explicit unique barrier labels required')
    for label in labels:
        if Path(label).name!=label:raise ValueError('Unsafe label')
        done=P.read(root/'evaluation'/label/('raw_complete_'+subset+'.json'))
        if done['status']!='COMPLETE' or done['outputs']!=len(qs):raise ValueError('RAW wave not complete: '+label)
        for q in qs:
            folder=root/'evaluation'/label/q['request_id']/'raw';path=folder/'receipt.json';r=P.read(path)
            if r.get('label',r.get('arm'))!=label or r['request_id']!=q['request_id'] or r['generator_forwards']!=50 or r['raw_sha256']!=P.sha(folder/'raw.npz'):
                raise ValueError('RAW barrier identity differs: '+str(path))
            for artifact,digest in r.get('files',{}).items():
                if P.sha(artifact)!=digest:raise ValueError('RAW barrier evidence changed: '+artifact)
            receipts.append(path)
    return qs,receipts


def run(root,label,barrier_labels,subset='full',checkpoint=OLD.D0,estimated_peak_mib=3500):
    root=P.resolve(root)
    if label not in barrier_labels:raise ValueError('Current label absent from RAW wave')
    qs,barrier=verify_raw_barrier(root,barrier_labels,subset);out=root/'evaluation'/label
    final=out/('post_complete_'+subset+'.json')
    if final.exists():
        old=P.read(final)
        for path,digest in old['files'].items():
            if P.sha(path)!=digest:raise ValueError('Completed POST changed')
            receipt=P.read(path)
            if receipt['post_sha256']!=P.sha(Path(path).parent/'post.npz'):
                raise ValueError('Completed POST content changed')
            for nested,expected in receipt['files'].items():
                if P.sha(nested)!=expected:raise ValueError('Completed POST evidence changed')
        return old
    attempt=out/('postprocess_attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    OLD.C.write(attempt/'all_RAW_barrier.json',dict(status='PASS',labels=barrier_labels,subset=subset,
        outputs=len(barrier),files={str(p):P.sha(p) for p in barrier}))
    pending=[]
    for q in qs:
        dest=out/q['request_id']/'post'
        if (dest/'receipt.json').exists():
            receipt=P.read(dest/'receipt.json')
            if receipt['post_sha256']!=P.sha(dest/'post.npz'):raise ValueError('Saved POST changed')
        elif dest.exists() and (not (dest/'proposal_receipt.json').exists() or (dest/'geometry_attempt.json').exists()):
            raise RuntimeError('Unresolved D0/geometry attempt requires audit: '+str(dest))
        else:pending.append(q)
    need_model=any(not (out/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model=None;calls=count=0;started=time.perf_counter()
    try:
        from meshflow_control.geo_io import write_indexed_ply
        from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
        from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard
        if need_model:
            from r2.pilots.bench_control_transfer_v1 import train as T
            resources=T.resources();OLD.C.write(attempt/'resources_before.json',resources);dispatch_guard(resources,estimated_peak_mib)
            OLD.C.write(attempt/'dispatch.json',dict(estimated_process_peak_mib=estimated_peak_mib,workers=1,
                headroom='max(2GiB,10%total) plus1.2x comparable measured peak',controller_owns_concurrency=True))
            model,identity=OLD.load_bench_denoiser(P.resolve(checkpoint));OLD.C.write(attempt/'model_identity.json',identity)
            OLD.C.write(attempt/'resources_loaded.json',T.resources())
        for q in pending:
            rawdir=out/q['request_id']/'raw';dest=out/q['request_id']/'post'
            raw=OLD.C.arrays(rawdir/'raw.npz');source=OLD.C.arrays(rawdir/'source_condition.npz')
            if not dest.exists():
                dest.mkdir();OLD.C.write(dest/'attempt.json',dict(label=label,request=q,raw_sha256=P.sha(rawdir/'raw.npz'),
                    D0_checkpoint_sha256=OLD.D0_SHA,D0_budget=1,GEO_budget=1,schema=VERSION))
            if (dest/'proposal_receipt.json').exists():
                r=P.read(dest/'proposal_receipt.json')
                if r['raw_sha256']!=P.sha(rawdir/'raw.npz') or r['proposal_sha256']!=P.sha(dest/'proposal.npz'):
                    raise ValueError('Saved proposal differs')
                proposal=OLD.C.arrays(dest/'proposal.npz')['official_unconstrained_proposal']
            else:
                import torch
                from meshflow_control.denoise_model import predict_proposal
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                OLD.event(dest/'calls.jsonl','D0_FORWARD_STARTED',budget=1)
                proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
                OLD.event(dest/'calls.jsonl','D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick)
                P.array_file(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
                OLD.C.write(dest/'proposal_receipt.json',dict(status='COMPLETE',D0_forwards=1,raw_sha256=P.sha(rawdir/'raw.npz'),
                    proposal_sha256=P.sha(dest/'proposal.npz'),official_checkpoint_sha256=OLD.D0_SHA,seconds=time.perf_counter()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='Single real D0 forward, resident model included; allocator not full process VRAM'))
            OLD.C.write(dest/'geometry_attempt.json',dict(utc=OLD.C.utc(),calls=1))
            pre,canonical,post,geometry,oriented,anchors=geometry_from_proposal(raw,proposal,source)
            P.array_file(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:int(raw['known_faces'])],known_faces=raw['known_faces'],
                raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
            P.array_file(dest/'post_canonical.npz',canonical);P.array_file(dest/'post.npz',post)
            write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
            OLD.C.write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
            OLD.C.write(dest/'anchor_identity.json',anchors);OLD.C.write(dest/'source_restoration.json',oriented['restoration_summary'])
            orientation=oriented['orientation_result'];OLD.C.write(dest/'orientation.json',dict(summary=orientation['summary'],components=orientation['component_rows'],edges=orientation['edge_rows']))
            files={str(p):P.sha(p) for p in dest.iterdir() if p.is_file()}
            OLD.C.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,label=label,arm=label,request_id=q['request_id'],
                condition_id=q['condition_id'],post_sha256=P.sha(dest/'post.npz'),raw_sha256=P.sha(rawdir/'raw.npz'),
                official_D0_sha256=OLD.D0_SHA,D0_forwards=1,GEO_calls=1,source_labels_read=False,anchors=anchors,files=files))
            count+=1
            if need_model and (count in (1,2) or count%8==0):OLD.C.write(attempt/('resources_output'+str(count)+'.json'),T.resources())
            print(json.dumps(dict(stage='POST',label=label,completed=count,total=len(pending),request_id=q['request_id'])),flush=True)
        receipts=[out/q['request_id']/'post/receipt.json' for q in qs]
        result=dict(status='COMPLETE',label=label,subset=subset,outputs=len(qs),new_D0_forwards=calls,new_GEO_calls=count,
            seconds=time.perf_counter()-started,barrier_labels=barrier_labels,files={str(p):P.sha(p) for p in receipts})
        P.immutable(final,result);return result
    except BaseException as exc:
        OLD.C.write(attempt/'failure.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),returned_D0_calls=calls,
            completed_outputs=count,no_automatic_retry=True));raise
    finally:
        OLD.C.write(attempt/'resources_finished.json',OLD.C.resources())
        if model is not None:
            import torch
            del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(P.OUT));p.add_argument('--label',required=True)
    p.add_argument('--barrier-labels',nargs='+',required=True);p.add_argument('--panel',choices=('full','process'),default='full')
    p.add_argument('--checkpoint',default=str(OLD.D0));p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args();print(json.dumps(run(a.root,a.label,a.barrier_labels,a.panel,a.checkpoint,a.estimated_peak_mib)))
