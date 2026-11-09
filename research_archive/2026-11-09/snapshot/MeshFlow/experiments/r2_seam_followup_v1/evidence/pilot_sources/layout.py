"""Frozen B generator with exact cached LONG layouts: 28 RAW and 28 D0.

The only generation intervention is the occupancy layout. Old model, Gaussian,
query, scoring, GEO_C and winding implementations are imported read-only.
No head forward, training, source-label inference or automatic model retry.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from r2.pilots.seam_abc_v1 import generate as G
from r2.pilots.seam_abc_v1 import postprocess as P

ROOT=G.REPOSITORY
OLD=ROOT/'experiments/r2_seam_abc_v1'
OUT=ROOT/'experiments/r2_seam_followup_v1/layout'
LABEL='B_LONG_LAYOUT'
VERSION='seam_followup_frozen_B_LONG_layout_v1'


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'registration.json'
    if path.exists():
        value=G.read(path);G.verify_files(value['protected_files']);return value
    old=G.read(OLD/'eval_manifest.json');oldreg=G.read(OLD/'registration.json')
    required=('request_id','condition_id','parent_id','noise_index','N','K','seed',
              'split','inference_npz','noise_npz','metadata_npz','query_descriptor')
    requests=[{key:q[key] for key in required} for q in old['requests']]
    if len(requests)!=28 or len({q['request_id'] for q in requests})!=28:
        raise ValueError('Exactly 28 fixed requests required')
    protected={str(OLD/'eval_manifest.json'):G.sha(OLD/'eval_manifest.json')}
    layouts={}
    from r2.prototypes.seam_slot_c200_v1.slot_data import inference_layout
    for q in requests:
        bundle=G.read_bundle(q);source=G.source_from_bundle(q,bundle);G.read_noise(q)
        for name in ('inference_npz','noise_npz','metadata_npz'):
            protected[q[name]]=G.sha(q[name])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
        if q['condition_id'] in layouts:continue
        found=[]
        for folder in sorted((OLD/'evaluation/LONG/layouts').iterdir()):
            rec=G.read(folder/'receipt.json')
            if rec['condition_id']==q['condition_id']:found.append(folder)
        if len(found)!=1:raise ValueError('Exact cached LONG condition layout missing')
        lp=found[0];rec=G.checked_receipt(lp/'receipt.json')
        if rec['C_bundle_sha256']!=G.sha(q['inference_npz']):raise ValueError('C identity mismatch')
        arrays=G.load_npz(lp/'layout.npz');rebuilt=inference_layout(**bundle,occupied=arrays['occupied'])
        for key in rebuilt:
            if not G.same(np.asarray(rebuilt[key]),arrays[key]):raise ValueError('Layout differs: '+key)
        savedsource=G.load_npz(lp/'source_condition.npz')
        if set(source)!=set(savedsource) or any(not G.same(source[k],savedsource[k]) for k in source):
            raise ValueError('Source winding metadata changed')
        for f in ('receipt.json','layout.npz','source_condition.npz'):protected[str(lp/f)]=G.sha(lp/f)
        layouts[q['condition_id']]=dict(path=str(lp/'layout.npz'),receipt=str(lp/'receipt.json'),
            source_path=str(lp/'source_condition.npz'),active_edges=int(arrays['occupied'].sum()),
            total_edges=len(arrays['occupied']),head=rec['head'])
    generator=OLD/'inference/B_POOL47_generator.pt'
    gmeta=G.read(generator.with_suffix('.json'))
    if G.sha(generator)!=gmeta['sha256']:raise ValueError('B generator changed')
    protected[str(generator)]=gmeta['sha256'];protected[str(generator.with_suffix('.json'))]=G.sha(generator.with_suffix('.json'))
    d0=oldreg['d0_checkpoint'];protected[d0['path']]=d0['sha256']
    # Baseline outputs and scores are read-only comparison assets, never labels.
    for q in requests:
        for arm in ('LONG','B_POOL47'):
            for stage in ('raw','post'):
                dest=OLD/'evaluation'/arm/q['request_id']/stage
                r=G.checked_receipt(dest/'receipt.json')
                protected[str(dest/'receipt.json')]=G.sha(dest/'receipt.json')
                protected[str(dest/(stage+'.npz'))]=G.sha(dest/(stage+'.npz'))
    G.verify_files(protected)
    value=dict(schema=VERSION,status='FROZEN',label=LABEL,requests=requests,layouts=layouts,
        generator=dict(path=str(generator),sha256=gmeta['sha256']),d0_checkpoint=d0,
        budget=dict(training_updates=0,occupancy_forwards=0,pointer_forwards=0,
            RAW_outputs=28,generator_forwards=1400,D0_outputs=28,D0_forwards=28,GEO_calls=28),
        resources=dict(generator_estimate_mib=2600,D0_estimate_mib=3500),
        inference='Same B final live generator; exact LONG cached C-only layouts; same original Gaussian; Euler50',
        interpretation='Inference-layout intervention within frozen B; not removal of training occupancy/data effects',
        no_labels_read=True,protected_files=protected)
    G.write(path,value)
    G.event(OUT/'ledger.jsonl','REGISTERED',requests=28,new_model_calls=0)
    return value


def contract(require_go=False):
    reg=G.read(OUT/'registration.json')
    if reg['schema']!=VERSION or reg['status']!='FROZEN':raise ValueError('Unfrozen registration')
    G.verify_files(reg['protected_files'])
    if require_go:
        go=G.read(OUT/'dispatch_go.json')
        if go.get('registration_sha256')!=G.sha(OUT/'registration.json') or go.get('code_sha256')!=G.sha(__file__):
            raise ValueError('Parent execution freeze missing or changed')
        if go.get('authorized') is not True:raise ValueError('Parent dispatch not released')
        G.verify_files(go['dependency_hashes'])
    return reg


def load_layout(reg,q):
    item=reg['layouts'][q['condition_id']]
    a=G.load_npz(item['path']);logits=a.pop('occupancy_logits')
    # NPZ scalar arrays round-trip exactly, but the original runtime API expects
    # Python scalars for shape/time construction. No geometric array is changed.
    a={k:(v.item() if np.asarray(v).shape==() else v) for k,v in a.items()}
    return a,logits,G.load_npz(item['source_path']),item


def aggregate(stage,reg,attempt=None,wall=None):
    receipts=[OUT/'evaluation'/LABEL/q['request_id']/stage/'receipt.json' for q in reg['requests']]
    values=[G.checked_receipt(p) for p in receipts]
    if any(v['arm']!=LABEL for v in values):raise ValueError('Output identity differs')
    result=dict(status='COMPLETE',schema=VERSION,arm=LABEL,outputs=28,
        generator_forwards=sum(v.get('generator_forwards',0) for v in values),
        D0_forwards=sum(v.get('D0_forwards',0) for v in values),
        new_occupancy_forwards=0,new_training_updates=0,wall_seconds=wall,
        files={str(p):G.sha(p) for p in receipts})
    path=OUT/(stage+'_complete.json')
    if path.exists():return G.checked_receipt(path)
    G.write(path,result)
    G.event(OUT/'ledger.jsonl',stage.upper()+'_COMPLETE',**{k:v for k,v in result.items() if k!='files'})
    return result


def generate():
    reg=contract(require_go=True)
    if (OUT/'raw_complete.json').exists():return G.checked_receipt(OUT/'raw_complete.json')
    pending=[]
    for q in reg['requests']:
        dest=OUT/'evaluation'/LABEL/q['request_id']/'raw'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():raise RuntimeError('Unresolved RAW attempt; no automatic retry '+str(dest))
        else:pending.append(q)
    if not pending:return aggregate('raw',reg)
    snap=G.resources();attempt=OUT/('generation_attempt_'+str(time.time_ns()));attempt.mkdir()
    G.write(attempt/'resources_before.json',snap)
    G.dispatch_guard(snap,reg['resources']['generator_estimate_mib'])
    from meshflow_control import runtime
    runtime.configure_stable_runtime()
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    model,payload,identity=G._load_generator(OLD,'B_POOL47')
    G.write(attempt/'model_identity.json',identity);G.write(attempt/'resources_resident.json',G.resources())
    count=0;started=time.perf_counter()
    try:
        for q in pending:
            layout,logits,source,item=load_layout(reg,q)
            dest=OUT/'evaluation'/LABEL/q['request_id']/'raw';dest.mkdir(parents=True)
            G.write(dest/'attempt.json',dict(request=q,arm=LABEL,checkpoint=identity,unix=time.time(),budget_forwards=50))
            G.write(dest/'resources_before.json',G.resources())
            raw,trajectory,audit=G.rollout(model,layout,G.read_noise(q),dest/'calls.jsonl')
            arrays=G.raw_arrays(raw,layout,logits,source)
            G.save_npz(dest/'raw.npz',arrays);G.save_npz(dest/'trajectory.npz',dict(state=trajectory))
            G.save_npz(dest/'source_condition.npz',source)
            write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces'])
            G.write(dest/'audit.json',audit)
            files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
            G.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=LABEL,
                request_id=q['request_id'],condition_id=q['condition_id'],checkpoint=identity,
                input_bundle_sha256=G.sha(q['inference_npz']),noise_sha256=G.sha(q['noise_npz']),
                layout_path=item['path'],layout_sha256=G.sha(item['path']),layout_source='EXACT_CACHED_LONG',
                raw_path=str(dest/'raw.npz'),raw_sha256=G.sha(dest/'raw.npz'),
                generator_forwards=50,D0_forwards=0,occupancy_forwards=0,source_labels_read=False,
                audit=audit,files=files))
            count+=1
            G.event(OUT/'ledger.jsonl','RAW_SAVED',request_id=q['request_id'],generator_forwards=50)
            print(json.dumps(dict(stage='RAW',completed=count,request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise ValueError('Frozen B changed')
        G.write(attempt/'resources_finished.json',G.resources())
        return aggregate('raw',reg,attempt,time.perf_counter()-started)
    except BaseException as e:
        G.write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=count,no_retry=True));raise
    finally:
        del model;torch.cuda.empty_cache()


def postprocess():
    reg=contract(require_go=True)
    G.checked_receipt(OUT/'raw_complete.json') # All 28 RAW before any D0.
    if (OUT/'post_complete.json').exists():return G.checked_receipt(OUT/'post_complete.json')
    pending=[]
    for q in reg['requests']:
        dest=OUT/'evaluation'/LABEL/q['request_id']/'post'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():
            if not (dest/'proposal_receipt.json').exists() or (dest/'geometry_attempt.json').exists():
                raise RuntimeError('Unresolved D0/geometry attempt; no replay '+str(dest))
            G.checked_receipt(dest/'proposal_receipt.json');pending.append(q)
        else:pending.append(q)
    if not pending:return aggregate('post',reg)
    attempt=OUT/('post_attempt_'+str(time.time_ns()));attempt.mkdir()
    G.write(attempt/'all_RAW_barrier.json',dict(status='PASS',RAW=28,receipt_sha256=G.sha(OUT/'raw_complete.json')))
    checkpoint=reg['d0_checkpoint']
    if checkpoint['sha256']!='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36':raise ValueError('D0 changed')
    need=any(not (OUT/'evaluation'/LABEL/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model=None;calls=count=0;started=time.perf_counter()
    if need:
        snap=G.resources();G.write(attempt/'resources_before.json',snap)
        G.dispatch_guard(snap,reg['resources']['D0_estimate_mib'])
        from meshflow_control.denoise_model import load_denoiser,predict_proposal
        model,identity=load_denoiser(checkpoint['path'])
        G.write(attempt/'model_identity.json',identity);G.write(attempt/'resources_resident.json',G.resources())
    try:
        from meshflow_control.geo_io import write_indexed_ply
        for q in pending:
            rawdir=OUT/'evaluation'/LABEL/q['request_id']/'raw';G.checked_receipt(rawdir/'receipt.json')
            raw=G.load_npz(rawdir/'raw.npz');source=G.load_npz(rawdir/'source_condition.npz')
            dest=rawdir.parent/'post'
            if not dest.exists():
                dest.mkdir();G.write(dest/'attempt.json',dict(arm=LABEL,request=q,D0_budget=1,GEO_calls=1,unix=time.time()))
            if (dest/'proposal_receipt.json').exists():
                p=G.checked_receipt(dest/'proposal_receipt.json')
                if p['raw_sha256']!=G.sha(rawdir/'raw.npz'):raise ValueError('Proposal RAW mismatch')
                proposal=G.load_npz(dest/'proposal.npz')['official_unconstrained_proposal']
            else:
                import torch
                G.write(dest/'resources_before_D0.json',G.resources())
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                G.event(dest/'calls.jsonl','D0_FORWARD_STARTED',budget=1)
                proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
                G.event(dest/'calls.jsonl','D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick)
                G.save_npz(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
                G.write(dest/'proposal_receipt.json',dict(status='COMPLETE',raw_sha256=G.sha(rawdir/'raw.npz'),
                    D0_forwards=1,seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='Per D0 forward; resident model included; not process VRAM',
                    files={str(dest/'proposal.npz'):G.sha(dest/'proposal.npz')}))
            G.write(dest/'geometry_attempt.json',dict(unix=time.time(),geometry_calls=1,orientation_calls=1))
            pre,canonical,final,geometry,oriented,anchors=P.geometry_from_proposal(raw,proposal,source)
            G.save_npz(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:q['K']],known_faces=raw['known_faces'],
                raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
            G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',final)
            write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
            G.write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
            G.write(dest/'anchor_identity.json',anchors);G.write(dest/'source_restoration.json',oriented['restoration_summary'])
            orient=oriented['orientation_result']
            G.write(dest/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
            files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
            G.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=LABEL,request_id=q['request_id'],
                condition_id=q['condition_id'],raw_sha256=G.sha(rawdir/'raw.npz'),post_path=str(dest/'post.npz'),
                post_sha256=G.sha(dest/'post.npz'),D0_forwards=1,GEO_calls=1,orientation_solves=1,
                official_D0_sha256=checkpoint['sha256'],anchors=anchors,source_labels_used=False,files=files))
            count+=1;G.event(OUT/'ledger.jsonl','POST_SAVED',request_id=q['request_id'],D0_forwards=1)
            print(json.dumps(dict(stage='POST',completed=count,request_id=q['request_id'])),flush=True)
        G.write(attempt/'resources_finished.json',G.resources())
        return aggregate('post',reg,attempt,time.perf_counter()-started)
    except BaseException as e:
        G.write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=count,D0_calls=calls,no_retry=True));raise
    finally:
        if model is not None:
            import torch
            del model;torch.cuda.empty_cache()


def verify():
    reg=contract()
    states={}
    for stage in ('raw','post'):
        p=OUT/(stage+'_complete.json')
        states[stage]=G.checked_receipt(p)['status'] if p.exists() else 'NOT_COMPLETE'
    return dict(status=states,new_model_calls=0,registered_requests=len(reg['requests']))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('register','raw','post','verify'))
    a=p.parse_args();fn={'register':register,'raw':generate,'post':postprocess,'verify':verify}[a.stage]
    r=fn();print(json.dumps(G.clean({k:v for k,v in r.items() if k not in ('requests','protected_files','files')})),flush=True)


if __name__=='__main__':main()
