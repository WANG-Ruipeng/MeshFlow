"""Frozen Chair S2000 new-back inputs: exactly registered Euler50 and single D0.

Reuse the audited Chair numerical functions; no optimizer, training, source
completion or source-owner labels enter the generation path. Each worker owns
whole conditions and disjoint output paths. Completed scientific calls are
never silently repeated on resume.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
sys.path.insert(0, str(ROOT))
import numpy as np
from r2.pilots.seam_slot_c200_v1 import generate as G

OUT = ROOT / 'experiments/r2_chair_back_transfer_v1'
LABEL = 'S2000'
GEN = ROOT / 'experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt'
HEAD = ROOT / 'experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt'
D0 = ROOT / 'experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt'


def selected(shard, shards):
    manifest = G.read(OUT / 'eval_manifest.json')
    rows = manifest['requests']
    assert len(rows) == 16 and len({q['condition_id'] for q in rows}) == 8
    cids = sorted({q['condition_id'] for q in rows})
    owned = set(cids[shard::shards])
    return [q for q in rows if q['condition_id'] in owned]


def assets(q):
    # The manifest's exact files are immutable; metadata contains only original C.
    for field in ('inference_npz','noise_npz','metadata_npz','source_npz'):
        digest = q.get(field + '_sha256') or q.get(field.removesuffix('_npz') + '_sha256')
        if digest is None:
            raise ValueError('Missing input hash: '+field)
        if G.sha(q[field]) != digest:
            raise ValueError('Changed input: '+field)
    bundle = G.read_bundle(q)
    source = G.source_from_bundle(q, bundle)
    G.read_noise(q)
    return bundle, source


def checked(receipt):
    return G.checked_receipt(receipt)


def load_generator():
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    receipt = G.read(GEN.with_suffix('.json'))
    if G.sha(GEN) != receipt['sha256']:
        raise ValueError('Generator SHA mismatch')
    payload = torch.load(GEN, map_location='cpu', weights_only=True, mmap=True)
    if (payload['global_step'],payload['local_step'],payload['EMA']) != (15000,2000,False):
        raise ValueError('Wrong frozen S2000 endpoint')
    expected = payload['generator_state_sha256']
    if state_hash(payload['generator_state']) != expected:
        raise ValueError('Generator tensor hash mismatch')
    model = build_model(payload['model_config'], readout_mode='none', trainable=False)
    model.load_state_dict(payload['generator_state'], strict=True)
    model.to('cuda').set_trainable(False).eval()
    return model, expected, dict(path=str(GEN),sha256=receipt['sha256'],generator_state_sha256=expected,
        global_step=15000,local_step=2000,EMA=False)


def layout(head, q, head_identity):
    import torch
    from meshflow_control.seam.inference import predict_layout
    from meshflow_control.seam.slot_data import inference_layout
    bundle, source = assets(q)
    dest = OUT / 'evaluation' / LABEL / 'layouts' / q['condition_id']
    if (dest / 'receipt.json').exists():
        rec = checked(dest / 'receipt.json')
        if rec['head'] != head_identity or rec['C_bundle_sha256'] != G.sha(q['inference_npz']):
            raise ValueError('Saved layout identity differs')
        a = G.load_npz(dest / 'layout.npz')
        value = inference_layout(**bundle, occupied=a['occupied'])
        for key in ('anchor_ids','edge_records','seam_slots','bulk_slots'):
            if not G.same(a[key], value[key]): raise ValueError('Saved layout mismatch')
        return value, a['occupancy_logits'], source, dest
    if dest.exists(): raise RuntimeError('Unresolved head attempt; no automatic retry: '+str(dest))
    dest.mkdir(parents=True)
    G.write(dest/'attempt.json',dict(condition_id=q['condition_id'],head=head_identity,source_labels_read=False))
    G.event(dest/'calls.jsonl','HEAD_FORWARD_STARTED',budget=1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        value, logits = predict_layout(head,bundle)
    torch.cuda.synchronize()
    G.event(dest/'calls.jsonl','HEAD_FORWARD_RETURNED')
    G.save_npz(dest/'layout.npz',{**value,'occupancy_logits':logits})
    G.save_npz(dest/'source_condition.npz',source)
    files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
    G.write(dest/'receipt.json',dict(status='COMPLETE',condition_id=q['condition_id'],head=head_identity,
        C_bundle_sha256=G.sha(q['inference_npz']),occupancy_forward_calls=1,threshold_logit=0.,
        active=int(value['occupied'].sum()),boundary_edges=len(value['occupied']),source_labels_read=False,files=files))
    return value,logits,source,dest


def raw_run(rows,attempt):
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.seam.occupancy import OccupancyHead
    from meshflow_control.geo_io import write_indexed_ply
    pending=[]
    for q in rows:
        assets(q)
        dest=OUT/'evaluation'/LABEL/q['request_id']/'raw'
        if (dest/'receipt.json').exists(): checked(dest/'receipt.json')
        elif dest.exists(): raise RuntimeError('Unresolved RAW attempt: '+str(dest))
        else: pending.append(q)
    if not pending: return dict(outputs=len(rows),new_rollouts=0,new_generator_forwards=0)
    runtime.configure_stable_runtime()
    model,expected,identity=load_generator()
    hr=G.read(HEAD.with_suffix('.json'))
    if G.sha(HEAD)!=hr['sha256']:raise ValueError('LONG head SHA mismatch')
    hp=torch.load(HEAD,map_location='cpu',weights_only=True,mmap=True)
    if state_hash(hp['head_state'])!=hp['head_state_sha256'] or hp['threshold_logit']!=0.:
        raise ValueError('Frozen LONG head differs')
    head=OccupancyHead(**hp['head_config']).eval().requires_grad_(False)
    head.load_state_dict(hp['head_state'],strict=True);head.to('cuda')
    hid=dict(path=str(HEAD),sha256=hr['sha256'],head_state_sha256=hp['head_state_sha256'])
    G.write(attempt/'model_identity.json',dict(generator=identity,head=hid))
    G.write(attempt/'resources_loaded.json',G.resources())
    layouts={};count=0
    for q in pending:
        cid=q['condition_id']
        if cid not in layouts:layouts[cid]=layout(head,q,hid)
        lay,logits,source,lp=layouts[cid]
        dest=OUT/'evaluation'/LABEL/q['request_id']/'raw';dest.mkdir(parents=True)
        G.write(dest/'attempt.json',dict(request=q,checkpoint=identity,forward_budget=50))
        raw,traj,audit=G.rollout(model,lay,G.read_noise(q),dest/'calls.jsonl')
        arrays=G.raw_arrays(raw,lay,logits,source)
        G.save_npz(dest/'raw.npz',arrays);G.save_npz(dest/'trajectory.npz',dict(state=traj))
        G.save_npz(dest/'source_condition.npz',source)
        write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces'])
        G.write(dest/'audit.json',audit)
        files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
        G.write(dest/'receipt.json',dict(status='COMPLETE',arm=LABEL,label=LABEL,request_id=q['request_id'],
            condition_id=cid,checkpoint=identity,raw_path=str(dest/'raw.npz'),raw_sha256=G.sha(dest/'raw.npz'),
            noise_sha256=G.sha(q['noise_npz']),input_bundle_sha256=G.sha(q['inference_npz']),
            layout_receipt=str(lp/'receipt.json'),generator_forwards=50,D0_forwards=0,
            source_labels_read=False,training_updates=0,audit=audit,files=files))
        count+=1;G.write(attempt/f'resources_output{count}.json',G.resources())
        print(json.dumps(dict(stage='RAW',request_id=q['request_id'],completed=count,total=len(pending),seconds=audit['seconds'])),flush=True)
    if state_hash(model.state_dict())!=expected or state_hash(head.state_dict())!=hp['head_state_sha256']:
        raise RuntimeError('Frozen weights changed')
    return dict(outputs=len(rows),new_rollouts=count,new_generator_forwards=50*count)


def raw_barrier():
    rows=G.read(OUT/'eval_manifest.json')['requests']
    for q in rows:
        r=checked(OUT/'evaluation'/LABEL/q['request_id']/'raw/receipt.json')
        if r['generator_forwards']!=50 or r['request_id']!=q['request_id']:
            raise ValueError('RAW barrier mismatch')


def post_run(rows,attempt):
    import torch
    from meshflow_control.denoise_model import load_denoiser,predict_proposal,OFFICIAL_SHA
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
    raw_barrier();pending=[]
    for q in rows:
        dest=OUT/'evaluation'/LABEL/q['request_id']/'post'
        if (dest/'receipt.json').exists():checked(dest/'receipt.json')
        elif dest.exists() and (not (dest/'proposal_receipt.json').exists() or (dest/'geometry_attempt.json').exists()):
            raise RuntimeError('Unresolved POST attempt; no automatic replay: '+str(dest))
        else:pending.append(q)
    if not pending:return dict(outputs=len(rows),new_D0_forwards=0)
    need=any(not (OUT/'evaluation'/LABEL/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model=None;calls=0
    if need:
        model,identity=load_denoiser(D0)
        G.write(attempt/'model_identity.json',identity);G.write(attempt/'resources_loaded.json',G.resources())
    for i,q in enumerate(pending):
        rawdir=OUT/'evaluation'/LABEL/q['request_id']/'raw'
        raw=G.load_npz(rawdir/'raw.npz');source=G.load_npz(rawdir/'source_condition.npz')
        dest=OUT/'evaluation'/LABEL/q['request_id']/'post'
        if not dest.exists():
            dest.mkdir();G.write(dest/'attempt.json',dict(request=q,D0_checkpoint_sha256=OFFICIAL_SHA,D0_budget=1,GEO_budget=1))
        if (dest/'proposal_receipt.json').exists():
            r=checked(dest/'proposal_receipt.json')
            if r['raw_sha256']!=G.sha(rawdir/'raw.npz'):raise ValueError('D0 input changed')
            proposal=G.load_npz(dest/'proposal.npz')['official_unconstrained_proposal']
        else:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            G.event(dest/'calls.jsonl','D0_FORWARD_STARTED',budget=1)
            proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
            G.event(dest/'calls.jsonl','D0_FORWARD_RETURNED')
            G.save_npz(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
            G.write(dest/'proposal_receipt.json',dict(status='COMPLETE',D0_forwards=1,raw_sha256=G.sha(rawdir/'raw.npz'),
                official_checkpoint_sha256=OFFICIAL_SHA,seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                files={str(dest/'proposal.npz'):G.sha(dest/'proposal.npz')}))
        G.write(dest/'geometry_attempt.json',dict(calls=1,unix=time.time()))
        pre,canonical,post,geometry,oriented,anchors=geometry_from_proposal(raw,proposal,source)
        G.save_npz(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:int(raw['known_faces'])],known_faces=raw['known_faces'],
            raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',post)
        write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        G.write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
        G.write(dest/'anchor_identity.json',anchors);G.write(dest/'source_restoration.json',oriented['restoration_summary'])
        ori=oriented['orientation_result'];G.write(dest/'orientation.json',dict(summary=ori['summary'],components=ori['component_rows'],edges=ori['edge_rows']))
        files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
        G.write(dest/'receipt.json',dict(status='COMPLETE',arm=LABEL,label=LABEL,request_id=q['request_id'],condition_id=q['condition_id'],
            raw_path=str(rawdir/'raw.npz'),raw_sha256=G.sha(rawdir/'raw.npz'),post_path=str(dest/'post.npz'),post_sha256=G.sha(dest/'post.npz'),
            pre_geo_path=str(dest/'pre_geo.npz'),D0_forwards=1,GEO_calls=1,official_D0_sha256=OFFICIAL_SHA,
            source_labels_read=False,anchors=anchors,files=files))
        G.write(attempt/f'resources_output{i+1}.json',G.resources())
        print(json.dumps(dict(stage='POST',request_id=q['request_id'],completed=i+1,total=len(pending))),flush=True)
    return dict(outputs=len(rows),new_D0_forwards=calls,new_GEO_calls=len(pending))


def run(stage,shard=0,shards=1):
    rows=selected(shard,shards)
    reg=G.read(OUT/'registration.json')
    if reg['status']!='FROZEN' or G.sha(OUT/'eval_manifest.json')!=reg['eval_manifest_sha256']:
        raise ValueError('Registration not frozen or input manifest changed')
    for p,digest in reg['scientific_code_hashes'].items():
        if G.sha(p)!=digest:raise ValueError('Frozen scientific code changed: '+p)
    for p,digest in reg['model_hashes'].items():
        # Full official D0 hash is checked by its strict loader before use.
        if Path(p)!=D0 and G.sha(p)!=digest:raise ValueError('Protected model changed')
    dest=OUT/'dispatch';dest.mkdir(exist_ok=True)
    attempt=dest/f'{stage}_shard{shard}_attempt{time.time_ns()}';attempt.mkdir()
    snap=G.resources();G.write(attempt/'resources_before.json',snap)
    estimate=2600 if stage=='raw' else 3500
    G.dispatch_guard(snap,estimate)
    G.write(attempt/'dispatch.json',dict(stage=stage,shard=shard,shards=shards,requests=[q['request_id'] for q in rows],
        process_peak_estimate_mib=estimate,no_automatic_model_replay=True))
    start=time.perf_counter()
    try:
        result=raw_run(rows,attempt) if stage=='raw' else post_run(rows,attempt)
        G.write(attempt/'complete.json',dict(status='COMPLETE',wall_seconds=time.perf_counter()-start,**result))
    except BaseException as error:
        G.write(attempt/'failure.json',dict(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise
    finally:G.write(attempt/'resources_finished.json',G.resources())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['raw','post'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();run(a.stage,a.shard,a.shards)
