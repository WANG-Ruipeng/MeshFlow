"""Gate-controlled frozen LONG/B generation under one C-only owner layout.

56 registered RAW outputs precede every D0 call. No training/head forward occurs
here. Each GPU worker owns one model, one arm, one ledger and 28 fixed noises.
Old sampling, official D0, GEO_C and source-winding code remain read-only.
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
OUT=ROOT/'experiments/r2_owner_layout_v1'
ARMS={'LONG_OWNER':'LONG','B_POOL47_OWNER':'B_POOL47'}
VERSION='owner_layout_frozen_generator_v1'


def _layout_items(value):
    raw=value.get('layouts',value)
    return raw if isinstance(raw,dict) else {r['condition_id']:r for r in raw}


def prepare(layout_manifest,gate_receipt):
    """CPU only; consumes existing C-only head predictions, never predicts."""
    OUT.mkdir(parents=True,exist_ok=True)
    final=OUT/'generation_registration.json'
    if final.exists():
        result=G.read(final);G.verify_files(result['protected_files']);return result
    lm,gate=Path(layout_manifest).resolve(),Path(gate_receipt).resolve()
    layout_manifest_value=G.read(lm);decision=G.read(gate)
    if decision.get('status')!='PASS' or decision.get('generation_allowed') is not True or decision.get('candidate')!='CONTEXT':
        raise ValueError('Frozen CONTEXT gate must explicitly PASS and permit generation')
    if decision['prediction_manifest_sha256']!=G.sha(lm):
        raise ValueError('Gate prediction manifest identity differs')
    items=_layout_items(layout_manifest_value)
    old=G.read(OLD/'eval_manifest.json');oldreg=G.read(OLD/'registration.json')
    required=('request_id','condition_id','parent_id','noise_index','N','K','seed',
              'split','inference_npz','noise_npz','metadata_npz','query_descriptor')
    requests=[{key:q[key] for key in required} for q in old['requests']]
    if len(requests)!=28 or len({q['request_id'] for q in requests})!=28:
        raise ValueError('Exactly 28 existing request identities required')
    if not {q['condition_id'] for q in requests}.issubset(items):
        raise ValueError('All seven registered C-only layout predictions required')
    protected={str(lm):G.sha(lm),str(gate):G.sha(gate),str(OLD/'eval_manifest.json'):G.sha(OLD/'eval_manifest.json')}
    layouts={}
    from r2.prototypes.seam_slot_c200_v1.slot_data import inference_layout
    from meshflow_control.seam.slot_data import edge_features as head_edge_features
    for q in requests:
        bundle=G.read_bundle(q);source=G.source_from_bundle(q,bundle);G.read_noise(q)
        for name in ('inference_npz','noise_npz','metadata_npz'):protected[q[name]]=G.sha(q[name])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
        if q['condition_id'] in layouts:continue
        item=items[q['condition_id']];pred=Path(item['prediction_npz']).resolve();rp=Path(item['receipt']).resolve()
        receipt=G.read(rp)
        if receipt['condition_id']!=q['condition_id'] or receipt['C_bundle_sha256']!=G.sha(q['inference_npz']):
            raise ValueError('Prediction C identity differs')
        if receipt.get('source_labels_used_in_forward') is not False:
            raise ValueError('Head inference receipt must certify no source labels read')
        if receipt.get('head_sha256')!=decision['checkpoint_sha256'] or receipt.get('threshold')!=.5 or receipt.get('features')!='C_only26':
            raise ValueError('Head checkpoint, threshold or feature definition differs')
        a=G.load_npz(pred)
        occupied=a['occupied'];logits=a['occupancy_logits'] if 'occupancy_logits' in a else a['logits']
        if occupied.dtype!=np.bool_ or logits.dtype!=np.float32 or logits.shape!=occupied.shape or not np.isfinite(logits).all():
            raise ValueError('Owner predictions require bool occupied and finite FP32 logits in exact C boundary order')
        if not np.array_equal(occupied,logits>=0):raise ValueError('Frozen threshold=.5 violated')
        layout=inference_layout(**bundle,occupied=occupied)
        head_edges,head_features=head_edge_features(bundle['C_vertices'],bundle['C_faces_original'],int(bundle['total_N']))
        if not G.same(head_edges,layout['edge_records']) or not G.same(head_features,layout['features']):
            raise ValueError('C-only head and generator disagree on ordered boundary features')
        # If producer saved full layout geometry, ensure it is precisely the
        # original C-only representation rather than a different owner decoder.
        for key in layout:
            if key in a and not G.same(np.asarray(layout[key]),a[key]):raise ValueError('Producer layout differs: '+key)
        # Head producer computes its original ordered C-only26 features through
        # boundary_from_C. Reconstruct that identical order from the protected C.
        folder=OUT/'generation_inputs'/q['parent_id'][:8]
        if folder.exists():raise ValueError('Incomplete registration artifact; explicit recovery required')
        folder.mkdir(parents=True)
        G.save_npz(folder/'layout.npz',dict(**{k:np.asarray(v) for k,v in layout.items()},occupancy_logits=logits))
        G.save_npz(folder/'source_condition.npz',source)
        record=dict(status='COMPLETE',condition_id=q['condition_id'],C_bundle_sha256=G.sha(q['inference_npz']),
            prediction_npz=str(pred),prediction_sha256=G.sha(pred),prediction_receipt=str(rp),prediction_receipt_sha256=G.sha(rp),
            active_edges=int(occupied.sum()),total_edges=len(occupied),new_head_forwards=0,source_labels_read=False,
            files={str(p):G.sha(p) for p in (folder/'layout.npz',folder/'source_condition.npz')})
        G.write(folder/'receipt.json',record)
        for p in (pred,rp,folder/'layout.npz',folder/'source_condition.npz',folder/'receipt.json'):protected[str(p)]=G.sha(p)
        layouts[q['condition_id']]=dict(path=str(folder/'layout.npz'),source_path=str(folder/'source_condition.npz'),
            receipt=str(folder/'receipt.json'),active_edges=int(occupied.sum()),total_edges=len(occupied))
    models={}
    for arm,base in ARMS.items():
        mr=ROOT/'experiments/r2_seam_quality_four_v1' if base=='LONG' else OLD
        mp=mr/'inference'/(base+'_generator.pt');meta=G.read(mp.with_suffix('.json'))
        if G.sha(mp)!=meta['sha256']:raise ValueError('Generator identity differs')
        models[arm]=dict(source_arm=base,path=str(mp),sha256=meta['sha256'])
        protected[str(mp)]=meta['sha256'];protected[str(mp.with_suffix('.json'))]=G.sha(mp.with_suffix('.json'))
    d0=oldreg['d0_checkpoint'];protected[d0['path']]=d0['sha256']
    protected[decision['checkpoint']]=decision['checkpoint_sha256']
    G.verify_files(protected)
    result=dict(schema=VERSION,status='FROZEN',arms=ARMS,requests=requests,layouts=layouts,models=models,
        gate_receipt=str(gate),gate_receipt_sha256=G.sha(gate),layout_manifest=str(lm),d0_checkpoint=d0,
        budget=dict(training_updates=0,head_forwards=0,pointer_forwards=0,RAW_outputs=56,generator_forwards=2800,
            D0_outputs=56,D0_forwards=56,GEO_calls=56),
        resources=dict(generator_estimate_mib=2600,D0_estimate_mib=3500),protected_files=protected,
        contract='Two frozen final generators; same new C-only layout per C; unchanged Gaussian/N/C/Euler50; all56RAW before D0',
        geometry='Single official D0; restore C and seam anchors; original GEO_C/source winding; binding is separate CPU stage')
    G.write(final,result);return result


def contract(require_go=False):
    reg=G.read(OUT/'generation_registration.json')
    if reg['schema']!=VERSION or reg['status']!='FROZEN' or reg['arms']!=ARMS:raise ValueError('Registration differs')
    G.verify_files(reg['protected_files'])
    gate=G.read(reg['gate_receipt'])
    if gate.get('status')!='PASS' or gate.get('generation_allowed') is not True:raise ValueError('Gate not passed')
    if require_go:
        go=G.read(OUT/'generation_dispatch_go.json')
        if go.get('authorized') is not True or go.get('registration_sha256')!=G.sha(OUT/'generation_registration.json') or go.get('code_sha256')!=G.sha(__file__):
            raise ValueError('Parent execution release absent or changed')
        G.verify_files(go['dependency_hashes'])
    return reg


def load_layout(reg,q):
    item=reg['layouts'][q['condition_id']];a=G.load_npz(item['path']);logits=a.pop('occupancy_logits')
    a={k:(v.item() if np.asarray(v).shape==() else v) for k,v in a.items()}
    return a,logits,G.load_npz(item['source_path']),item


def aggregate(stage,reg,arm,wall=None):
    folder=OUT/'evaluation'/arm
    receipts=[folder/q['request_id']/stage/'receipt.json' for q in reg['requests']]
    values=[G.checked_receipt(p) for p in receipts]
    if any(v['arm']!=arm for v in values):raise ValueError('Output identity differs')
    result=dict(status='COMPLETE',schema=VERSION,arm=arm,outputs=28,
        generator_forwards=sum(v.get('generator_forwards',0) for v in values),
        D0_forwards=sum(v.get('D0_forwards',0) for v in values),new_head_forwards=0,training_updates=0,
        wall_seconds=wall,files={str(p):G.sha(p) for p in receipts})
    path=folder/(stage+'_complete.json')
    if path.exists():return G.checked_receipt(path)
    G.write(path,result);G.event(folder/'ledger.jsonl',stage.upper()+'_COMPLETE',**{k:v for k,v in result.items() if k!='files'})
    return result


def raw(arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    reg=contract(require_go=True);ar=OUT/'evaluation'/arm;ar.mkdir(parents=True,exist_ok=True)
    if (ar/'raw_complete.json').exists():return G.checked_receipt(ar/'raw_complete.json')
    pending=[]
    for q in reg['requests']:
        dest=ar/q['request_id']/'raw'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():raise RuntimeError('Unresolved RAW; no automatic retry '+str(dest))
        else:pending.append(q)
    if not pending:return aggregate('raw',reg,arm)
    snap=G.resources();attempt=ar/('generation_attempt_'+str(time.time_ns()));attempt.mkdir()
    G.write(attempt/'resources_before.json',snap);G.dispatch_guard(snap,reg['resources']['generator_estimate_mib'])
    from meshflow_control import runtime
    runtime.configure_stable_runtime()
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    model,payload,identity=G._load_generator(OLD,ARMS[arm])
    G.write(attempt/'model_identity.json',identity);G.write(attempt/'resources_resident.json',G.resources())
    count=0;started=time.perf_counter()
    try:
        for q in pending:
            layout,logits,source,item=load_layout(reg,q);dest=ar/q['request_id']/'raw';dest.mkdir(parents=True)
            G.write(dest/'attempt.json',dict(request=q,arm=arm,checkpoint=identity,unix=time.time(),budget_forwards=50))
            G.write(dest/'resources_before.json',G.resources())
            value,trajectory,audit=G.rollout(model,layout,G.read_noise(q),dest/'calls.jsonl')
            arrays=G.raw_arrays(value,layout,logits,source)
            G.save_npz(dest/'raw.npz',arrays);G.save_npz(dest/'trajectory.npz',dict(state=trajectory))
            G.save_npz(dest/'source_condition.npz',source);write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces'])
            G.write(dest/'audit.json',audit)
            files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
            G.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=arm,request_id=q['request_id'],
                condition_id=q['condition_id'],checkpoint=identity,input_bundle_sha256=G.sha(q['inference_npz']),
                noise_sha256=G.sha(q['noise_npz']),layout_path=item['path'],layout_sha256=G.sha(item['path']),
                raw_path=str(dest/'raw.npz'),raw_sha256=G.sha(dest/'raw.npz'),generator_forwards=50,D0_forwards=0,
                head_forwards=0,source_labels_read=False,audit=audit,files=files))
            count+=1;G.event(ar/'ledger.jsonl','RAW_SAVED',request_id=q['request_id'],generator_forwards=50)
            if count==1:G.write(attempt/'resources_first_actual_rollout.json',G.resources())
            print(json.dumps(dict(stage='RAW',arm=arm,completed=count,request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise ValueError('Frozen model changed')
        G.write(attempt/'resources_finished.json',G.resources())
        return aggregate('raw',reg,arm,time.perf_counter()-started)
    except BaseException as e:
        G.write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=count,no_automatic_retry=True));raise
    finally:
        del model;torch.cuda.empty_cache()


def all_raw(reg):
    files={}
    for arm in ARMS:
        r=G.checked_receipt(OUT/'evaluation'/arm/'raw_complete.json')
        if r['outputs']!=28 or r['generator_forwards']!=1400:raise ValueError('RAW budget incomplete')
        for q in reg['requests']:
            p=OUT/'evaluation'/arm/q['request_id']/'raw/receipt.json';v=G.checked_receipt(p)
            if v['arm']!=arm or v['request_id']!=q['request_id']:raise ValueError('RAW identity mismatch')
            files[str(p)]=G.sha(p)
    if len(files)!=56:raise ValueError('All56RAW barrier failed')
    return files


def post(arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    reg=contract(require_go=True);barrier=all_raw(reg);ar=OUT/'evaluation'/arm
    if (ar/'post_complete.json').exists():return G.checked_receipt(ar/'post_complete.json')
    pending=[]
    for q in reg['requests']:
        dest=ar/q['request_id']/'post'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():
            if not (dest/'proposal_receipt.json').exists() or (dest/'geometry_attempt.json').exists():
                raise RuntimeError('Unresolved D0/geometry; no automatic replay '+str(dest))
            G.checked_receipt(dest/'proposal_receipt.json');pending.append(q)
        else:pending.append(q)
    if not pending:return aggregate('post',reg,arm)
    attempt=ar/('post_attempt_'+str(time.time_ns()));attempt.mkdir()
    G.write(attempt/'all_RAW_barrier.json',dict(status='PASS',RAW=56,files=barrier))
    checkpoint=reg['d0_checkpoint']
    if checkpoint['sha256']!='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36':raise ValueError('D0 differs')
    need=any(not (ar/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model=None;calls=count=0;started=time.perf_counter()
    if need:
        snap=G.resources();G.write(attempt/'resources_before.json',snap);G.dispatch_guard(snap,reg['resources']['D0_estimate_mib'])
        from meshflow_control.denoise_model import load_denoiser,predict_proposal
        model,identity=load_denoiser(checkpoint['path'])
        G.write(attempt/'model_identity.json',identity);G.write(attempt/'resources_resident.json',G.resources())
    try:
        from meshflow_control.geo_io import write_indexed_ply
        for q in pending:
            rd=ar/q['request_id']/'raw';G.checked_receipt(rd/'receipt.json')
            raw_arrays=G.load_npz(rd/'raw.npz');source=G.load_npz(rd/'source_condition.npz');dest=rd.parent/'post'
            if not dest.exists():
                dest.mkdir();G.write(dest/'attempt.json',dict(arm=arm,request=q,D0_budget=1,GEO_calls=1,unix=time.time()))
            if (dest/'proposal_receipt.json').exists():
                receipt=G.checked_receipt(dest/'proposal_receipt.json')
                if receipt['raw_sha256']!=G.sha(rd/'raw.npz'):raise ValueError('Proposal input changed')
                proposal=G.load_npz(dest/'proposal.npz')['official_unconstrained_proposal']
            else:
                import torch
                G.write(dest/'resources_before_D0.json',G.resources())
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                G.event(dest/'calls.jsonl','D0_FORWARD_STARTED',budget=1)
                proposal=predict_proposal(model,raw_arrays['raw']);torch.cuda.synchronize();calls+=1
                G.event(dest/'calls.jsonl','D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick)
                G.save_npz(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
                G.write(dest/'proposal_receipt.json',dict(status='COMPLETE',raw_sha256=G.sha(rd/'raw.npz'),D0_forwards=1,
                    seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='Per D0 call; resident model included; not total process VRAM',
                    files={str(dest/'proposal.npz'):G.sha(dest/'proposal.npz')}))
            G.write(dest/'geometry_attempt.json',dict(unix=time.time(),geometry_calls=1,orientation_calls=1))
            pre,canonical,final,geometry,oriented,anchors=P.geometry_from_proposal(raw_arrays,proposal,source)
            G.save_npz(dest/'pre_geo.npz',dict(output=pre,C=raw_arrays['raw'][:q['K']],known_faces=raw_arrays['known_faces'],
                raw_anchor_ids=raw_arrays['raw_anchor_ids'],raw_seam_slot_mask=raw_arrays['raw_seam_slot_mask']))
            G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',final)
            write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
            G.write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
            G.write(dest/'anchor_identity.json',anchors);G.write(dest/'source_restoration.json',oriented['restoration_summary'])
            orient=oriented['orientation_result'];G.write(dest/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
            files={str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
            G.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=arm,request_id=q['request_id'],condition_id=q['condition_id'],
                raw_sha256=G.sha(rd/'raw.npz'),post_path=str(dest/'post.npz'),post_sha256=G.sha(dest/'post.npz'),D0_forwards=1,
                GEO_calls=1,orientation_solves=1,official_D0_sha256=checkpoint['sha256'],anchors=anchors,source_labels_used=False,files=files))
            count+=1;G.event(ar/'ledger.jsonl','POST_SAVED',request_id=q['request_id'],D0_forwards=1)
            if count==1:G.write(attempt/'resources_first_actual_D0.json',G.resources())
            print(json.dumps(dict(stage='POST',arm=arm,completed=count,request_id=q['request_id'])),flush=True)
        G.write(attempt/'resources_finished.json',G.resources())
        return aggregate('post',reg,arm,time.perf_counter()-started)
    except BaseException as e:
        G.write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=count,D0_calls=calls,no_automatic_retry=True));raise
    finally:
        if model is not None:
            import torch
            del model;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','raw','post'))
    p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--layouts');p.add_argument('--gate')
    a=p.parse_args()
    if a.stage=='prepare':
        if not a.layouts or not a.gate:p.error('prepare requires --layouts and --gate')
        result=prepare(a.layouts,a.gate)
    else:
        if not a.arm:p.error('raw/post require --arm')
        result=(raw if a.stage=='raw' else post)(a.arm)
    print(json.dumps(G.clean({k:v for k,v in result.items() if k not in ('requests','protected_files','files','layouts')})),flush=True)

if __name__=='__main__':main()
