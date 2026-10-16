"""Frozen occupancy intervention. New outputs only; no training or implicit retry.

Run with the historical WSL environment and PYTHONPATH=r2/src:repository.
prepare --include-source registers 32 Euler50 rollouts and 32 single D0 calls.
generate/post accept --arm and optional comma-separated --request-ids.
capacity is CPU-only and must run in a separate process after D0 proposals.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _n in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_n, '1')
import argparse
import copy
import json
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from r2.pilots.seam_slot_c200_v1 import generate as G
from r2.pilots.seam_slot_c200_v1 import postprocess as P

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'experiments/r2_seam_slot_c200_v1'
OUT = ROOT/'experiments/r2_seam_occupancy_rollout_v1'
ARMS = ('ALL_ON', 'SOURCE_ORACLE')
VERSION = 'r2_frozen_seam_occupancy_intervention_v1'


def put(path, value):
    path = Path(path)
    if path.exists():
        if G.read(path) != G.clean(value): raise ValueError('Refuse changed frozen output '+str(path))
    else: G.write(path, value)


def receipts(folder):
    return {str(p): G.sha(p) for p in sorted(Path(folder).iterdir()) if p.is_file() and p.name != 'receipt.json'}


def code_files():
    paths = [Path(__file__), Path(G.__file__), Path(P.__file__)]
    monitor = Path(__file__).with_name('monitor.py')
    if monitor.exists(): paths.append(monitor)
    paths += list((ROOT/'r2/prototypes/seam_slot_c200_v1').glob('*.py'))
    paths += [ROOT/'r2/pilots/seam_geometry_guard_v1/capacity_guard.py']
    paths += [ROOT/'r2/src/meshflow_control'/s for s in
              ('runtime.py','denoise_model.py','geo_c.py','geo_io.py','source_winding.py','checkpoints.py')]
    paths += list((ROOT/'r2/src/meshflow_control/models').rglob('*.py'))
    paths += list((ROOT/'r2/src/meshflow_control/denoiser_models').glob('*.py'))
    return {str(p): G.sha(p) for p in paths}


def prepare(include_source=False):
    if (OUT/'registration.json').exists():
        r, m = registered()
        if ('SOURCE_ORACLE' in m['arms']) != include_source: raise ValueError('Cannot change registered arms')
        return r
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('Incomplete prepare; inspect before recovery')
    OUT.mkdir(parents=True, exist_ok=True)
    oldreg = G.read(OLD/'registration.json'); m = copy.deepcopy(G.read(OLD/'eval_manifest.json'))
    arms = list(ARMS if include_source else ARMS[:1]); m['arms'] = arms
    m['schema'] = VERSION; m['new_rollouts'] = len(arms)*16; m['D0_calls'] = len(arms)*16
    m['occupancy'] = 'ALL_ON: saved logit + log(15.5) >=0; SOURCE_ORACLE: source labels, diagnostic only'
    m['scientific_scope'] = 'Four seen TRAIN conditions, sixteen fixed old requests per layout; no new noise or training'
    protected = {str(OLD/'eval_manifest.json'): G.sha(OLD/'eval_manifest.json'),
                 str(OLD/'registration.json'): G.sha(OLD/'registration.json')}
    for q in m['requests']:
        for key in ('inference_npz','metadata_npz','noise_npz'):
            protected[q[key]] = G.sha(q[key])
        protected[q['query_descriptor']['path']] = q['query_descriptor']['sha256']
        G.read_noise(q); G.source_from_bundle(q, G.read_bundle(q))
        oldbase = OLD/'evaluation/SEAM_SLOT_C200'/q['request_id']
        capbase = ROOT/'experiments/r2_seam_geometry_guard_v1/capacity/evaluation/SEAM_SLOT_C200'/q['request_id']
        for base, names in ((oldbase,('raw/raw.npz','raw/receipt.json','raw/source_condition.npz',
                                      'post/pre_geo.npz','post/post.npz','post/receipt.json')),
                            (capbase,('post/post.npz','post/receipt.json'))):
            for name in names:
                p=base/name
                protected[str(p)]=G.sha(p)
    for p in (OLD/'evaluation/per_output.csv',ROOT/'experiments/r2_seam_geometry_guard_v1/capacity/per_output.csv'):
        if p.exists():protected[str(p)]=G.sha(p)
    old_layouts = {}
    for dest in sorted((OLD/'evaluation/SEAM_SLOT_C200/layouts').iterdir()):
        rr = G.checked_receipt(dest/'receipt.json'); cid = rr['condition_id']
        old_layouts[cid] = (G.load_npz(dest/'layout.npz'), dest)
        protected[str(dest/'layout.npz')] = G.sha(dest/'layout.npz')
        protected[str(dest/'receipt.json')] = G.sha(dest/'receipt.json')
    from r2.prototypes.seam_slot_c200_v1.slot_data import inference_layout
    layouts = {}; totals = {a:0 for a in arms}; label_total = 0
    for ci, c in enumerate(sorted(m['conditions'], key=lambda x:x['condition_id'])):
        cid = c['condition_id']; q = next(x for x in m['requests'] if x['condition_id']==cid)
        bundle = G.read_bundle(q); source = G.source_from_bundle(q,bundle)
        old, old_path = old_layouts[cid]; logits = old['occupancy_logits']
        corrected = logits.astype(np.float64)+np.log(15.5)
        all_on = corrected >= 0
        if not all_on.all() or not np.array_equal(old['occupied'], logits>=0): raise ValueError('Cached analytic audit differs')
        labels = None
        if include_source:
            protected[c['labels_npz']] = G.sha(c['labels_npz'])
            with np.load(c['labels_npz'],allow_pickle=False) as z:
                labels = z['occupied_labels'].astype(bool)
                for key in ('C_vertices','C_faces_original','C_faces_model'):
                    if not G.same(z[key],bundle[key]):raise ValueError('Source occupancy C identity mismatch')
                if not G.same(z['edge_features'],old['features']):raise ValueError('Source occupancy edge order/features mismatch')
            if labels.shape != logits.shape: raise ValueError('Source label edge count mismatch')
            label_total += int(labels.sum())
        for arm in arms:
            occupied = all_on if arm=='ALL_ON' else labels
            layout = inference_layout(**bundle, occupied=occupied)
            if not G.same(layout['edge_records'], old['edge_records']): raise ValueError('C edge ordering changed')
            dest = OUT/'evaluation'/arm/'layouts'/f'C{ci:02d}'; dest.mkdir(parents=True)
            arrays = {k:np.asarray(v) for k,v in layout.items()}
            arrays.update(occupancy_logits=logits, corrected_logits=corrected)
            G.save_npz(dest/'layout.npz', arrays); G.save_npz(dest/'source_condition.npz',source)
            put(dest/'receipt.json',dict(status='COMPLETE', arm=arm, condition_id=cid,
                source_labels_read=arm=='SOURCE_ORACLE', diagnostic_only=arm=='SOURCE_ORACLE',
                source_layout_receipt=str(old_path/'receipt.json'), occupancy_forward_calls=0,
                decision='cached_logits + log(15.5) >= 0' if arm=='ALL_ON' else 'source occupied labels',
                files=receipts(dest)))
            layouts.setdefault(arm,{})[cid] = str(dest); totals[arm] += int(occupied.sum())
    if totals['ALL_ON'] != 66 or include_source and label_total != 62: raise ValueError('Expected panel totals changed')
    generator = OLD/'inference/SEAM_SLOT_C200_generator.pt'
    gr = G.read(generator.with_suffix('.json'))
    if G.sha(generator) != gr['sha256']: raise ValueError('Frozen generator changed')
    protected[str(generator)] = gr['sha256']; protected[str(generator.with_suffix('.json'))] = G.sha(generator.with_suffix('.json'))
    put(OUT/'eval_manifest.json', m)
    r = dict(schema=VERSION,status='FROZEN',arms=arms,requests=16,training_updates=0,
        new_rollouts=16*len(arms),generator_forwards=800*len(arms),D0_calls=16*len(arms),
        geometry_backends=['ORIGINAL_GEO','CAPACITY_GEO'], layouts=layouts, occupied_totals=totals,
        source_labels_usage='SOURCE_ORACLE diagnostic only; never supplied to ALL_ON',
        model_source_root=str(OLD), generator_path=str(generator), generator_sha256=gr['sha256'],
        d0_checkpoint=oldreg['d0_checkpoint'], original_logit_threshold=0., analytic_logit_shift=float(np.log(15.5)),
        noise_policy='same saved full Gaussian arrays and slot IDs; newly fixed coordinates overwritten by C anchors',
        precision='BF16 forward; FP32 parameters and Euler50 integration; deterministic math SDPA',
        all_RAW_before_D0=True, retries=0, resources=dict(generator_estimate_mib=2600,D0_estimate_mib=2600),
        protected=protected, code_hashes=code_files(), manifest_sha256=G.sha(OUT/'eval_manifest.json'))
    put(OUT/'registration.json',r)
    return r


def registered():
    r = G.read(OUT/'registration.json'); m = G.read(OUT/'eval_manifest.json')
    if r['status']!='FROZEN' or G.sha(OUT/'eval_manifest.json')!=r['manifest_sha256']: raise ValueError('Frozen manifest differs')
    G.verify_files(r['code_hashes']); G.verify_files(r['protected'])
    for arm in m['arms']:
        for dest in r['layouts'][arm].values(): G.checked_receipt(Path(dest)/'receipt.json')
    return r,m


def selected(m, arm, request_ids):
    if arm not in m['arms']: raise ValueError('Unregistered arm')
    ids = None if not request_ids else set(request_ids.split(','))
    if ids is not None and not ids <= {q['request_id'] for q in m['requests']}: raise ValueError('Unregistered request')
    return [q for q in m['requests'] if ids is None or q['request_id'] in ids]


class Monitor:
    def __init__(self, folder):
        self.folder=folder; self.stop=threading.Event(); self.samples=[]; self.thread=None
    def __enter__(self):
        def watch():
            while not self.stop.is_set():
                snap=G.resources(); self.samples.append(snap)
                G.event(self.folder/'resource_samples.jsonl','RESOURCE_SAMPLE',snapshot=snap,worker_pid=os.getpid())
                self.stop.wait(2)
        self.thread=threading.Thread(target=watch,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
        put(self.folder/'resource_monitor.json',dict(worker_pid=os.getpid(),sample_count=len(self.samples),
            scope='nvidia-smi whole-device and per-process observations; allocator peaks separate',samples=self.samples))


def aggregate(m,arm,stage,base=OUT):
    paths=[base/'evaluation'/arm/q['request_id']/stage/'receipt.json' for q in m['requests']]
    if not all(p.exists() for p in paths): return dict(status='PARTIAL',arm=arm,stage=stage,completed=sum(p.exists() for p in paths),registered=16)
    for p in paths:G.checked_receipt(p)
    out=dict(status='COMPLETE',arm=arm,outputs=16,files={str(p):G.sha(p) for p in paths})
    put(base/'evaluation'/arm/(stage+'_complete.json'),out);return out


def generate(arm,request_ids=None):
    r,m=registered(); qs=selected(m,arm,request_ids); pending=[]
    for q in qs:
        dest=OUT/'evaluation'/arm/q['request_id']/'raw'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():raise RuntimeError('Unresolved RAW attempt; no implicit retry: '+str(dest))
        else:pending.append(q)
    if not pending:return aggregate(m,arm,'raw')
    stage=OUT/'execution'/arm/('generate_'+str(time.time_ns()));stage.mkdir(parents=True)
    snap=G.resources();put(stage/'resources_before.json',snap);G.dispatch_guard(snap,r['resources']['generator_estimate_mib'])
    from meshflow_control import runtime
    runtime.configure_stable_runtime()
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    model=None;count=0;started=time.perf_counter()
    try:
        with Monitor(stage):
            model,payload,identity=G._load_generator(OLD,'SEAM_SLOT_C200');put(stage/'model_identity.json',identity)
            for q in pending:
                lp=Path(r['layouts'][arm][q['condition_id']]);layout=G.load_npz(lp/'layout.npz');source=G.load_npz(lp/'source_condition.npz')
                for name in ('N','K','free_coordinate_count'):layout[name]=int(layout[name])
                dest=OUT/'evaluation'/arm/q['request_id']/'raw';dest.mkdir(parents=True)
                put(dest/'attempt.json',dict(arm=arm,request=q,budget_generator_forwards=50,checkpoint=identity,unix=time.time()))
                z=G.read_noise(q);raw,trajectory,audit=G.rollout(model,layout,z,dest/'calls.jsonl')
                arrays=G.raw_arrays(raw,layout,layout['occupancy_logits'],source)
                arrays['corrected_logits']=layout['corrected_logits']
                G.save_npz(dest/'raw.npz',arrays);G.save_npz(dest/'trajectory.npz',dict(state=trajectory))
                G.save_npz(dest/'source_condition.npz',source);write_indexed_ply(dest/'raw.ply',arrays['vertices'],arrays['faces'])
                put(dest/'audit.json',audit)
                put(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=arm,request_id=q['request_id'],condition_id=q['condition_id'],
                    source_labels_read=arm=='SOURCE_ORACLE',diagnostic_only=arm=='SOURCE_ORACLE',generator_forwards=50,D0_forwards=0,
                    noise_sha256=G.sha(q['noise_npz']),layout_receipt=str(lp/'receipt.json'),layout_receipt_sha256=G.sha(lp/'receipt.json'),
                    checkpoint=identity,raw_path=str(dest/'raw.npz'),raw_sha256=G.sha(dest/'raw.npz'),audit=audit,files=receipts(dest)))
                count+=1;print(json.dumps(dict(stage='RAW',arm=arm,request_id=q['request_id'],new_completed=count,seconds=audit['seconds'])),flush=True)
            if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise RuntimeError('Generator state changed')
        put(stage/'completion.json',dict(status='COMPLETE',rollouts=count,generator_forwards=count*50,wall_seconds=time.perf_counter()-started))
        return aggregate(m,arm,'raw')
    except BaseException as e:
        put(stage/'failure.json',dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc(),completed_requests=count,no_implicit_retry=True));raise
    finally:
        if model is not None:del model;torch.cuda.empty_cache()


def raw_barrier(m):
    for arm in m['arms']:
        for q in m['requests']:G.checked_receipt(OUT/'evaluation'/arm/q['request_id']/'raw/receipt.json')


def post(arm,request_ids=None):
    r,m=registered();raw_barrier(m);pending=[]
    for q in selected(m,arm,request_ids):
        dest=OUT/'evaluation'/arm/q['request_id']/'post'
        if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json')
        elif dest.exists():
            if not (dest/'proposal_receipt.json').exists() or (dest/'geometry_attempt.json').exists():
                raise RuntimeError('Unresolved D0/GEO attempt; no implicit retry: '+str(dest))
            G.checked_receipt(dest/'proposal_receipt.json');pending.append(q)
        else:pending.append(q)
    if not pending:return aggregate(m,arm,'post')
    stage=OUT/'execution'/arm/('post_'+str(time.time_ns()));stage.mkdir(parents=True)
    snap=G.resources();put(stage/'resources_before.json',snap)
    need_model=any(not (OUT/'evaluation'/arm/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model=None;calls=count=0;started=time.perf_counter()
    try:
        with Monitor(stage):
            if need_model:
                G.dispatch_guard(snap,r['resources']['D0_estimate_mib'])
                from meshflow_control.denoise_model import load_denoiser,predict_proposal
                import torch
                model,identity=load_denoiser(r['d0_checkpoint']['path']);put(stage/'model_identity.json',identity)
            from meshflow_control.geo_io import write_indexed_ply
            for q in pending:
                rawdir=OUT/'evaluation'/arm/q['request_id']/'raw';raw=G.load_npz(rawdir/'raw.npz');source=G.load_npz(rawdir/'source_condition.npz')
                dest=OUT/'evaluation'/arm/q['request_id']/'post';dest.mkdir(parents=True,exist_ok=True)
                if not (dest/'attempt.json').exists():put(dest/'attempt.json',dict(arm=arm,request=q,D0_budget=1,RAW_barrier_outputs=len(m['arms'])*16))
                if (dest/'proposal_receipt.json').exists():
                    pr=G.checked_receipt(dest/'proposal_receipt.json')
                    if pr['raw_sha256']!=G.sha(rawdir/'raw.npz'):raise ValueError('Cached proposal mismatch')
                    proposal=G.load_npz(dest/'proposal.npz')['official_unconstrained_proposal']
                else:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    G.event(dest/'calls.jsonl','D0_FORWARD_STARTED',budget=1)
                    proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
                    G.event(dest/'calls.jsonl','D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick)
                    G.save_npz(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
                    put(dest/'proposal_receipt.json',dict(status='COMPLETE',raw_sha256=G.sha(rawdir/'raw.npz'),D0_forwards=1,
                        official_checkpoint_sha256=r['d0_checkpoint']['sha256'],seconds=time.perf_counter()-tick,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope='single D0 allocator with resident D0, not whole-process VRAM',files={str(dest/'proposal.npz'):G.sha(dest/'proposal.npz')}))
                put(dest/'geometry_attempt.json',dict(unix=time.time(),GEO_calls=1))
                pre,canonical,final,geometry,oriented,anchors=P.geometry_from_proposal(raw,proposal,source)
                G.save_npz(dest/'pre_geo.npz',dict(output=pre,C=pre[:int(raw['known_faces'])],known_faces=raw['known_faces'],
                    raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
                G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',final)
                write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
                put(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
                put(dest/'anchor_identity.json',anchors);put(dest/'source_restoration.json',oriented['restoration_summary'])
                orient=oriented['orientation_result'];put(dest/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
                put(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,arm=arm,request_id=q['request_id'],condition_id=q['condition_id'],
                    D0_forwards=1,GEO_calls=1,source_labels_read=arm=='SOURCE_ORACLE',anchors=anchors,
                    post_path=str(dest/'post.npz'),post_sha256=G.sha(dest/'post.npz'),files=receipts(dest)))
                count+=1;print(json.dumps(dict(stage='POST',arm=arm,request_id=q['request_id'],completed=count)),flush=True)
        put(stage/'completion.json',dict(status='COMPLETE',D0_forwards=calls,geometry=count,wall_seconds=time.perf_counter()-started))
        return aggregate(m,arm,'post')
    except BaseException as e:
        put(stage/'failure.json',dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc(),D0_returns=calls,completed_requests=count,no_implicit_retry=True));raise
    finally:
        if model is not None:del model;torch.cuda.empty_cache()


def capacity(arm=None,request_ids=None):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    r,m=registered();raw_barrier(m)
    from r2.pilots.seam_geometry_guard_v1.capacity_guard import finish
    from meshflow_control.geo_io import write_indexed_ply
    summaries=[]
    for a in ([arm] if arm else m['arms']):
        for q in selected(m,a,request_ids):
            old=OUT/'evaluation'/a/q['request_id'];G.checked_receipt(old/'post/receipt.json')
            dest=OUT/'capacity/evaluation'/a/q['request_id']/'post'
            if (dest/'receipt.json').exists():G.checked_receipt(dest/'receipt.json');continue
            if dest.exists():raise RuntimeError('Unresolved capacity attempt; no implicit retry')
            dest.mkdir(parents=True);put(dest/'attempt.json',dict(arm=a,request_id=q['request_id'],unix=time.time(),model_calls=0))
            raw=G.load_npz(old/'raw/raw.npz');pre=G.load_npz(old/'post/pre_geo.npz')['output'];source=G.load_npz(old/'raw/source_condition.npz')
            canonical,final,result,oriented,anchors,logs=finish(raw,pre,source,True)
            G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',final)
            write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
            put(dest/'geometry.json',dict(stats=result['stats'],face_records=result['face_records']))
            put(dest/'merge_log.json',logs);put(dest/'anchor_identity.json',anchors)
            put(dest/'source_restoration.json',oriented['restoration_summary'])
            orient=oriented['orientation_result'];put(dest/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
            put(dest/'receipt.json',dict(status='COMPLETE',arm=a,request_id=q['request_id'],condition_id=q['condition_id'],model_calls=0,
                pre_geo_sha256=G.sha(old/'post/pre_geo.npz'),post_sha256=G.sha(dest/'post.npz'),anchors=anchors,files=receipts(dest)))
            print(json.dumps(dict(stage='CAPACITY',arm=a,request_id=q['request_id'])),flush=True)
        summaries.append(aggregate(m,a,'post',OUT/'capacity'))
    return summaries


def status():
    if not (OUT/'registration.json').exists():return dict(status='NOT_PREPARED')
    r=G.read(OUT/'registration.json');m=G.read(OUT/'eval_manifest.json')
    return dict(registration=r['status'],budgets={k:r[k] for k in ('new_rollouts','generator_forwards','D0_calls')},
        counts={a:{s:sum((OUT/'evaluation'/a/q['request_id']/s/'receipt.json').exists() for q in m['requests']) for s in ('raw','post')} for a in m['arms']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','generate','post','capacity','status'],required=True)
    p.add_argument('--include-source',action='store_true');p.add_argument('--arm',choices=ARMS);p.add_argument('--request-ids')
    a=p.parse_args()
    if a.stage=='prepare':out=prepare(a.include_source)
    elif a.stage=='status':out=status()
    elif a.stage=='capacity':out=capacity(a.arm,a.request_ids)
    elif not a.arm:p.error('--arm required')
    else:out=globals()[a.stage](a.arm,a.request_ids)
    print(json.dumps(G.clean(out),indent=2))

if __name__=='__main__':main()
