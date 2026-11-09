"""Independent fixed-12 comparison; no models run on import or preparation.

Original 177-field scorer functions use private globals. No original experiment
module globals or original assets are modified. PATCH/LOOP/RL have explicit
sampling paths: an unavailable mechanism raises instead of using plain sampling.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import types
from . import common as C

ROOT, OUT, OLD = C.ROOT, C.OUT, C.OLD
MANIFEST = OUT/'data/eval_manifest.json'
LABELS, NEW_LABELS = C.LABELS, C.NEW_LABELS
STAGES = ('RAW','BASE_POST','CLEAN_FINAL_POST')
VERSION = 'bench_three_mechanisms_fixed12_evaluation_v1'
SCORE_REGISTRATION = OUT/'evaluation/score_registration_v2.json'


def dispatch_guard(snapshot,estimate_mib):
    """Adapt the maintained flat resource snapshot to the unchanged old guard."""
    from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard as original_guard
    if 'gpu' in snapshot:return original_guard(snapshot,estimate_mib)
    import math
    keys=('total_mib','used_mib','free_mib','gpu_util_percent')
    if any(k not in snapshot or not math.isfinite(float(snapshot[k])) for k in keys):
        raise ValueError('Incomplete finite nvidia-smi resource snapshot')
    row=','.join([str(snapshot['gpu_name']),*[str(snapshot[k]) for k in keys]])
    return original_guard({'gpu':{'returncode':0,'stdout':row}},estimate_mib)


def budget():
    return dict(requests=12,parents=6,new_RAW=72,new_D0=72,
        logical_Euler_vector_fields=3600,generator_forwards=3600,
        RL16_additional_policy_forwards_max=120,LOOP_extra_block_calls=2400,
        new_score_rows=216,reused_score_rows=36,total_score_rows=252)


def root_for(label,q=None):
    if label not in LABELS: raise ValueError(label)
    return OLD if label=='EXPOSURE' else OUT


def requests(label=None):
    if label is not None and label not in LABELS: raise ValueError(label)
    qs=C.read(MANIFEST)['requests']
    if len(qs)!=12 or len({q['request_id'] for q in qs})!=12 or len({q['parent_id'] for q in qs})!=6:
        raise ValueError('Frozen six-parent/twelve-request panel changed')
    if {q['parent_id'][:6] for q in qs}!={p[:6] for p in C.PANEL_PREFIXES}:
        raise ValueError('Unregistered evaluation parent')
    return qs


def new_requests(label):
    return requests(label) if label in NEW_LABELS else []


def label_spec(label):
    table={'PLAIN200':('PLAIN',200),'PLAIN267':('PLAIN',267),
           'PATCH200':('PATCH',200),'LOOP200':('LOOP',200)}
    if label in table:
        arm,step=table[label]
        return dict(mode=arm,arm=arm,local_step=step,global_step=C.BASE_STEP+step,subset='full',
            schema='bench_three_mechanisms_generator_v1',
            generator_path=str(OUT/'training'/arm/f'generator_step{step}.pt'))
    if label in ('RL_ZERO','RL16'):
        path=C.SOURCE_INFERENCE if label=='RL_ZERO' else OUT/'rl/training/opportunity_015/checkpoint.pt'
        return dict(mode=label,arm=label,local_step=0 if label=='RL_ZERO' else 16,
            global_step=None,subset='full',schema='REFERENCE' if label=='RL_ZERO' else 'bench_local_quality_rl_v1',
            generator_path=str(path))
    raise ValueError('No baseline sampling authorized')


def training_barrier():
    """All frozen training endpoints must exist before any new evaluation rollout."""
    files={}
    for label in ('PLAIN200','PLAIN267','PATCH200','LOOP200','RL16'):
        p=C.resolve(label_spec(label)['generator_path'])
        if not p.exists():raise FileNotFoundError('Evaluation waits for all training endpoints: '+str(p))
        sidecar=p.with_suffix('.json')
        if sidecar.exists():
            receipt=C.read(sidecar)
            if receipt.get('sha256')!=C.sha(p):raise ValueError('Endpoint sidecar differs')
            files[str(sidecar)]=C.sha(sidecar)
        files[str(p)]=C.sha(p)
    return files


def _code_hashes(code,kind):
    inherited=C.read(OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json'))['code']
    C.verify(inherited)
    return {**inherited,**{str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')},**code}


def all_raw_barrier():
    panel=C.read(OUT/'audits/reference_cache_audit.json');C.verify(panel['files']);C.verify(panel['protected'])
    paths=[]
    for label in NEW_LABELS:
        engine().verify_raw(label)
        paths.extend(engine().raw_receipt(q,label) for q in requests())
    if len(paths)!=72:raise ValueError('All 72 new RAWs must precede any D0')
    return paths


def _cpu_worker(job):
    label,q,mode,regsha=job
    e=engine();result=e.raw_one(label,q,regsha) if mode=='raw' else e.post_one(label,q,regsha)
    if 'torch' in sys.modules:raise RuntimeError('CPU scoring imported torch')
    print('EVALUATED',mode,label,q['request_id'],flush=True)
    return result


def cpu_dispatch_identity(label):
    e=engine()
    from r2.pilots.bench_four_routes_v1.exposure import evaluation as original
    if label not in NEW_LABELS:raise ValueError(label)
    return dict(label=label,output_root=str(e.root_for(label)),original_root=str(original.OUT),
        worker_module=e.cpu_one.__module__,new_root_preserved=e.OUT==OUT,
        original_root_unchanged=original.OUT==OLD,model_calls=0,scoring_calls=0)


@lru_cache(maxsize=1)
def engine():
    from r2.pilots.bench_four_routes_v1.exposure import evaluation as original
    before=(original.OUT,original.MANIFEST,original.LABELS)
    ns=dict(vars(original))
    for name,obj in vars(original).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(original):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            fn.__kwdefaults__=obj.__kwdefaults__;ns[name]=fn
    ns.update(OUT=OUT,OLD_M=OLD,MANIFEST=MANIFEST,LABELS=LABELS,NEW_LABELS=NEW_LABELS,
        ARMS={label:label for label in NEW_LABELS},VERSION=VERSION,__file__=__file__,
        requests=requests,new_requests=new_requests,root_for=root_for,budget=budget,
        prepare_panel=prepare_panel,prepare_manifest=prepare_panel,prepare_scores=prepare_scores,
        registration_path=lambda:C.REGISTRATION,label_spec=label_spec,
        reuse_spec=lambda q,label:dict(root=str(OLD),label=label) if label=='EXPOSURE' else None,
        inference_code_hashes=_code_hashes,all_raw_barrier=all_raw_barrier,cpu_one=_cpu_worker,
        sample=sample,collect=collect,run_cpu=run_cpu)
    if (original.OUT,original.MANIFEST,original.LABELS)!=before:raise AssertionError('Original globals changed')
    return types.SimpleNamespace(**ns)


def prepare_panel():
    done=OUT/'audits/reference_cache_audit.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);C.verify(r['protected']);return r
    manifest=C.read(MANIFEST);qs=requests();fields=manifest['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Original177 metrics changed')
    C.verify(manifest['protected'])
    src=OLD/'data/eval_manifest.json';source=C.read(src)
    original_q={q['request_id']:q for q in source['requests']}
    if any(q!=original_q[q['request_id']] for q in qs):raise ValueError('Subset input fields differ from original')
    protected={**manifest['protected'],str(src):C.sha(src),str(OLD/'results/completion.json'):C.sha(OLD/'results/completion.json')}
    original_rows={(r['stage'],r['request_id']):r for r in C.read(OLD/'results/metric_rows.json') if r['label']=='EXPOSURE'}
    protected[str(OLD/'results/metric_rows.json')]=C.sha(OLD/'results/metric_rows.json')
    e=engine();cells=[]
    for q in qs:
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            p=C.resolve(q[key]);h=C.sha(p)
            if h!=q[hkey]:raise ValueError('Fixed input hash changed '+key)
            protected[str(p)]=h
        query=C.resolve(q['query_descriptor']['path'])
        if C.sha(query)!=q['query_descriptor']['sha256']:raise ValueError('Fixed query changed')
        protected[str(query)]=C.sha(query)
        for kind in ('raw','post'):
            folder=e.producer('EXPOSURE',q,kind);r=C.read(folder/'receipt.json')
            protected.update(r['files']);protected[str(folder/'receipt.json')]=C.sha(folder/'receipt.json')
        for stage in STAGES:
            row,score,folder=e.authenticate_score('EXPOSURE',q,stage)
            if any(row[m]!=original_rows[stage,q['request_id']][m] for m in fields):raise ValueError('Cached baseline changed')
            cells.append(dict(label='EXPOSURE',request_id=q['request_id'],stage=stage,source=str(folder)))
            for name in ('row.json','score.json','complete.json'):protected[str(folder/name)]=C.sha(folder/name)
        p=OLD/'cpu/EXPOSURE'/q['request_id']/'complete.json';r=C.read(p)
        C.verify(r['files']);protected.update(r['files']);protected[str(p)]=C.sha(p)
    oldscore=C.read(OLD/'evaluation/score_registration.json')
    C.verify(oldscore['source_hashes']);protected.update(oldscore['source_hashes'])
    for kind in ('RAW','D0'):
        p=OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json');r=C.read(p)
        C.verify(r['code']);protected.update(r['code']);protected[str(p)]=C.sha(p)
    plan=dict(status='FROZEN',schema=VERSION,labels=list(LABELS),new_labels=list(NEW_LABELS),stages=list(STAGES),budget=budget(),
        primary_contrasts=['PATCH200-PLAIN200','LOOP200-PLAIN200','LOOP200-PLAIN267','RL16-RL_ZERO'],
        baseline='EXPOSURE cached only; no model/scoring rerun',new_initial_noise_arrays=0,new_queries=0,new_cameras_or_sections=0,
        RL_transition_noise='Twelve saved new CPU arrays shared by RL_ZERO/RL16, disjoint training seeds',
        raw_barrier='All72 RAW receipts before first D0',
        chain='single officialBenchD0 -> GEO_C/sourcewinding -> HINGE_OPP015 -> ALPHA -> HOLE',
        PATCH_exception='Candidate shared-free decoder reapplied after D0; unconstrained proposal and displacement/area audit preserved',
        gallery='36 pages, seven rows x two views/two original sections;252 actual review cells',
        rating='Actual AI_PROXY required; metrics do not generate labels')
    C.write(OUT/'data/evaluation_plan.json',plan)
    result=dict(status='PASS',schema=VERSION,requests=12,parents=6,score_rows_reused=36,numeric_fields=177,
        source_score_rows=cells,protected=protected,model_calls=0,scoring_calls=0,new_noise=0,new_queries=0,
        files={str(p):C.sha(p) for p in (MANIFEST,OUT/'data/evaluation_plan.json')})
    C.write(done,result);return result


def prepare_scores():
    e=engine();CPU,scorers=e.cpu_modules()
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    prepare_panel()
    files=[Path(__file__),Path(CPU.__file__),Path(CPU.H.__file__),Path(CPU.alpha.__file__),Path(CPU.holes.__file__),Path(CPU.M.__file__)]
    code={str(p):C.sha(p) for p in files};code.update(source_hashes())
    result=dict(status='FROZEN',schema=VERSION+'_score_registration',labels=list(LABELS),stages=list(STAGES),
        eval_manifest_sha256=C.sha(MANIFEST),registration_sha256=C.sha(C.REGISTRATION),source_hashes=code,
        new_score_rows=216,reused_score_rows=36,metric_fields=C.read(MANIFEST)['metric_fields'],
        model_calls=0,new_queries=0,source_GT_success=False,all_native_seam_fields='NA_NOT_SEAM_REPRESENTATION',
        chain='Unchanged BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE')
    result.update(engineering_version=2,prior_registration_path=str(OUT/'evaluation/score_registration.json'),
        prior_registration_sha256=C.sha(OUT/'evaluation/score_registration.json'),
        amendment='Resource-snapshot adapter only; old source/scorer/metrics remain unchanged')
    C.write(SCORE_REGISTRATION,result);return result


def run_cpu(labels,mode,workers=2):
    """Private spawn dispatcher with explicit v2 registration; no old path rewrite."""
    import concurrent.futures
    import multiprocessing
    if workers not in (1,2) or mode not in ('raw','post'):raise ValueError('Frozen CPU scope')
    if not labels or len(set(labels))!=len(labels) or any(x not in NEW_LABELS for x in labels):
        raise ValueError('Only new fixed labels may be scored')
    reg=prepare_scores();e=engine();CPU,SC=e.cpu_modules();key=mode+'__'+'__'.join(labels)
    dest=OUT/'evaluation/completions'/(key+'.json')
    if dest.exists():
        r=C.read(dest);C.verify(r['files']);return r
    jobs=[(label,q,mode,C.sha(SCORE_REGISTRATION)) for label in labels for q in new_requests(label)]
    for label,q,_,_ in jobs:
        e.producer(label,q,'raw')
        if mode=='post':e.producer(label,q,'post')
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    tick=time.perf_counter();before=SC.resources();first=_cpu_worker(jobs[0])
    count=worker_count(SC.resources(),first,workers)
    dispatch=OUT/'evaluation/dispatch'/(key+'__'+str(time.time_ns())+'.json')
    C.write(dispatch,dict(workers=count,threads_per_worker=1,resources_before=before,
        resources_after_first=SC.resources(),first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0,
        score_registration=str(SCORE_REGISTRATION),score_registration_sha256=C.sha(SCORE_REGISTRATION)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*pool.map(_cpu_worker,jobs[1:])]
    C.verify(reg['source_hashes'])
    paths=[e.score_dir(label,q,'RAW')/'audit_receipt.json' if mode=='raw' else OUT/'cpu'/label/q['request_id']/'complete.json' for label,q,_,_ in jobs]
    result=dict(status='COMPLETE',mode=mode,labels=labels,cells=len(receipts),workers=count,seconds=time.perf_counter()-tick,
        model_calls=0,new_queries=0,dispatch=str(dispatch),files={str(p):C.sha(p) for p in paths})
    C.write(dest,result);return result


def prepare_rl_inputs():
    """CPU-only selection and saved transition arrays; no model/trajectory sampling."""
    if 'torch' in sys.modules:raise RuntimeError('RL preplan must be a fresh CPU process')
    import numpy as np
    from . import reward
    done=OUT/'data/rl_eval/complete.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);return r
    prepare_panel();rows=[]
    train_seeds={t['policy_noise_seed'] for s in C.read(OUT/'data/rl_schedule.json')['opportunities'] for t in s['trajectories']}
    for q in requests():
        rawpath=engine().producer('EXPOSURE',q,'raw')/'raw.npz'
        raw=C.arrays(rawpath)['raw'];reference=reward.reference(raw,int(q['K']))
        mask=np.asarray(reference['edit_mask'],bool)
        seed=C.seed('RL_EVAL',q['request_id'],'transition')
        if seed in train_seeds:raise ValueError('RL evaluation/training transition seed overlap')
        noise=reward.draw_policy_noise(seed,int(q['N']),mask)
        dest=OUT/'data/rl_eval'/q['request_id'];C.write(dest/'reference.json',reference)
        C.npz(dest/'transition.npz',dict(policy_noise=noise,edit_mask=mask,policy_noise_seed=np.asarray(seed,np.int64)))
        r=dict(request_id=q['request_id'],reference_RAW_path=str(rawpath),reference_RAW_sha256=C.sha(rawpath),
            reference_json=str(dest/'reference.json'),reference_sha256=C.sha(dest/'reference.json'),
            transition_npz=str(dest/'transition.npz'),transition_sha256=C.sha(dest/'transition.npz'),
            policy_noise_seed=seed,edit_faces=int(mask.sum()),initial_noise_sha256=q['noise_sha256'])
        rows.append(r)
    C.write(OUT/'data/rl_eval/manifest.json',dict(rows=rows,new_transition_arrays=12,new_initial_noise=0,
        shared_by=['RL_ZERO','RL16'],selection_uses_frozen_EXPOSURE_RAW=True))
    files={str(p):C.sha(p) for p in (OUT/'data/rl_eval').rglob('*') if p.is_file()}
    result=dict(status='COMPLETE',requests=12,model_calls=0,files=files);C.write(done,result);return result


def _layout(q,label):
    import numpy as np
    core,source=engine().source_assets(q)
    if label=='PATCH200':
        from . import patch
        lay=patch.inference_layout(**core)
    else:
        from r2.pilots.bench_natural_n_recipe_v1.runtime import inference_layout
        lay=inference_layout(**core)
    logits=np.full(len(lay['edge_records']),-1,np.float32)
    dest=OUT/'evaluation'/label/'layouts'/q['condition_id']
    if (dest/'receipt.json').exists():
        r=C.read(dest/'receipt.json');C.verify(r['files'])
        prior=C.arrays(dest/'layout.npz')
        if any(not np.array_equal(prior[k],v) for k,v in lay.items()):raise ValueError('Layout changed')
    else:
        if dest.exists():raise RuntimeError('Unsealed layout')
        C.npz(dest/'layout.npz',{**lay,'occupancy_logits':logits});C.npz(dest/'source_condition.npz',source)
        C.write(dest/'receipt.json',dict(status='COMPLETE',label=label,condition_id=q['condition_id'],
            occupancy_forward_calls=0,source_labels_read=False,input_sha256=q['inference_npz_sha256'],
            files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay,logits,source,dest


def _rollout_patch(model,layout,z,ledger):
    import numpy as np
    import torch
    from . import patch
    from meshflow_control.models.native import native_execution
    n,k=int(layout['N']),int(layout['K']);device=next(model.parameters()).device
    initial=patch.initial_state(layout,z)
    state=torch.from_numpy(initial.reshape(1,n,9)).to(device)
    known=torch.from_numpy(layout['known_mask'][None]).to(device);valid=torch.ones_like(known)
    y=torch.tensor([n],dtype=torch.int64,device=device);fixed=torch.from_numpy((layout['anchor_ids']>=0)[None]).to(device)
    anchors=torch.from_numpy(initial[None]).to(device);path=[initial.copy()]
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.no_grad(),native_execution(model,coordinates=state):
        for step in range(50):
            t=torch.full((1,),step/50,device=device,dtype=torch.float32)
            C.event(ledger,dict(event='FORWARD_STARTED',step=step))
            velocity=model(state,t,y,valid_mask=valid,known_mask=known)
            C.event(ledger,dict(event='FORWARD_COMPLETE',step=step))
            velocity=patch.project_velocity(velocity.float(),layout)
            current=state.reshape(1,n,3,3);nxt=current+velocity.float().reshape_as(current)/50
            nxt=torch.where(fixed[...,None],anchors,nxt)
            nxt=patch.project_coordinates(nxt,layout)
            if not bool(torch.isfinite(nxt).all()):raise FloatingPointError('PATCH nonfinite rollout')
            state=nxt.reshape(1,n,9);path.append(nxt[0].cpu().numpy().copy())
    if not torch.equal(rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):raise RuntimeError('PATCH consumed RNG')
    path=np.stack(path)
    if any(not np.array_equal(s[:k],initial[:k]) for s in path):raise ValueError('PATCH C changed')
    return path[-1],path,dict(generator_forwards=50,seconds=time.perf_counter()-tick,anchored_all51_exact=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        representation='C_ONLY_FREE_SHARED_PATCH',shared_pairs=len(layout['shared_corner_pairs']))


def sample(label,estimated_peak_mib=3500):
    if label not in NEW_LABELS:raise ValueError('Baseline sampling forbidden')
    e=engine();prepare_panel();endpoint_files=training_barrier()
    out=OUT/'evaluation'/label;complete=e.raw_completion(label)
    if complete.exists():e.verify_raw(label);return C.read(complete)
    if list(out.glob('sampling_attempt_*')):raise RuntimeError('Prior sample attempt: no silent retry')
    spec=label_spec(label);checkpoint=C.resolve(spec['generator_path']);checkpoint_hash=C.sha(checkpoint)
    protected={**endpoint_files,str(checkpoint):checkpoint_hash,str(MANIFEST):C.sha(MANIFEST),str(C.REGISTRATION):C.sha(C.REGISTRATION)}
    if label.startswith('RL'):
        plan=C.read(OUT/'data/rl_eval/complete.json');C.verify(plan['files']);protected.update(plan['files'])
    code=_code_hashes({},'RAW')
    C.write(OUT/'evaluation/inference_registrations'/(label+'__RAW.json'),dict(status='FROZEN',label=label,spec=spec,
        outputs=12,solver='Euler50',maximum_logical_vector_fields=600,
        maximum_additional_policy_forwards=120 if label=='RL16' else 0,
        new_initial_noise=0,new_transition_noise=0,source_labels_read=False,protected=protected,code=code))
    import numpy as np
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import sample as sampler
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    attempt=out/('sampling_attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    resources=C.resources();C.write(attempt/'resources_before.json',resources);dispatch_guard(resources,estimated_peak_mib)
    observer=GPUObserver(C.EXPERIMENT+'_'+label,interval=1.);observer.__enter__()
    model=None;completed=extra_calls=0;tick=time.perf_counter()
    try:
        runtime.configure_stable_runtime()
        if label.startswith('RL'):
            from . import rl
            if label=='RL16':
                payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if (payload['schema']!='bench_local_quality_rl_v1' or
                    int(payload['completed_opportunities'])!=16 or payload['update_in_progress']):
                    raise ValueError('RL evaluation requires the fixed completed16 endpoint')
                if payload['base_file_sha256']!=C.sha(C.SOURCE_INFERENCE):
                    raise ValueError('RL frozen-base identity differs')
                del payload
            model=rl.load_policy(device='cuda',checkpoint_path=None if label=='RL_ZERO' else checkpoint)
            if isinstance(model,tuple):raise TypeError('RL load_policy must return model, not implicit identity tuple')
            p=None
        else:
            p=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            if p['schema']!=spec['schema'] or p['arm']!=spec['arm'] or int(p['local_step'])!=spec['local_step'] or int(p['global_step'])!=spec['global_step'] or p.get('EMA',False):raise ValueError('Wrong supervised endpoint')
            if state_hash(p['generator_state'])!=p['generator_state_sha256']:raise ValueError('Endpoint state hash')
            if label=='LOOP200':from .loop import build_model
            else:from r2.pilots.bench_natural_n_recipe_v1.runtime import build_model
            model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
            model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
        model.eval();before=state_hash(model.state_dict())
        identity=dict(path=str(checkpoint),sha256=checkpoint_hash,state_sha256=before,local_step=spec['local_step'],global_step=spec['global_step'],EMA=False)
        C.write(attempt/'model_identity.json',identity);C.write(attempt/'resources_loaded.json',C.resources())
        plans={r['request_id']:r for r in C.read(OUT/'data/rl_eval/manifest.json')['rows']} if label.startswith('RL') else {}
        for q in requests():
            dest=out/q['request_id']/'raw'
            if dest.exists():raise RuntimeError('Existing unaggregated RAW; requires explicit recovery')
            lay,logits,source,lp=_layout(q,label);dest.mkdir(parents=True)
            C.write(dest/'attempt.json',dict(label=label,request=q,generator=identity,logical_forward_budget=50))
            noise=C.arrays(q['noise_npz'])['z'];additional=0
            if label=='PATCH200':
                from . import patch
                raw,trajectory,audit=_rollout_patch(model,lay,noise,dest/'calls.jsonl')
                mesh=patch.raw_arrays(raw,lay,logits,source)
            elif label.startswith('RL'):
                plan=plans[q['request_id']];bundle=C.arrays(plan['transition_npz']);base_raw=C.arrays(plan['reference_RAW_path'])['raw']
                def event_fn(*args,**kwargs):
                    record=dict(kwargs)
                    if args:
                        if isinstance(args[0],dict):record.update(args[0])
                        else:record['event']=str(args[0])
                    C.event(dest/'calls.jsonl',record)
                raw,trajectory,audit,_=rl.rollout(model,lay,noise,plan['policy_noise_seed'],base_raw=base_raw,
                    mode='reference' if label=='RL_ZERO' else 'current',edit_mask=bundle['edit_mask'],
                    policy_noise=bundle['policy_noise'],event_fn=event_fn)
                additional=int(audit['generator_forwards'])-50
                if not 0<=additional<=10 or (label=='RL_ZERO' and additional):raise ValueError('Unregistered RL forward count')
                C.npz(dest/'policy_noise.npz',bundle)
                C.write(dest/'policy_plan.json',plan)
                mesh=raw_arrays(raw,lay,logits,source)
            else:
                raw,trajectory,audit=sampler.rollout(model,lay,noise,'PLAIN',dest/'calls.jsonl')
                mesh=raw_arrays(raw,lay,logits,source)
            if label=='LOOP200':audit.update(loop=model.loop_metadata(),executed_block_calls=800)
            C.npz(dest/'raw.npz',mesh);C.npz(dest/'trajectory.npz',dict(state=trajectory));C.npz(dest/'source_condition.npz',source)
            write_indexed_ply(dest/'raw.ply',mesh['vertices'],mesh['faces']);C.write(dest/'audit.json',audit)
            C.write(dest/'receipt.json',dict(status='COMPLETE',execution='NEW',label=label,arm=label,request_id=q['request_id'],condition_id=q['condition_id'],
                generator=identity,checkpoint_sha256=checkpoint_hash,generator_forwards=50,logical_vector_fields=50,
                generator_forwards_semantics='Euler velocity evaluations; additional policy calls and loop blocks separately counted',
                additional_policy_forwards=additional,new_generator_forwards=50+additional,D0_forwards=0,new_D0_forwards=0,
                raw_sha256=C.sha(dest/'raw.npz'),noise_sha256=q['noise_sha256'],source_labels_read=False,head_forwards=0,
                layout_receipt=str(lp/'receipt.json'),representation=spec['mode'],audit=audit,
                files={str(f):C.sha(f) for f in dest.iterdir() if f.is_file()}))
            completed+=1;extra_calls+=additional
            if completed in (1,2,12):C.write(attempt/f'resources_output{completed}.json',C.resources())
            print(json.dumps(dict(stage='RAW',label=label,completed=completed,request_id=q['request_id'])),flush=True)
        if before!=state_hash(model.state_dict()):raise ValueError('Frozen evaluation model mutated')
        C.verify(protected);C.verify(code)
        result=dict(status='COMPLETE',label=label,subset='full',outputs=12,total_new_rollouts=12,total_new_generator_forwards=600+extra_calls,
            logical_vector_fields=600,additional_policy_forwards=extra_calls,total_new_head_forwards=0,checkpoint_sha256=checkpoint_hash,
            seconds=time.perf_counter()-tick,files={str(e.raw_receipt(q,label)):C.sha(e.raw_receipt(q,label)) for q in requests()})
        C.write(complete,result);return result
    except BaseException as exc:
        C.write(attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_outputs=completed,
            additional_policy_forwards_returned=extra_calls,no_automatic_retry=True));raise
    finally:
        observer.__exit__(None,None,None)
        C.write(attempt/'resources.json',dict(summary=observer.summary(),samples=observer.samples))
        del model;torch.cuda.empty_cache()


def collect():
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,aggregate,numeric
    done=OUT/'results/completion.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);C.verify(r['source_rows']);return r
    reg=C.read(SCORE_REGISTRATION);C.verify(reg['source_hashes'])
    metrics=C.read(MANIFEST)['metric_fields'];rows=[];sources={};e=engine()
    for label in LABELS:
        for q in requests():
            for stage in STAGES:
                src,score,folder=e.authenticate_score(label,q,stage);row=decorate(src,label,q,stage)
                row.update(seam_applicability=score['seam_to_bulk']['status'],eval_group=q['eval_group'],report_group=q['report_group'],
                    result_origin='REUSED_READ_ONLY' if label=='EXPOSURE' else 'NEW_REGISTERED',source_score_label=label)
                if any(row[m]!=src[m] for m in metrics):raise ValueError('Original metric altered')
                rows.append(row)
                for name in ('row.json','score.json','complete.json'):sources[str(folder/name)]=C.sha(folder/name)
    if len(rows)!=252:raise ValueError('Score cardinality differs')
    conditions=aggregate(rows,('label','stage','report_group','eval_group','parent_id','condition_id'),metrics)
    parents=aggregate(conditions,('label','stage','report_group','eval_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','report_group','eval_group'),metrics)
    by={(r['label'],r['stage'],r['request_id']):r for r in rows};pairs=[]
    contrasts=[('EXPOSURE',label) for label in NEW_LABELS]+[('PLAIN200','PATCH200'),('PLAIN200','LOOP200'),('PLAIN267','LOOP200'),('RL_ZERO','RL16')]
    for ref,cand in contrasts:
        for stage in STAGES:
            for q in requests():
                a,b=by[ref,stage,q['request_id']],by[cand,stage,q['request_id']]
                row=dict(reference=ref,candidate=cand,stage=stage,**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group','report_group')})
                for m in metrics:
                    row[m+'__reference']=a[m];row[m+'__candidate']=b[m]
                    row[m+'__delta']=b[m]-a[m] if numeric(a[m]) and numeric(b[m]) else None
                pairs.append(row)
    delta=[m+'__delta' for m in metrics]
    pc=aggregate(pairs,('reference','candidate','stage','report_group','eval_group','parent_id','condition_id'),delta)
    pp=aggregate(pc,('reference','candidate','stage','report_group','eval_group','parent_id'),delta)
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=summary,
        paired_per_output=pairs,paired_per_condition=pc,paired_per_parent=pp)
    for name,table in tables.items():C.csvwrite(OUT/'results'/(name+'.csv'),table)
    C.write(OUT/'results/metric_rows.json',rows)
    files={str(OUT/'results'/(name+'.csv')):C.sha(OUT/'results'/(name+'.csv')) for name in tables}
    files[str(OUT/'results/metric_rows.json')]=C.sha(OUT/'results/metric_rows.json')
    result=dict(status='COMPLETE',rows=252,new_score_rows=216,reused_score_rows=36,paired_rows=len(pairs),metric_fields=metrics,
        source_rows=sources,model_calls=0,scoring_calls=0,new_queries=0,files=files)
    C.write(done,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','prepare-rl','prepare-scores','cpu-contract','sample','collect'))
    p.add_argument('--label',choices=NEW_LABELS);p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args()
    if a.action=='prepare':r=prepare_panel()
    elif a.action=='prepare-rl':r=prepare_rl_inputs()
    elif a.action=='prepare-scores':r=prepare_scores()
    elif a.action=='cpu-contract':r=engine().cpu_contract()
    elif a.action=='collect':r=collect()
    else:
        if not a.label:p.error('--label required')
        r=sample(a.label,a.estimated_peak_mib)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected','source_rows','source_score_rows','source_hashes','rows','metric_fields')},ensure_ascii=False))


if __name__=='__main__':main()
