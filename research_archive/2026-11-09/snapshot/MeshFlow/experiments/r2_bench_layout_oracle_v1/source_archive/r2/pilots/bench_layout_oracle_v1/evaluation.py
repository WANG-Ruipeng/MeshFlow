"""Frozen paired layout-information evaluation; original scorers stay read-only.

Only ``sample`` invokes the generator. Source-region P is privileged diagnostic
information, not a C-only deployment result. RAW for every registered mode must
finish before the first official single D0 invocation.
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
P_MANIFEST = OUT/'data/layout_eval_manifest.json'
LABELS = ('EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL')
NEW_LABELS = LABELS[1:]
STAGES = ('RAW','BASE_POST','CLEAN_FINAL_POST')
VERSION = 'bench_layout_oracle_evaluation_v1'
SCORE_REGISTRATION = OUT/'evaluation/score_registration.json'


def dispatch_guard(snapshot, estimate_mib):
    from r2.pilots.bench_three_mechanisms_v1.evaluation import dispatch_guard as guard
    return guard(snapshot, estimate_mib)


def requests(label=None):
    if label is not None and label not in LABELS: raise ValueError(label)
    qs=C.read(MANIFEST)['requests']
    if not qs or len({q['request_id'] for q in qs})!=len(qs): raise ValueError('Empty/duplicate panel')
    reg=C.read(C.REGISTRATION)
    expected=reg.get('eval_requests',12)
    if len(qs)!=expected: raise ValueError('Registered evaluation cardinality changed')
    return qs


def budget():
    n=len(requests()); return dict(requests=n,parents=len({q['parent_id'] for q in requests()}),
        new_RAW=3*n,new_D0=3*n,generator_forwards=150*n,logical_Euler_vector_fields=150*n,
        new_score_rows=9*n,reused_score_rows=3*n,total_score_rows=12*n)


def root_for(label,q=None):
    if label not in LABELS: raise ValueError(label)
    return OLD if label=='EXPOSURE' else OUT


def new_requests(label):
    if label not in LABELS: raise ValueError(label)
    return requests(label) if label in NEW_LABELS else []


def label_spec(label):
    if label not in NEW_LABELS: raise ValueError('Baseline sampling forbidden')
    arm='PLAIN' if label=='PLAIN1000' else 'LAYOUT'
    return dict(mode=arm,arm=arm,local_step=1000,global_step=C.BASE_STEP+1000,subset='full',
        schema='bench_layout_oracle_generator_v1',
        generator_path=str(OUT/'training'/arm/'generator_step1000.pt'),
        input_scope='ORACLE_PRIVILEGED_SOURCE_REGIONS' if label=='LAYOUT1000' else 'C_ONLY',
        layout_mask='REGISTERED_VALID' if label=='LAYOUT1000' else 'ALL_FALSE')


def training_barrier():
    files={}
    for label in ('PLAIN1000','LAYOUT1000'):
        p=C.resolve(label_spec(label)['generator_path'])
        if not p.exists(): raise FileNotFoundError('Wait for both fixed training endpoints: '+str(p))
        sidecar=p.with_suffix('.json')
        if sidecar.exists():
            r=C.read(sidecar)
            if r.get('sha256')!=C.sha(p): raise ValueError('Endpoint sidecar hash differs')
            files[str(sidecar)]=C.sha(sidecar)
        files[str(p)]=C.sha(p)
    return files


def _code_hashes(code,kind):
    inherited=C.read(OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json'))['code']
    C.verify(inherited)
    return {**inherited,**{str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')},**code}


def all_raw_barrier():
    audit=C.read(OUT/'audits/reference_cache_audit.json'); C.verify(audit['files']); C.verify(audit['protected'])
    paths=[]
    for label in NEW_LABELS:
        engine().verify_raw(label)
        paths.extend(engine().raw_receipt(q,label) for q in requests())
    if len(paths)!=budget()['new_RAW']: raise ValueError('Every registered RAW must precede D0')
    return paths


def _cpu_worker(job):
    label,q,mode,regsha=job
    e=engine(); result=e.raw_one(label,q,regsha) if mode=='raw' else e.post_one(label,q,regsha)
    if 'torch' in sys.modules: raise RuntimeError('CPU scoring imported torch')
    print('EVALUATED',mode,label,q['request_id'],flush=True)
    return result


@lru_cache(maxsize=1)
def engine():
    from r2.pilots.bench_four_routes_v1.exposure import evaluation as original
    before=(original.OUT,original.MANIFEST,original.LABELS)
    ns=dict(vars(original))
    for name,obj in vars(original).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(original):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            fn.__kwdefaults__=obj.__kwdefaults__; ns[name]=fn
    ns.update(OUT=OUT,OLD_M=OLD,MANIFEST=MANIFEST,LABELS=LABELS,NEW_LABELS=NEW_LABELS,
        ARMS={label:label for label in NEW_LABELS},VERSION=VERSION,__file__=__file__,
        requests=requests,new_requests=new_requests,root_for=root_for,budget=budget,
        prepare_panel=prepare_panel,prepare_manifest=prepare_panel,prepare_scores=prepare_scores,
        registration_path=lambda:C.REGISTRATION,label_spec=label_spec,
        reuse_spec=lambda q,label:dict(root=str(OLD),label=label) if label=='EXPOSURE' else None,
        inference_code_hashes=_code_hashes,all_raw_barrier=all_raw_barrier,cpu_one=_cpu_worker,
        sample=sample,collect=collect,run_cpu=run_cpu)
    if (original.OUT,original.MANIFEST,original.LABELS)!=before: raise AssertionError('Original globals changed')
    return types.SimpleNamespace(**ns)


def layout_record(q):
    manifest=C.read(P_MANIFEST)
    rows=manifest.get('requests',manifest.get('rows'))
    if not isinstance(rows,list): raise ValueError('Layout manifest needs request rows')
    found=[r for r in rows if r['request_id']==q['request_id']]
    if len(found)!=1: raise ValueError('Exactly one registered layout per request required')
    row=found[0]; path=C.resolve(row['layout_npz'])
    if C.sha(path)!=row['layout_npz_sha256']: raise ValueError('Layout input hash changed')
    bundle=C.arrays(path)
    import numpy as np
    if (bundle['tokens'].shape!=(2,8) or bundle['tokens'].dtype!=np.float32 or
        bundle['valid'].shape!=(2,) or bundle['valid'].dtype!=np.bool_ or
        not np.isfinite(bundle['tokens']).all()): raise ValueError('Layout token contract')
    return row,bundle


def prepare_panel():
    done=OUT/'audits/reference_cache_audit.json'
    if done.exists():
        r=C.read(done); C.verify(r['files']); C.verify(r['protected']); return r
    manifest=C.read(MANIFEST); qs=requests(); fields=manifest['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177: raise ValueError('Original177 metrics changed')
    C.verify(manifest.get('protected',{}))
    src=OLD/'data/eval_manifest.json'; source=C.read(src)
    original={q['request_id']:q for q in source['requests']}
    if any(q!=original[q['request_id']] for q in qs): raise ValueError('Original request fields changed')
    protected={**manifest.get('protected',{}),str(src):C.sha(src),str(P_MANIFEST):C.sha(P_MANIFEST),
        str(OLD/'results/completion.json'):C.sha(OLD/'results/completion.json')}
    oldrows={(r['stage'],r['request_id']):r for r in C.read(OLD/'results/metric_rows.json') if r['label']=='EXPOSURE'}
    protected[str(OLD/'results/metric_rows.json')]=C.sha(OLD/'results/metric_rows.json')
    e=engine(); cells=[]
    for q in qs:
        lr,_=layout_record(q); protected[lr['layout_npz']]=lr['layout_npz_sha256']
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),
                        ('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            p=C.resolve(q[key]); h=C.sha(p)
            if h!=q[hkey]: raise ValueError('Input changed: '+key)
            protected[str(p)]=h
        query=C.resolve(q['query_descriptor']['path'])
        if C.sha(query)!=q['query_descriptor']['sha256']: raise ValueError('Query changed')
        protected[str(query)]=C.sha(query)
        for kind in ('raw','post'):
            folder=e.producer('EXPOSURE',q,kind); receipt=C.read(folder/'receipt.json')
            protected.update(receipt['files']); protected[str(folder/'receipt.json')]=C.sha(folder/'receipt.json')
        for stage in STAGES:
            row,score,folder=e.authenticate_score('EXPOSURE',q,stage)
            if any(row[m]!=oldrows[stage,q['request_id']][m] for m in fields): raise ValueError('Cached baseline changed')
            cells.append(dict(label='EXPOSURE',request_id=q['request_id'],stage=stage,source=str(folder)))
            for name in ('row.json','score.json','complete.json'): protected[str(folder/name)]=C.sha(folder/name)
        p=OLD/'cpu/EXPOSURE'/q['request_id']/'complete.json'; r=C.read(p)
        C.verify(r['files']); protected.update(r['files']); protected[str(p)]=C.sha(p)
    oldscore=C.read(OLD/'evaluation/score_registration.json'); C.verify(oldscore['source_hashes']); protected.update(oldscore['source_hashes'])
    for kind in ('RAW','D0'):
        p=OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json'); r=C.read(p)
        C.verify(r['code']); protected.update(r['code']); protected[str(p)]=C.sha(p)
    plan=dict(status='FROZEN',schema=VERSION,labels=list(LABELS),new_labels=list(NEW_LABELS),stages=list(STAGES),budget=budget(),
        primary_contrasts=['LAYOUT1000-PLAIN1000','LAYOUT1000-LAYOUT_NULL'],
        scope='LAYOUT1000 uses privileged source coarse regions; not C-only deployment or a mathematical upper bound',
        baseline='EXPOSURE cached; no model/scoring rerun',new_initial_noise_arrays=0,new_queries=0,new_cameras_or_sections=0,
        raw_barrier='All registered new RAWs before first D0',
        chain='single officialBenchD0 -> GEO_C/sourcewinding -> HINGE_OPP015 -> ALPHA -> HOLE',
        rating='Actual AI_PROXY required; metrics never generate visual labels')
    C.write(OUT/'data/evaluation_plan.json',plan)
    result=dict(status='PASS',schema=VERSION,requests=len(qs),parents=len({q['parent_id'] for q in qs}),
        score_rows_reused=3*len(qs),numeric_fields=177,source_score_rows=cells,protected=protected,
        model_calls=0,scoring_calls=0,new_noise=0,new_queries=0,
        files={str(p):C.sha(p) for p in (MANIFEST,P_MANIFEST,OUT/'data/evaluation_plan.json')})
    C.write(done,result); return result


def prepare_scores():
    e=engine(); CPU,scorers=e.cpu_modules()
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    prepare_panel()
    files=[Path(__file__),Path(CPU.__file__),Path(CPU.H.__file__),Path(CPU.alpha.__file__),Path(CPU.holes.__file__),Path(CPU.M.__file__)]
    code={str(p):C.sha(p) for p in files}; code.update(source_hashes())
    result=dict(status='FROZEN',schema=VERSION+'_score_registration',labels=list(LABELS),stages=list(STAGES),
        eval_manifest_sha256=C.sha(MANIFEST),registration_sha256=C.sha(C.REGISTRATION),source_hashes=code,
        new_score_rows=budget()['new_score_rows'],reused_score_rows=budget()['reused_score_rows'],
        metric_fields=C.read(MANIFEST)['metric_fields'],model_calls=0,new_queries=0,source_GT_success=False,
        all_native_seam_fields='NA_NOT_SEAM_REPRESENTATION',chain='Unchanged BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE')
    C.write(SCORE_REGISTRATION,result); return result


def run_cpu(labels,mode,workers=2):
    import concurrent.futures
    import multiprocessing
    if workers not in (1,2) or mode not in ('raw','post'): raise ValueError('Frozen CPU scope')
    if not labels or len(set(labels))!=len(labels) or any(x not in NEW_LABELS for x in labels): raise ValueError('Unknown new labels')
    reg=prepare_scores(); e=engine(); CPU,SC=e.cpu_modules(); key=mode+'__'+'__'.join(labels)
    dest=OUT/'evaluation/completions'/(key+'.json')
    if dest.exists(): r=C.read(dest); C.verify(r['files']); return r
    jobs=[(label,q,mode,C.sha(SCORE_REGISTRATION)) for label in labels for q in new_requests(label)]
    for label,q,_,_ in jobs:
        e.producer(label,q,'raw')
        if mode=='post': e.producer(label,q,'post')
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    tick=time.perf_counter(); before=SC.resources(); first=_cpu_worker(jobs[0]); count=worker_count(SC.resources(),first,workers)
    dispatch=OUT/'evaluation/dispatch'/(key+'__'+str(time.time_ns())+'.json')
    C.write(dispatch,dict(workers=count,threads_per_worker=1,resources_before=before,resources_after_first=SC.resources(),
        first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0,score_registration_sha256=C.sha(SCORE_REGISTRATION)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*pool.map(_cpu_worker,jobs[1:])]
    C.verify(reg['source_hashes'])
    paths=[e.score_dir(label,q,'RAW')/'audit_receipt.json' if mode=='raw' else OUT/'cpu'/label/q['request_id']/'complete.json' for label,q,_,_ in jobs]
    result=dict(status='COMPLETE',mode=mode,labels=labels,cells=len(receipts),workers=count,seconds=time.perf_counter()-tick,
        model_calls=0,new_queries=0,dispatch=str(dispatch),files={str(p):C.sha(p) for p in paths})
    C.write(dest,result); return result


def _layout(q,label):
    import numpy as np
    from r2.pilots.bench_natural_n_recipe_v1.runtime import inference_layout
    core,source=engine().source_assets(q); lay=inference_layout(**core)
    logits=np.full(len(lay['edge_records']),-1,np.float32)
    dest=OUT/'evaluation'/label/'layouts'/q['condition_id']
    if (dest/'receipt.json').exists():
        r=C.read(dest/'receipt.json'); C.verify(r['files']); prior=C.arrays(dest/'layout.npz')
        if any(not np.array_equal(prior[k],v) for k,v in lay.items()): raise ValueError('Native layout changed')
    else:
        if dest.exists(): raise RuntimeError('Unsealed layout')
        C.npz(dest/'layout.npz',{**lay,'occupancy_logits':logits}); C.npz(dest/'source_condition.npz',source)
        C.write(dest/'receipt.json',dict(status='COMPLETE',label=label,condition_id=q['condition_id'],
            occupancy_forward_calls=0,source_labels_read=False,input_sha256=q['inference_npz_sha256'],
            files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay,logits,source,dest


def sample(label,estimated_peak_mib=3500):
    if label not in NEW_LABELS: raise ValueError('Baseline sampling forbidden')
    e=engine(); prepare_panel(); endpoint_files=training_barrier()
    out=OUT/'evaluation'/label; complete=e.raw_completion(label)
    if complete.exists(): e.verify_raw(label); return C.read(complete)
    if list(out.glob('sampling_attempt_*')): raise RuntimeError('Prior sample attempt; no silent retry')
    spec=label_spec(label); checkpoint=C.resolve(spec['generator_path']); checkpoint_hash=C.sha(checkpoint)
    protected={**endpoint_files,str(checkpoint):checkpoint_hash,str(MANIFEST):C.sha(MANIFEST),
        str(P_MANIFEST):C.sha(P_MANIFEST),str(C.REGISTRATION):C.sha(C.REGISTRATION)}
    code=_code_hashes({},'RAW')
    C.write(OUT/'evaluation/inference_registrations'/(label+'__RAW.json'),dict(status='FROZEN',label=label,spec=spec,
        outputs=len(requests()),solver='Euler50',maximum_logical_vector_fields=50*len(requests()),
        new_initial_noise=0,input_scope=spec['input_scope'],protected=protected,code=code))
    import numpy as np
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import sample as sampler
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    attempt=out/('sampling_attempt_'+str(time.time_ns())); attempt.mkdir(parents=True)
    resources=C.resources(); C.write(attempt/'resources_before.json',resources); dispatch_guard(resources,estimated_peak_mib)
    observer=GPUObserver(C.EXPERIMENT+'_'+label,interval=1.); observer.__enter__()
    model=None; completed=0; tick=time.perf_counter()
    try:
        runtime.configure_stable_runtime()
        p=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if (p['schema']!=spec['schema'] or p['arm']!=spec['arm'] or int(p['local_step'])!=1000 or
            int(p['global_step'])!=spec['global_step'] or p.get('EMA',False)): raise ValueError('Wrong endpoint')
        if state_hash(p['generator_state'])!=p['generator_state_sha256']: raise ValueError('Endpoint state hash')
        if spec['arm']=='PLAIN':
            from r2.pilots.bench_natural_n_recipe_v1.runtime import build_model
        else:
            from .model import build_model
        model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
        model.load_state_dict(p['generator_state'],strict=True); model.to('cuda').set_trainable(False).eval()
        before=state_hash(model.state_dict())
        identity=dict(path=str(checkpoint),sha256=checkpoint_hash,state_sha256=before,local_step=1000,
            global_step=spec['global_step'],EMA=False,input_scope=spec['input_scope'])
        if spec['arm']=='LAYOUT': identity['layout_module']=model.layout_metadata()
        C.write(attempt/'model_identity.json',identity); C.write(attempt/'resources_loaded.json',C.resources())
        for q in requests():
            dest=out/q['request_id']/'raw'
            if dest.exists(): raise RuntimeError('Existing unaggregated RAW requires explicit recovery')
            lay,logits,source,lp=_layout(q,label); dest.mkdir(parents=True)
            lr,pbundle=layout_record(q); protected[lr['layout_npz']]=lr['layout_npz_sha256']
            C.write(dest/'attempt.json',dict(label=label,request=q,generator=identity,logical_forward_budget=50))
            noise=C.arrays(q['noise_npz'])['z']
            if spec['arm']=='PLAIN':
                raw,trajectory,audit=sampler.rollout(model,lay,noise,'PLAIN',dest/'calls.jsonl')
            else:
                tokens=torch.from_numpy(pbundle['tokens'][None]).to('cuda')
                valid=torch.from_numpy(pbundle['valid'][None]).to('cuda')
                if label=='LAYOUT_NULL': valid=torch.zeros_like(valid)
                with model.layout_context(tokens,valid):
                    raw,trajectory,audit=sampler.rollout(model,lay,noise,'PLAIN',dest/'calls.jsonl')
            mesh=raw_arrays(raw,lay,logits,source)
            C.npz(dest/'raw.npz',mesh); C.npz(dest/'trajectory.npz',dict(state=trajectory)); C.npz(dest/'source_condition.npz',source)
            C.npz(dest/'layout_information.npz',dict(tokens=pbundle['tokens'],
                valid=pbundle['valid'] if label=='LAYOUT1000' else np.zeros(2,bool),center_C=pbundle['center_C']))
            C.write(dest/'layout_information.json',dict(record=lr,input_scope=spec['input_scope'],
                source_coarse_regions_supplied=label=='LAYOUT1000',missing_does_not_mean_absent=True))
            write_indexed_ply(dest/'raw.ply',mesh['vertices'],mesh['faces']); C.write(dest/'audit.json',audit)
            C.write(dest/'receipt.json',dict(status='COMPLETE',execution='NEW',label=label,arm=spec['arm'],
                request_id=q['request_id'],condition_id=q['condition_id'],generator=identity,checkpoint_sha256=checkpoint_hash,
                generator_forwards=50,logical_vector_fields=50,new_generator_forwards=50,D0_forwards=0,new_D0_forwards=0,
                raw_sha256=C.sha(dest/'raw.npz'),noise_sha256=q['noise_sha256'],source_labels_read=label=='LAYOUT1000',
                input_scope=spec['input_scope'],head_forwards=0,
                head_accounting='Legacy occupancy head only; layout branch is inside each generator forward',
                layout_branch_invocations=50 if spec['arm']=='LAYOUT' else 0,
                layout_receipt=str(lp/'receipt.json'),
                representation='NATIVE_ALL_FREE',audit=audit,files={str(f):C.sha(f) for f in dest.iterdir() if f.is_file()}))
            completed+=1
            if completed in (1,2,len(requests())): C.write(attempt/f'resources_output{completed}.json',C.resources())
            print(json.dumps(dict(stage='RAW',label=label,completed=completed,request_id=q['request_id'])),flush=True)
        if before!=state_hash(model.state_dict()): raise ValueError('Frozen evaluation model mutated')
        C.verify(protected); C.verify(code)
        result=dict(status='COMPLETE',label=label,subset='full',outputs=len(requests()),total_new_rollouts=len(requests()),
            total_new_generator_forwards=50*len(requests()),logical_vector_fields=50*len(requests()),total_new_head_forwards=0,
            checkpoint_sha256=checkpoint_hash,seconds=time.perf_counter()-tick,input_scope=spec['input_scope'],
            files={str(e.raw_receipt(q,label)):C.sha(e.raw_receipt(q,label)) for q in requests()})
        C.write(complete,result); return result
    except BaseException as exc:
        C.write(attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_outputs=completed,no_automatic_retry=True)); raise
    finally:
        observer.__exit__(None,None,None)
        C.write(attempt/'resources.json',dict(summary=observer.summary(),samples=observer.samples))
        del model; torch.cuda.empty_cache()


def collect():
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,aggregate,numeric
    done=OUT/'results/completion.json'
    if done.exists(): r=C.read(done); C.verify(r['files']); C.verify(r['source_rows']); return r
    reg=C.read(SCORE_REGISTRATION); C.verify(reg['source_hashes'])
    metrics=C.read(MANIFEST)['metric_fields']; rows=[]; sources={}; e=engine()
    for label in LABELS:
        for q in requests():
            for stage in STAGES:
                src,score,folder=e.authenticate_score(label,q,stage); row=decorate(src,label,q,stage)
                row.update(seam_applicability=score['seam_to_bulk']['status'],eval_group=q['eval_group'],report_group=q['report_group'],
                    result_origin='REUSED_READ_ONLY' if label=='EXPOSURE' else 'NEW_REGISTERED',source_score_label=label,
                    input_scope='ORACLE_PRIVILEGED_SOURCE_REGIONS' if label=='LAYOUT1000' else 'C_ONLY',
                    source_style_success_criterion=False)
                if any(row[m]!=src[m] for m in metrics): raise ValueError('Original metric altered')
                rows.append(row)
                for name in ('row.json','score.json','complete.json'): sources[str(folder/name)]=C.sha(folder/name)
    if len(rows)!=budget()['total_score_rows']: raise ValueError('Score cardinality differs')
    conditions=aggregate(rows,('label','stage','report_group','eval_group','parent_id','condition_id'),metrics)
    parents=aggregate(conditions,('label','stage','report_group','eval_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','report_group','eval_group'),metrics)
    by={(r['label'],r['stage'],r['request_id']):r for r in rows}; pairs=[]
    contrasts=[('EXPOSURE',label) for label in NEW_LABELS]+[('PLAIN1000','LAYOUT1000'),('PLAIN1000','LAYOUT_NULL'),('LAYOUT_NULL','LAYOUT1000')]
    for ref,cand in contrasts:
        for stage in STAGES:
            for q in requests():
                a,b=by[ref,stage,q['request_id']],by[cand,stage,q['request_id']]
                row=dict(reference=ref,candidate=cand,stage=stage,**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group','report_group')})
                for m in metrics:
                    row[m+'__reference']=a[m]; row[m+'__candidate']=b[m]
                    row[m+'__delta']=b[m]-a[m] if numeric(a[m]) and numeric(b[m]) else None
                pairs.append(row)
    delta=[m+'__delta' for m in metrics]
    pc=aggregate(pairs,('reference','candidate','stage','report_group','eval_group','parent_id','condition_id'),delta)
    pp=aggregate(pc,('reference','candidate','stage','report_group','eval_group','parent_id'),delta)
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=summary,
        paired_per_output=pairs,paired_per_condition=pc,paired_per_parent=pp)
    for name,table in tables.items(): C.csvwrite(OUT/'results'/(name+'.csv'),table)
    C.write(OUT/'results/metric_rows.json',rows)
    files={str(OUT/'results'/(name+'.csv')):C.sha(OUT/'results'/(name+'.csv')) for name in tables}
    files[str(OUT/'results/metric_rows.json')]=C.sha(OUT/'results/metric_rows.json')
    result=dict(status='COMPLETE',rows=len(rows),new_score_rows=budget()['new_score_rows'],
        reused_score_rows=budget()['reused_score_rows'],paired_rows=len(pairs),metric_fields=metrics,source_rows=sources,
        model_calls=0,scoring_calls=0,new_queries=0,files=files)
    C.write(done,result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','prepare-scores','sample','collect'))
    p.add_argument('--label',choices=NEW_LABELS); p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args()
    if a.action=='prepare': r=prepare_panel()
    elif a.action=='prepare-scores': r=prepare_scores()
    elif a.action=='collect': r=collect()
    else:
        if not a.label: p.error('--label required')
        r=sample(a.label,a.estimated_peak_mib)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected','source_rows','source_score_rows','source_hashes','metric_fields')},ensure_ascii=False))


if __name__=='__main__': main()
