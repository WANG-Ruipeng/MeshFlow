"""Frozen Coverage30 evaluation: read-only P0, registered A/B Euler50 and D0.

Only explicit sample/d0 subcommands can import a neural framework. CPU stages
reuse the unchanged 177-field scorer and guarded HINGE/ALPHA/HOLE cascade.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse, collections, concurrent.futures, copy, csv, hashlib, inspect, io, json
import multiprocessing, resource, sys, time, traceback, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_exposure_allocation_v1'
OLD = ROOT/'experiments/r2_bench_seat_coverage_v1'
MANIFEST = OUT/'data/eval_manifest.json'
LABELS = ('P0', 'A', 'B')
NEW_LABELS = ('A', 'B')
ARMS = {'A':'A_50_50', 'B':'B_75_25'}
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
GROUP_MAP = {'CURRENT_TRAIN':'OLD', 'NEW_TRAIN':'FOCUS', 'DEV':'DEV', 'NEW_PARENT_HOLDOUT':'HOLDOUT'}
VERSION = 'bench_exposure_allocation_evaluation_v1'


def resolve(path):
    s=str(path).replace('\\','/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    p=Path(s);return p if p.is_absolute() else ROOT/p


def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def immutable(path, obj):
    path=resolve(path)
    if path.exists():
        if read(path)!=obj:raise ValueError('Immutable file differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def verify(book):
    for p,h in book.items():
        if sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))


def arrays(path):
    import numpy as np
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def csvwrite(path, rows):
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import csvwrite as write
    return write(resolve(path), rows)


def requests(label=None):
    if label is not None and label not in LABELS:raise ValueError('Unknown model label')
    qs=read(MANIFEST)['requests']
    if len(qs)!=30 or len({q['request_id'] for q in qs})!=30:raise ValueError('Fixed30 request identity differs')
    return qs


def storage_label(label):return 'COVERAGE_FINAL' if label=='P0' else label
def root_for(label):return OLD if label=='P0' else OUT


def raw_completion(label):return root_for(label)/'evaluation'/storage_label(label)/'raw_complete_full.json'
def raw_receipt(q,label):return root_for(label)/'evaluation'/storage_label(label)/q['request_id']/'raw/receipt.json'


def verify_one_raw(q,label):
    p=raw_receipt(q,label);r=read(p)
    if (r['status']!='COMPLETE' or r['label']!=storage_label(label) or r['request_id']!=q['request_id']
        or r['generator_forwards']!=50 or r['D0_forwards']!=0
        or r['noise_sha256']!=q['noise_sha256'] or r['raw_sha256']!=sha(p.parent/'raw.npz')):
        raise ValueError('RAW producer identity differs: '+str(p))
    verify(r['files']);return p


def verify_raw(label):
    qs=requests(label);done=read(raw_completion(label))
    if done['status']!='COMPLETE' or done['outputs']!=30:raise ValueError('Incomplete RAW wave '+label)
    verify(done['files'])
    return qs,[verify_one_raw(q,label) for q in qs]


def producer(label,q,kind):
    if kind not in ('raw','post'):raise ValueError(kind)
    folder=root_for(label)/'evaluation'/storage_label(label)/q['request_id']/kind
    if kind=='raw':verify_one_raw(q,label);return folder
    r=read(folder/'receipt.json')
    if (r['status']!='COMPLETE' or r['label']!=storage_label(label) or r['request_id']!=q['request_id']
        or r['D0_forwards']!=1 or r['GEO_calls']!=1 or r['post_sha256']!=sha(folder/'post.npz')):
        raise ValueError('POST producer identity differs: '+str(folder))
    verify(r['files']);return folder


def score_dir(label,q,stage):
    root=root_for(label);label=storage_label(label)
    if stage=='RAW':return root/'evaluation/scores'/label/q['request_id']/'RAW'
    if stage not in STAGES:raise ValueError(stage)
    return root/'cpu'/label/q['request_id']/stage/'scores'


def authenticate_score(label,q,stage):
    folder=score_dir(label,q,stage);done=read(folder/'complete.json')
    if done['status']!='PASS' or done['model_calls']!=0:raise ValueError('Incomplete original score')
    verify(done['files'])
    for name in ('row.json','score.json'):
        p=folder/name
        if not any(resolve(s)==p and h==sha(p) for s,h in done['files'].items()):raise ValueError('Score file not sealed')
    src=read(folder/'row.json');score=read(folder/'score.json')
    if score['seam_to_bulk']['status']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Native seam status must remain NA')
    for k in ('request_id','condition_id','parent_id','split'):
        if src[k]!=q[k]:raise ValueError('Original score identity differs '+k)
    for k in ('N','K','noise_index'):
        if int(src[k])!=int(q[k]):raise ValueError('Original score numeric identity differs '+k)
    return src,score,folder


def prepare_panel():
    """Authenticate existing P0 caches, freeze the exact original30 requests; no scorer."""
    path=OUT/'audits/P0_cache_audit.json'
    if path.exists():
        r=read(path);verify(r['files']);verify(r['protected']);return r
    original=read(OLD/'eval_manifest.json');qs=copy.deepcopy(original['requests'])
    if len(qs)!=30:raise ValueError('Coverage panel is not30')
    if collections.Counter(q['eval_group'] for q in qs)!={'CURRENT_TRAIN':16,'NEW_TRAIN':4,'DEV':8,'NEW_PARENT_HOLDOUT':2}:raise ValueError('Original group counts changed')
    fields=read(OLD/'results/completion.json')['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Original numeric metric dictionary differs')
    frozen={str(OLD/'eval_manifest.json'):sha(OLD/'eval_manifest.json'),str(OLD/'results/completion.json'):sha(OLD/'results/completion.json')}
    cells=[]
    for q in qs:
        q['report_group']=GROUP_MAP[q['eval_group']]
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('noise_npz','noise_sha256')):
            p=resolve(q[key]);h=sha(p)
            if h!=q[hkey]:raise ValueError('Inherited input/hash changed '+key)
            frozen[str(p)]=h
        p=resolve(q['query_descriptor']['path']);h=sha(p)
        if h!=q['query_descriptor']['sha256']:raise ValueError('Inherited query changed')
        frozen[str(p)]=h
        for kind in ('raw','post'):
            folder=producer('P0',q,kind);frozen[str(folder/'receipt.json')]=sha(folder/'receipt.json')
            for p in folder.iterdir():
                if p.is_file() and p.suffix in ('.npz','.json'):frozen[str(p)]=sha(p)
        for stage in STAGES:
            row,score,folder=authenticate_score('P0',q,stage)
            cells.append(dict(request_id=q['request_id'],stage=stage,report_group=q['report_group'],
                N=q['N'],K=q['K'],seam_status=score['seam_to_bulk']['status'],source=str(folder),
                missing_declared_fields=[m for m in fields if m not in row]))
            if cells[-1]['missing_declared_fields']:raise ValueError('Original177 field absent')
            for name in ('complete.json','row.json','score.json'):
                p=folder/name;frozen[str(p)]=sha(p)
    if any({k:v for k,v in q.items() if k!='report_group'}!=original_q for q,original_q in zip(qs,original['requests'])):
        raise ValueError('An original request field was changed')
    # q values are inherited verbatim; this extra descriptor changes no scientific input.
    manifest=dict(schema=VERSION+'_panel',status='FROZEN',labels=list(LABELS),new_labels=list(NEW_LABELS),
        stages=list(STAGES),requests=qs,requests_per_label=dict.fromkeys(LABELS,30),
        source_manifest=str(OLD/'eval_manifest.json'),source_manifest_sha256=sha(OLD/'eval_manifest.json'),
        p0=dict(root=str(OLD),label='COVERAGE_FINAL',mode='READ_ONLY_NO_REGENERATION'),
        groups={'OLD':dict(parents=8,requests=16),'FOCUS':dict(parents=2,requests=4),
                'DEV':dict(parents=4,requests=8),'HOLDOUT':dict(parents=1,requests=2)},
        new_RAW=60,new_D0=60,generator_forwards=3000,total_scoring_rows=270,
        new_scoring_rows=180,reused_scoring_rows=90,new_noise=0,new_queries=0,
        metric_fields=fields,protected=frozen,
        chain='single official Bench D0 -> GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE',
        exclusions=['No SHAPE/RELDEF/GEO_ONLY','No new parent/noise evaluation','No source-GT style ranking'],
        uncertainty='Historical DEV and df6 are development/regression evidence, not new independent tests.')
    immutable(MANIFEST,manifest)
    result=dict(status='PASS',schema=VERSION+'_P0_cache_audit',requests=30,parents=15,score_rows=90,
        numeric_fields=177,source_score_rows=cells,original_q_fields_bitwise_json_equal=True,
        model_calls=0,scoring_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
        protected=frozen,files={str(MANIFEST):sha(MANIFEST)},code_sha256=sha(__file__))
    immutable(path,result);return result


def registration_path():
    for name in ('registration.json','training_registration.json'):
        p=OUT/name
        if p.exists():return p
    raise FileNotFoundError('Root registration not frozen')


def cpu_modules():
    if 'torch' in sys.modules:raise RuntimeError('CPU process already imported torch')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    from r2.pilots.seam_abc_v1 import eval_common as C
    return CPU,C


def prepare_scores():
    CPU,C=cpu_modules()
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    panel=prepare_panel();verify(read(MANIFEST)['protected'])
    files=[Path(__file__),Path(CPU.__file__),Path(CPU.H.__file__),Path(CPU.alpha.__file__),Path(CPU.holes.__file__),Path(CPU.M.__file__)]
    code={str(p):sha(p) for p in files};code.update(source_hashes())
    result=dict(status='FROZEN',schema=VERSION+'_score_registration',labels=list(LABELS),stages=list(STAGES),
        eval_manifest_sha256=sha(MANIFEST),registration_sha256=sha(registration_path()),source_hashes=code,
        new_score_rows=180,reused_score_rows=90,metric_fields=read(MANIFEST)['metric_fields'],
        model_calls=0,new_queries=0,source_GT_success=False,all_native_seam_fields='NA_NOT_SEAM_REPRESENTATION',
        chain='Unchanged BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE')
    immutable(OUT/'evaluation/score_registration.json',result);return result


def raw_one(label,q,regsha):
    if label not in NEW_LABELS:raise ValueError('P0 scorer rerun forbidden')
    CPU,C=cpu_modules();rawdir=producer(label,q,'raw')
    dest=score_dir(label,q,'RAW');done=dest/'audit_receipt.json'
    identity=dict(label=label,request_id=q['request_id'],stage='RAW',registration_sha256=regsha,mesh_sha256=sha(rawdir/'raw.npz'))
    if done.exists():
        old=read(done)
        if old['identity']!=identity:raise ValueError('RAW score identity differs')
        verify(old['files']);return old
    if dest.exists():raise RuntimeError('Partial RAW score requires explicit engineering recovery: '+str(dest))
    start=time.perf_counter();CPU.score_current(q,arrays(rawdir/'raw.npz'),dest,'RAW',False)
    result=dict(status='PASS',identity=identity,seconds=time.perf_counter()-start,model_calls=0,new_queries=0,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    immutable(done,result);return result


def post_one(label,q,regsha):
    if label not in NEW_LABELS:raise ValueError('P0 cascade/scoring rerun forbidden')
    CPU,C=cpu_modules();rawdir=producer(label,q,'raw');postdir=producer(label,q,'post')
    cache=OUT/'base_pair_caches'/label/q['request_id'];done=cache/'complete.json'
    if done.exists():
        r=read(done)
        if r['mesh_sha256']!=sha(postdir/'post.npz'):raise ValueError('BASE pair cache differs')
        verify(r['files'])
    else:
        if cache.exists():raise RuntimeError('Partial BASE pair scan requires explicit recovery')
        immutable(cache/'started.json',dict(mesh_sha256=sha(postdir/'post.npz'),model_calls=0))
        pairs=CPU.H.scan(arrays(postdir/'post.npz')['output'],int(q['K']))
        immutable(cache/'pairs.json',pairs)
        immutable(done,dict(status='COMPLETE',mesh_sha256=sha(postdir/'post.npz'),full_pair_scans=1,
            files={str(p):sha(p) for p in cache.iterdir() if p.is_file()}))
    return CPU.process_one(q,label,postdir/'post.npz',postdir/'pre_geo.npz',rawdir/'raw.npz',cache/'pairs.json',
                           OUT/'cpu'/label/q['request_id'],False,regsha)


def cpu_one(job):
    label,q,mode,regsha=job
    r=raw_one(label,q,regsha) if mode=='raw' else post_one(label,q,regsha)
    if 'torch' in sys.modules:raise RuntimeError('Scoring imported torch')
    print('EVALUATED',mode,label,q['request_id'],flush=True);return r


def run_cpu(labels,mode,workers=3):
    if len(set(labels))!=len(labels) or any(x not in NEW_LABELS for x in labels) or not labels:raise ValueError('A/B CPU labels required')
    if mode not in ('raw','post'):raise ValueError(mode)
    reg=prepare_scores();CPU,C=cpu_modules();key=mode+'__'+'__'.join(labels)
    dest=OUT/'evaluation/completions'/(key+'.json')
    if dest.exists():
        r=read(dest);verify(r['files']);return r
    jobs=[(label,q,mode,sha(OUT/'evaluation/score_registration.json')) for label in labels for q in requests()]
    for label,q,_,_ in jobs:
        producer(label,q,'raw')
        if mode=='post':producer(label,q,'post')
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    tick=time.perf_counter();before=C.resources();first=cpu_one(jobs[0])
    count=worker_count(C.resources(),first,workers)
    dispatch=OUT/'evaluation/dispatch'/(key+'__'+str(time.time_ns())+'.json')
    immutable(dispatch,dict(workers=count,threads_per_worker=1,resources_before=before,
        resources_after_first=C.resources(),first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*pool.map(cpu_one,jobs[1:])]
    verify(reg['source_hashes'])
    paths=[score_dir(label,q,'RAW')/'audit_receipt.json' if mode=='raw' else OUT/'cpu'/label/q['request_id']/'complete.json' for label,q,_,_ in jobs]
    result=dict(status='COMPLETE',mode=mode,labels=labels,cells=len(receipts),workers=count,seconds=time.perf_counter()-tick,
        model_calls=0,new_queries=0,dispatch=str(dispatch),files={str(p):sha(p) for p in paths})
    immutable(dest,result);return result


def collect():
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,aggregate,numeric
    done=OUT/'results/completion.json'
    if done.exists():
        r=read(done);verify(r['files']);verify(r['source_rows']);return r
    reg=read(OUT/'evaluation/score_registration.json');verify(reg['source_hashes'])
    metrics=read(MANIFEST)['metric_fields'];rows=[];sources={}
    for label in LABELS:
        for q in requests():
            for stage in STAGES:
                src,score,folder=authenticate_score(label,q,stage)
                row=decorate(src,label,q,stage);row['seam_applicability']=score['seam_to_bulk']['status']
                row.update(eval_group=q['eval_group'],report_group=q['report_group'],
                    result_origin='REUSED_READ_ONLY' if label=='P0' else 'NEW_REGISTERED',
                    source_score_label=storage_label(label))
                for m in metrics:
                    if row[m]!=src[m]:raise ValueError('Original metric altered '+m)
                rows.append(row)
                for name in ('row.json','score.json','complete.json'):
                    p=folder/name;sources[str(p)]=sha(p)
    if len(rows)!=270:raise ValueError('Expected270 fixed score cells')
    conditions=aggregate(rows,('label','stage','report_group','eval_group','parent_id','condition_id'),metrics)
    parents=aggregate(conditions,('label','stage','report_group','eval_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','report_group','eval_group'),metrics)
    pairs=[];by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for ref,cand in (('P0','A'),('P0','B'),('A','B')):
        for stage in STAGES:
            for q in requests():
                a,b=by[ref,stage,q['request_id']],by[cand,stage,q['request_id']]
                row=dict(reference=ref,candidate=cand,stage=stage,**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group','report_group')})
                for m in metrics:
                    av,bv=a[m],b[m];row[m+'__reference']=av;row[m+'__candidate']=bv
                    row[m+'__delta']=bv-av if numeric(av) and numeric(bv) else None
                pairs.append(row)
    deltas=[m+'__delta' for m in metrics]
    pc=aggregate(pairs,('reference','candidate','stage','report_group','eval_group','parent_id','condition_id'),deltas)
    pp=aggregate(pc,('reference','candidate','stage','report_group','eval_group','parent_id'),deltas)
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=summary,
                paired_per_output=pairs,paired_per_condition=pc,paired_per_parent=pp)
    for name,table in tables.items():csvwrite(OUT/'results'/(name+'.csv'),table)
    immutable(OUT/'results/metric_rows.json',rows)
    result=dict(status='COMPLETE',rows=270,new_score_rows=180,reused_score_rows=90,paired_rows=270,
        metric_fields=metrics,source_rows=sources,model_calls=0,scoring_calls=0,new_queries=0,
        collector_sha256=sha(__file__),eval_manifest_sha256=sha(MANIFEST),
        files={str(OUT/'results'/(name+'.csv')):sha(OUT/'results'/(name+'.csv')) for name in tables})
    result['files'][str(OUT/'results/metric_rows.json')]=sha(OUT/'results/metric_rows.json')
    immutable(done,result);return result


def label_spec(label):
    if label not in NEW_LABELS:raise ValueError('P0 is read-only, no model dispatch allowed')
    return dict(mode='PLAIN',arm=ARMS[label],local_step=1500,global_step=21000,subset='full',
        schema='bench_exposure_allocation_generator_v1',
        generator_path=str(OUT/'training'/ARMS[label]/'generator_step1500.pt'))


def freeze_sample(label):
    spec=label_spec(label);path=resolve(spec['generator_path']);sidecar=read(path.with_suffix('.json'))
    h=sha(path)
    if h!=sidecar['sha256']:raise ValueError('Frozen generator export differs')
    inputs=[path,path.with_suffix('.json'),MANIFEST,registration_path()]
    from r2.pilots.bench_seat_coverage_v1 import sample as old_sample
    from r2.pilots.bench_control_transfer_v1 import sample as sampler
    from r2.pilots.seam_slot_c200_v1 import generate
    from r2.pilots.bench_parts_convergence_v1 import data as layout_data
    from r2.pilots.bench_full_seat_curriculum_v1 import eval_runtime as source_runtime
    from meshflow_control.seam import model_adapter,occupancy
    code={str(Path(p)):sha(p) for p in (__file__,old_sample.__file__,sampler.__file__,generate.__file__,
        layout_data.__file__,source_runtime.__file__,model_adapter.__file__,occupancy.__file__)}
    result=dict(status='FROZEN',label=label,spec=spec,outputs=30,new_rollouts=30,reused_rollouts=0,
        solver='Euler50',maximum_head_forwards=0,maximum_generator_forwards=1500,new_noise=0,
        source_labels_read=False,representation='NATIVE_ALL_FREE',
        protected={str(p):sha(p) for p in inputs},code=code)
    immutable(OUT/'evaluation/inference_registrations'/(label+'__RAW.json'),result);return result


def sample_namespace():
    """Build a routed copy of the original functions; executes no model code."""
    from r2.pilots.bench_seat_coverage_v1 import sample as old
    from r2.pilots.bench_full_seat_curriculum_v1.eval_runtime import source_assets
    ns=dict(vars(old));ns.update(OUT=OUT,LABELS=('A','B','P0'),HEAD_BUDGETS=dict.fromkeys(NEW_LABELS,0),
        requests=requests,new_requests=requests,label_spec=label_spec,freeze_label=freeze_sample,
        reuse_spec=lambda q,label:None,raw_completion=raw_completion,raw_receipt=raw_receipt,
        verify_one_raw=verify_one_raw,verify_raw=verify_raw,source_assets=source_assets,
        read=read,resolve=resolve,sha=sha,verify=verify,arrays=arrays)
    for name,obj in vars(old).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(old) and name not in ('freeze_label',):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            fn.__kwdefaults__=obj.__kwdefaults__;ns[name]=fn
    # Methods imported from eval_runtime are replaced above; local freeze stays ours.
    ns['freeze_label']=freeze_sample
    return ns


def sample(label,estimated_peak_mib=2800):
    """Explicit GPU command only; clone original runtime without editing its files."""
    if label not in NEW_LABELS:raise ValueError('P0 regeneration prohibited')
    return sample_namespace()['run'](label,estimated_peak_mib)


def cpu_contract():
    """Authenticate exact C-only layouts and untouched paired-noise routing."""
    import numpy as np
    from r2.pilots.bench_full_seat_curriculum_v1.eval_runtime import source_assets
    from r2.pilots.bench_parts_convergence_v1.data import inference_layout,edge_features
    from r2.pilots.bench_seat_coverage_v1 import sample as old_sample
    ns=sample_namespace();checks=[];old_requests=read(OLD/'eval_manifest.json')['requests']
    for q,oq in zip(requests(),old_requests):
        if {k:v for k,v in q.items() if k!='report_group'}!=oq:raise ValueError('Original request changed')
        core,source=source_assets(q);edges,features=edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
        layout=inference_layout(**core,occupied=np.zeros(len(edges),dtype=bool))
        prior=arrays(OLD/'evaluation/COVERAGE_FINAL/layouts'/q['condition_id']/'layout.npz')
        for key in ('anchor_ids','edge_records','seam_slots','bulk_slots','known_mask'):
            if not np.array_equal(layout[key],prior[key]):raise ValueError('Prior Native inference layout differs '+key)
        z=arrays(q['noise_npz'])['z']
        if z.shape!=(q['N']-q['K'],3,3) or z.dtype!=np.float32:raise ValueError('Noise shape/dtype differs')
        for label in NEW_LABELS:
            if ns['reuse_spec'](q,label) is not None or ns['label_spec'](label)['mode']!='PLAIN':raise ValueError('Routed inference not Native')
        checks.append(dict(request_id=q['request_id'],N=q['N'],K=q['K'],old_q_equal=True,
            original_native_layout_equal=True,source_winding_metadata_validated=True,head_forwards=0,
            noise_sha256=sha(q['noise_npz']),query_sha256=sha(q['query_descriptor']['path'])))
    if ns['run'].__code__ is not old_sample.run.__code__ or ns['save_layout'].__code__ is not old_sample.save_layout.__code__:
        raise ValueError('Original sampler logic was changed')
    if 'torch' in sys.modules:raise RuntimeError('CPU contract imported torch')
    result=dict(status='PASS',requests=30,rows=checks,manifest_sha256=sha(MANIFEST),source_sha256=sha(__file__),
        original_sampler_code_objects_reused=True,all_RAW_before_D0=60,model_calls=0,scoring_calls=0,new_noise=0)
    immutable(OUT/'audits/eval_cpu_contract.json',result);return result


def all_raw_barrier():
    receipts=[]
    for label in NEW_LABELS:
        _,rr=verify_raw(label);receipts.extend(rr)
    if len(receipts)!=60:raise ValueError('All60 new RAW must precede every new D0')
    return receipts


def d0(label,estimated_peak_mib=3500):
    if label not in NEW_LABELS:raise ValueError('P0 D0 regeneration prohibited')
    barrier=all_raw_barrier()
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    from r2.pilots.seam_slot_c200_v1 import postprocess as geometry
    code={str(Path(p)):sha(p) for p in (__file__,legacy.__file__,legacy.OLD.__file__,geometry.__file__)}
    immutable(OUT/'evaluation/inference_registrations'/(label+'__D0.json'),dict(status='FROZEN',label=label,
        outputs=30,new_D0_budget=30,new_GEO_budget=30,code=code,D0_checkpoint=str(legacy.OLD.D0),
        D0_checkpoint_sha256=legacy.OLD.D0_SHA,RAW_barrier={str(p):sha(p) for p in barrier},
        contract='single official Bench D0 + original GEO_C/source winding; no SHAPE or GEO_ONLY'))
    def checked(root,labels,subset):
        if resolve(root)!=OUT or tuple(labels)!=NEW_LABELS or subset!='full':raise ValueError('D0 barrier differs')
        return requests(),all_raw_barrier()
    ns=dict(vars(legacy),verify_raw_barrier=checked)
    fn=types.FunctionType(legacy.run.__code__,ns,legacy.run.__name__,legacy.run.__defaults__,legacy.run.__closure__)
    fn.__kwdefaults__=legacy.run.__kwdefaults__
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    folder=OUT/'evaluation'/label/('D0_resource_observer_'+str(time.time_ns()));folder.mkdir(parents=True)
    observer=GPUObserver('bench_exposure_allocation_single_D0',interval=1.)
    observer.__enter__()
    try:
        result=fn(OUT,label,list(NEW_LABELS),'full',legacy.OLD.D0,estimated_peak_mib)
        if result['new_D0_forwards']>30:raise ValueError('D0 budget exceeded')
        verify(code);return result
    finally:
        observer.__exit__(None,None,None)
        immutable(folder/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples,
            scope='Authorized real D0/GEO only; no extra model probes'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare-panel','prepare-scores','cpu-contract','sample','d0','raw','post','collect'))
    p.add_argument('--model',choices=NEW_LABELS)
    p.add_argument('--models',nargs='+',choices=NEW_LABELS,default=list(NEW_LABELS))
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--estimated-peak-mib',type=float)
    a=p.parse_args()
    if a.action=='prepare-panel':result=prepare_panel()
    elif a.action=='cpu-contract':result=cpu_contract()
    elif a.action=='prepare-scores':result=prepare_scores()
    elif a.action in ('sample','d0'):
        if not a.model:p.error('--model A or B required')
        result=sample(a.model,a.estimated_peak_mib or 2800) if a.action=='sample' else d0(a.model,a.estimated_peak_mib or 3500)
    elif a.action=='collect':result=collect()
    else:result=run_cpu(a.models,a.action,a.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','source_rows','source_hashes','source_score_rows','metric_fields')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
