"""Natural-N evaluation: cached old30 P0 plus registered high-N P0/A/B.

Only sample/d0 execute models. The CPU layout contract imports runtime definitions
without building a model. Scoring reuses the unchanged 177-field scorer/cascade.
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
sys.path.insert(0, str(ROOT/'r2/src'))
OUT = ROOT/'experiments/r2_bench_natural_n_recipe_v1'
OLD = ROOT/'experiments/r2_bench_exposure_allocation_v1'
MANIFEST = OUT/'data/eval_manifest.json'
LABELS = ('P0', 'A', 'B')
NEW_LABELS = LABELS
ARMS = {'A':'CONTROL_OLD', 'B':'NATURAL_HIGH'}
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
GROUP_MAP = {'CURRENT_TRAIN':'OLD', 'NEW_TRAIN':'FOCUS', 'DEV':'DEV', 'NEW_PARENT_HOLDOUT':'HOLDOUT'}
VERSION = 'bench_natural_n_recipe_evaluation_v1'


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
    path=resolve(path);obj=json.loads(json.dumps(obj,ensure_ascii=False,allow_nan=False))
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
    if label is not None and label not in LABELS:raise ValueError('Unknown label')
    qs=read(MANIFEST)['requests']
    if not 34<=len(qs)<=38 or len({q['request_id'] for q in qs})!=len(qs):raise ValueError('Frozen panel cardinality differs')
    return qs

def cached_p0(q):return q['report_group'] not in ('HIGH_TRAIN','HIGH_HOLDOUT')
def new_requests(label):return [q for q in requests(label) if label!='P0' or not cached_p0(q)]
def budget():
    n=len(requests());new=3*n-30
    return dict(requests=n,new_RAW=new,new_D0=new,generator_forwards=50*new,new_score_rows=3*new,reused_score_rows=90,total_score_rows=9*n)


def storage_label(label,q=None):return 'B' if label=='P0' and (q is None or cached_p0(q)) else label
def root_for(label,q=None):return OLD if label=='P0' and (q is None or cached_p0(q)) else OUT


def raw_completion(label):return OUT/'evaluation'/label/'raw_complete_full.json'
def raw_receipt(q,label):return root_for(label,q)/'evaluation'/storage_label(label,q)/q['request_id']/'raw/receipt.json'


def verify_one_raw(q,label):
    p=raw_receipt(q,label);r=read(p)
    if (r['status']!='COMPLETE' or r['label']!=storage_label(label,q) or r['request_id']!=q['request_id']
        or r['generator_forwards']!=50 or r['D0_forwards']!=0
        or r['noise_sha256']!=q['noise_sha256'] or r['raw_sha256']!=sha(p.parent/'raw.npz')):
        raise ValueError('RAW producer identity differs: '+str(p))
    verify(r['files']);return p


def verify_raw(label):
    qs=requests(label);done=read(raw_completion(label))
    if done['status']!='COMPLETE' or done['outputs']!=len(qs):raise ValueError('Incomplete RAW wave '+label)
    verify(done['files'])
    return qs,[verify_one_raw(q,label) for q in qs]


def producer(label,q,kind):
    if kind not in ('raw','post'):raise ValueError(kind)
    folder=root_for(label,q)/'evaluation'/storage_label(label,q)/q['request_id']/kind
    if kind=='raw':verify_one_raw(q,label);return folder
    r=read(folder/'receipt.json')
    if (r['status']!='COMPLETE' or r['label']!=storage_label(label,q) or r['request_id']!=q['request_id']
        or r['D0_forwards']!=1 or r['GEO_calls']!=1 or r['post_sha256']!=sha(folder/'post.npz')):
        raise ValueError('POST producer identity differs: '+str(folder))
    verify(r['files']);return folder


def score_dir(label,q,stage):
    root=root_for(label,q);label=storage_label(label,q)
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
    """Freeze candidate inputs and authenticate old30 B outputs. No scorer/model."""
    done=OUT/'audits/P0_cache_audit.json'
    if done.exists():
        r=read(done);verify(r['files']);verify(r['protected']);return r
    cp=OUT/'data/eval_candidates.json';op=OLD/'data/eval_manifest.json'
    proposal=read(cp);old=read(op)['requests'];qs=copy.deepcopy(proposal['requests'])
    if len(old)!=30 or qs[:30]!=old:raise ValueError('Old30 input fields/order changed')
    if not 34<=len(qs)<=38 or len({q['request_id'] for q in qs})!=len(qs):raise ValueError('Panel cardinality')
    high=qs[30:];hp=collections.Counter(q['report_group'] for q in high)
    if hp['HIGH_TRAIN']!=4 or hp['HIGH_HOLDOUT'] not in (0,2,4) or sum(hp.values())!=hp['HIGH_TRAIN']+hp['HIGH_HOLDOUT']:raise ValueError('High panel groups')
    byparent=collections.defaultdict(list)
    for q in high:
        if not 257<=q['N']<=799:raise ValueError('High N contract')
        byparent[q['parent_id']].append(q)
    if len(byparent)!=len(high)//2 or any(sorted(q['noise_index'] for q in rr)!=[0,1] or len({q['condition_id'] for q in rr})!=1 for rr in byparent.values()):raise ValueError('High parent/noise contract')
    fields=read(OLD/'results/completion.json')['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Metric dictionary')
    frozen={str(p):sha(p) for p in (cp,op,OLD/'results/completion.json')};cells=[]
    for q in qs:
        if not 128<=q['N']<=799 or not 2<=q['K']<q['N']:raise ValueError('Native N/K')
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            p=resolve(q[key]);h=sha(p)
            if h!=q[hkey]:raise ValueError('Input SHA '+key)
            frozen[str(p)]=h
        p=resolve(q['query_descriptor']['path'])
        if sha(p)!=q['query_descriptor']['sha256']:raise ValueError('Query SHA')
        frozen[str(p)]=sha(p)
        if 'gallery_bounds' not in q or 'gallery_depth_bounds' not in q:raise ValueError('Fixed frame missing')
        if cached_p0(q):
            for kind in ('raw','post'):
                folder=producer('P0',q,kind)
                for p in folder.iterdir():
                    if p.is_file() and p.suffix in ('.npz','.json','.jsonl'):frozen[str(p)]=sha(p)
            for stage in STAGES:
                row,score,folder=authenticate_score('P0',q,stage)
                if any(m not in row for m in fields):raise ValueError('Old metric missing')
                cells.append(dict(request_id=q['request_id'],stage=stage,source=str(folder)))
                for name in ('row.json','score.json','complete.json'):
                    p=folder/name;frozen[str(p)]=sha(p)
    n=len(qs);new=3*n-30
    manifest=dict(schema=VERSION+'_panel',status='FROZEN',labels=list(LABELS),requests=qs,
        requests_per_label=dict.fromkeys(LABELS,n),new_requests_per_label={'P0':len(high),'A':n,'B':n},
        stages=list(STAGES),source_manifest=str(op),source_manifest_sha256=sha(op),
        p0=dict(root=str(OLD),label='B',old30='READ_ONLY_NO_REGENERATION',high='NEW_FROZEN_B_INFERENCE'),
        new_RAW=new,new_D0=new,generator_forwards=50*new,total_scoring_rows=9*n,new_scoring_rows=3*new,reused_scoring_rows=90,
        metric_fields=fields,protected=frozen,blocked_rows=proposal.get('blocked_rows',[]),new_queries_constructed=len(high)//2,
        uncertainty='OldDEV/HOLDOUT are reused development evidence. HIGH groups remain separate; no unique-source-style success.',
        chain='single official Bench D0 -> GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE')
    immutable(MANIFEST,manifest)
    result=dict(status='PASS',requests=n,old_reused_requests=30,new_high_requests=len(high),score_rows_reused=90,numeric_fields=177,
        source_score_rows=cells,protected=frozen,files={str(MANIFEST):sha(MANIFEST)},model_calls=0,scoring_calls=0,GPU_calls=0,code_sha256=sha(__file__))
    immutable(done,result);return result

def prepare_manifest():return prepare_panel()


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
        new_score_rows=budget()['new_score_rows'],reused_score_rows=90,metric_fields=read(MANIFEST)['metric_fields'],
        model_calls=0,new_queries=0,source_GT_success=False,all_native_seam_fields='NA_NOT_SEAM_REPRESENTATION',
        chain='Unchanged BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE')
    immutable(OUT/'evaluation/score_registration.json',result);return result


def raw_one(label,q,regsha):
    if label not in LABELS or (label=='P0' and cached_p0(q)):raise ValueError('Cached P0 scorer rerun forbidden')
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
    if label not in LABELS or (label=='P0' and cached_p0(q)):raise ValueError('Cached P0 cascade rerun forbidden')
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
    jobs=[(label,q,mode,sha(OUT/'evaluation/score_registration.json')) for label in labels for q in new_requests(label)]
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
                    result_origin='REUSED_READ_ONLY' if label=='P0' and cached_p0(q) else 'NEW_REGISTERED',
                    source_score_label=storage_label(label,q))
                for m in metrics:
                    if row[m]!=src[m]:raise ValueError('Original metric altered '+m)
                rows.append(row)
                for name in ('row.json','score.json','complete.json'):
                    p=folder/name;sources[str(p)]=sha(p)
    if len(rows)!=budget()['total_score_rows']:raise ValueError('Fixed score universe differs')
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
    result=dict(status='COMPLETE',rows=budget()['total_score_rows'],new_score_rows=budget()['new_score_rows'],reused_score_rows=90,paired_rows=budget()['total_score_rows'],
        metric_fields=metrics,source_rows=sources,model_calls=0,scoring_calls=0,new_queries=0,
        collector_sha256=sha(__file__),eval_manifest_sha256=sha(MANIFEST),
        files={str(OUT/'results'/(name+'.csv')):sha(OUT/'results'/(name+'.csv')) for name in tables})
    result['files'][str(OUT/'results/metric_rows.json')]=sha(OUT/'results/metric_rows.json')
    immutable(done,result);return result


def label_spec(label):
    if label not in LABELS:raise ValueError('Unknown label')
    if label=='P0':return dict(mode='PLAIN',arm='B_75_25',local_step=1500,global_step=21000,subset='full',
        schema='bench_exposure_allocation_generator_v1',generator_path=str(OLD/'training/B_75_25/generator_step1500.pt'))
    return dict(mode='PLAIN',arm=ARMS[label],local_step=1500,global_step=22500,subset='full',
        schema='bench_natural_n_recipe_generator_v1',generator_path=str(OUT/'training'/ARMS[label]/'generator_step1500.pt'))


def inference_code_hashes(code, kind):
    """Carry the complete inherited model/runtime identities into each wave."""
    inherited=read(OLD/'evaluation/inference_registrations'/('B__'+kind+'.json'))['code']
    training=read(registration_path())['training_code_hashes']
    verify(inherited);verify(training)
    result={**inherited,**training,**code}
    if kind=='D0':
        directory=ROOT/'r2/src/meshflow_control/denoiser_models'
        result[str(ROOT/'r2/src/meshflow_control/denoise_model.py')]=sha(ROOT/'r2/src/meshflow_control/denoise_model.py')
        source=read(directory/'source_manifest.json')
        result[str(directory/'source_manifest.json')]=sha(directory/'source_manifest.json')
        for name,digest in source['vendored_file_sha256'].items():
            if sha(directory/name)!=digest:raise ValueError('Official D0 source changed')
            result[str(directory/name)]=digest
    return result


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
    from . import runtime as n_runtime
    code={str(Path(p)):sha(p) for p in (__file__,n_runtime.__file__,old_sample.__file__,sampler.__file__,generate.__file__,
        layout_data.__file__,source_runtime.__file__,model_adapter.__file__,occupancy.__file__)}
    code=inference_code_hashes(code,'RAW')
    result=dict(status='FROZEN',label=label,spec=spec,outputs=len(requests(label)),new_rollouts=len(new_requests(label)),reused_rollouts=len(requests(label))-len(new_requests(label)),
        solver='Euler50',maximum_head_forwards=0,maximum_generator_forwards=50*len(new_requests(label)),new_noise=0,
        source_labels_read=False,representation='NATIVE_ALL_FREE',
        protected={str(p):sha(p) for p in inputs},code=code)
    immutable(OUT/'evaluation/inference_registrations'/(label+'__RAW.json'),result);return result


def reuse_spec(q,label):return dict(root=str(OLD),label='B') if label=='P0' and cached_p0(q) else None
def load_head(label,payload):return None,dict(kind='NONE_NATIVE_ALL_FREE',parameters=0,source_labels_read=False)
def source_assets(q):
    from r2.pilots.bench_full_seat_curriculum_v1.eval_runtime import source_assets as original
    return original(q)

def save_layout(q,label,head,head_identity):
    import numpy as np
    from .runtime import inference_layout,edge_features
    from r2.pilots.bench_seat_coverage_v1.sample import P
    core,source=source_assets(q);dest=OUT/'evaluation'/label/'layouts'/q['condition_id']
    edges,features=edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
    lay=inference_layout(**core,occupied=np.zeros(len(edges),bool));logits=np.full(len(edges),-1,np.float32)
    if (dest/'receipt.json').exists():
        r=read(dest/'receipt.json');verify(r['files'])
        if r['head']!=head_identity or r['input_sha256']!=q['inference_npz_sha256']:raise ValueError('Layout identity differs')
        prior=arrays(dest/'layout.npz')
        if any(not np.array_equal(prior[k],lay[k]) for k in ('anchor_ids','edge_records','seam_slots','bulk_slots','known_mask')):raise ValueError('Layout changed')
        return lay,logits,source,dest,0
    if dest.exists():raise RuntimeError('Partial layout needs review')
    dest.mkdir(parents=True);P.array_file(dest/'layout.npz',{**lay,'occupancy_logits':logits});P.array_file(dest/'source_condition.npz',source)
    immutable(dest/'receipt.json',dict(status='COMPLETE',condition_id=q['condition_id'],label=label,
        occupancy_forward_calls=0,source_labels_read=False,input_sha256=q['inference_npz_sha256'],head=head_identity,
        active=0,boundary_edges=len(edges),threshold_logit=None,closed_C_skipped=len(edges)==0,
        occupancy_logits_applicability='NA_NATIVE_NEGATIVE_PLACEHOLDER_NO_HEAD',files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
    return lay,logits,source,dest,0


def sample(label,estimated_peak_mib=3500):
    if label not in LABELS:raise ValueError('Unknown label')
    folder=OUT/'evaluation'/label
    if not raw_completion(label).exists() and list(folder.glob('sampling_attempt_*')):raise RuntimeError('Prior attempt needs review; no replay')
    return sample_run(label,estimated_peak_mib)


def sample_run(label, estimated_peak_mib=2800):
    spec, qs = label_spec(label), requests(label)
    out = OUT / "evaluation" / label
    if raw_completion(label).exists():
        verify_raw(label)
        return read(raw_completion(label))
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from .runtime import build_model
    from r2.pilots.bench_seat_coverage_v1.sample import P
    from meshflow_control.geo_io import write_indexed_ply
    from r2.pilots.bench_control_transfer_v1 import train as T, sample as S
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays, dispatch_guard
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    reg = freeze_sample(label)
    gen, gen_hash = resolve(spec["generator_path"]), sha(spec["generator_path"])
    pending = []
    for q in qs:
        if reuse_spec(q, label):
            verify_one_raw(q,label)
            continue
        dest = out / q["request_id"] / "raw"
        if (dest / "receipt.json").exists():
            old = read(dest / "receipt.json")
            verify(old["files"])
            if old["generator"]["sha256"] != gen_hash or old["noise_sha256"] != q["noise_sha256"]:
                raise ValueError("Completed RAW belongs to another generator/noise")
        elif dest.exists():
            raise RuntimeError("Unresolved RAW attempt, no silent replay: " + str(dest))
        else:
            pending.append(q)
    attempt = out / ("sampling_attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    snapshot = T.resources()
    T.write(attempt / "resources_before.json", snapshot)
    dispatch_guard(snapshot, estimated_peak_mib)
    T.write(attempt / "dispatch.json", dict(workers=1, controller_owns_concurrency=True,
        estimated_process_peak_mib=estimated_peak_mib,
        headroom="max(2GiB,10%total) plus 1.2x observed comparable process peak",
        serial_reason="One model/head owns this label's deterministic per-C layout cache; independent labels may dispatch separately when measured resources permit."))
    model = head = None
    completed = head_calls = 0
    tick = time.perf_counter()
    observer = GPUObserver("bench_natural_n_registered_new_requests", interval=1.)
    observer.__enter__()
    try:
        p = torch.load(gen, map_location="cpu", weights_only=True, mmap=True)
        if (p.get("schema") != spec['schema'] or p["arm"] != spec["arm"] or int(p["local_step"]) != spec["local_step"]
                or int(p["global_step"]) != spec["global_step"] or p.get("EMA", False)):
            raise ValueError("Wrong frozen live generator endpoint")
        if state_hash(p["generator_state"]) != p["generator_state_sha256"]:
            raise ValueError("Generator tensor hash differs")
        runtime.configure_stable_runtime()
        model = build_model(copy.deepcopy(p["model_config"]), readout_mode="none", trainable=False)
        model.load_state_dict(p["generator_state"], strict=True)
        model.to("cuda").set_trainable(False).eval()
        head, head_identity = load_head(label, p)
        identity = dict(path=str(gen), sha256=gen_hash, state_sha256=p["generator_state_sha256"],
                        local_step=p["local_step"], global_step=p["global_step"], EMA=False)
        T.write(attempt / "model_identity.json", dict(generator=identity, head=head_identity))
        T.write(attempt / "resources_loaded.json", T.resources())
        layouts = {}
        for q in pending:
            cid = q["condition_id"]
            if cid not in layouts:
                lay, logits, source, lp, calls = save_layout(q, label, head, head_identity)
                head_calls += calls
                layouts[cid] = lay, logits, source, lp
            lay, logits, source, lp = layouts[cid]
            dest = out / q["request_id"] / "raw"
            dest.mkdir(parents=True)
            T.write(dest / "attempt.json", dict(label=label, request=q, generator=identity, forward_budget=50))
            raw, trajectory, audit = S.rollout(model, lay, arrays(q["noise_npz"])["z"], spec["mode"], dest / "calls.jsonl")
            mesh = raw_arrays(raw, lay, logits, source)
            P.array_file(dest / "raw.npz", mesh)
            P.array_file(dest / "trajectory.npz", dict(state=trajectory))
            P.array_file(dest / "source_condition.npz", source)
            write_indexed_ply(dest / "raw.ply", mesh["vertices"], mesh["faces"])
            T.write(dest / "audit.json", audit)
            T.write(dest / "receipt.json", dict(status="COMPLETE", execution="NEW", label=label,
                arm=label, request_id=q["request_id"], condition_id=cid, generator=identity,
                checkpoint_sha256=gen_hash, generator_forwards=50, new_generator_forwards=50,
                D0_forwards=0, new_D0_forwards=0, raw_sha256=sha(dest / "raw.npz"),
                noise_sha256=q["noise_sha256"], source_labels_read=False,
                head_forwards=0, head_accounting="Counted once in unique-C layout receipt", layout_receipt=str(lp / "receipt.json"), representation=spec["mode"],
                audit=audit, files={str(f): sha(f) for f in dest.iterdir() if f.is_file()}))
            completed += 1
            if completed <= 2 or completed % 8 == 0:
                T.write(attempt / f"resources_output{completed}.json", T.resources())
                T.write(attempt / f"observed_peak_output{completed}.json", dict(
                    summary=observer.summary(), samples=observer.samples,
                    scoped_rollout_allocator=audit, extra_model_calls=0))
            print(json.dumps(dict(stage="RAW", label=label, completed=completed, pending=len(pending),
                                 request_id=q["request_id"], seconds=audit["seconds"])), flush=True)
        if state_hash(model.state_dict()) != p["generator_state_sha256"]:
            raise ValueError("Frozen generator mutated")
        if head is not None and state_hash(head.state_dict()) != head_identity["state_sha256"]:
            raise ValueError("Frozen occupancy head mutated")
        all_layouts = {q["condition_id"] for q in new_requests(label)}
        total_head = sum(read(out / "layouts" / cid / "receipt.json")["occupancy_forward_calls"] for cid in all_layouts)
        if total_head != 0:
            raise ValueError("Total registered head calls differ")
        verify(reg["protected"])
        verify(reg["code"])
        result = dict(status="COMPLETE", label=label, subset=spec["subset"], outputs=len(qs),
            new_rollouts_this_attempt=completed, new_forwards_this_attempt=50*completed,
            new_head_forwards_this_attempt=head_calls, total_new_rollouts=len(new_requests(label)),
            total_new_generator_forwards=50*len(new_requests(label)), total_new_head_forwards=total_head,
            reused_rollouts=len(qs)-len(new_requests(label)), checkpoint_sha256=gen_hash,
            seconds=time.perf_counter()-tick,
            files={str(raw_receipt(q,label)):sha(raw_receipt(q,label)) for q in qs})
        P.immutable(raw_completion(label), result)
        return result
    except BaseException as exc:
        T.write(attempt / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc(),
            completed_rollouts=completed, returned_head_calls=head_calls, no_automatic_retry=True))
        raise
    finally:
        observer.__exit__(None, None, None)
        T.write(attempt / "resource_observation.json", dict(summary=observer.summary(), samples=observer.samples))
        T.write(attempt / "resources_finished.json", T.resources())
        del model, head
        torch.cuda.empty_cache()




def cpu_contract():
    import numpy as np
    from .runtime import inference_layout,edge_features
    prepare_panel();checks=[]
    for q in requests():
        core,source=source_assets(q);edges,features=edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
        layout=inference_layout(**core,occupied=np.zeros(len(edges),bool))
        if len(layout['seam_slots']) or len(layout['bulk_slots'])!=q['N']-q['K']:raise ValueError('Not Native all-free')
        if cached_p0(q):
            prior=arrays(OLD/'evaluation/B/layouts'/q['condition_id']/'layout.npz')
            for key in ('anchor_ids','edge_records','seam_slots','bulk_slots','known_mask'):
                if not np.array_equal(layout[key],prior[key]):raise ValueError('Old layout changed '+key)
        z=arrays(q['noise_npz'])['z']
        if z.shape!=(q['N']-q['K'],3,3) or z.dtype!=np.float32:raise ValueError('Noise shape/dtype')
        checks.append(dict(request_id=q['request_id'],N=q['N'],K=q['K'],reused_old=cached_p0(q),head_forwards=0,source_winding_metadata_validated=True,noise_sha256=q['noise_sha256']))
    result=dict(status='PASS',requests=len(checks),rows=checks,manifest_sha256=sha(MANIFEST),new_RAW_before_D0=budget()['new_RAW'],model_calls=0,scoring_calls=0,new_noise=0)
    immutable(OUT/'audits/eval_cpu_contract.json',result);return result


def all_raw_barrier():
    receipts=[]
    for label in LABELS:
        verify_raw(label);receipts.extend(raw_receipt(q,label) for q in new_requests(label))
    if len(receipts)!=budget()['new_RAW']:raise ValueError('All new RAW must precede D0')
    return receipts


def d0(label,estimated_peak_mib=3500):
    if label not in LABELS:raise ValueError('Unknown label')
    barrier=all_raw_barrier()
    destination=OUT/'evaluation'/label
    if (destination/'post_complete_full.json').exists():
        done=read(destination/'post_complete_full.json');verify(done['files']);return done
    if list(destination.glob('postprocess_attempt_*')):raise RuntimeError('Prior D0 attempt needs review; no replay')
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    from r2.pilots.seam_slot_c200_v1 import postprocess as geometry
    code={str(Path(p)):sha(p) for p in (__file__,legacy.__file__,legacy.OLD.__file__,geometry.__file__)}
    code=inference_code_hashes(code,'D0')
    immutable(OUT/'evaluation/inference_registrations'/(label+'__D0.json'),dict(status='FROZEN',label=label,
        outputs=len(new_requests(label)),new_D0_budget=len(new_requests(label)),new_GEO_budget=len(new_requests(label)),code=code,D0_checkpoint=str(legacy.OLD.D0),
        D0_checkpoint_sha256=legacy.OLD.D0_SHA,RAW_barrier={str(p):sha(p) for p in barrier},
        contract='single official Bench D0 + original GEO_C/source winding; no SHAPE or GEO_ONLY'))
    def checked(root,labels,subset):
        if resolve(root)!=OUT or tuple(labels)!=NEW_LABELS or subset!='full':raise ValueError('D0 barrier differs')
        return new_requests(label),all_raw_barrier()
    ns=dict(vars(legacy),verify_raw_barrier=checked)
    fn=types.FunctionType(legacy.run.__code__,ns,legacy.run.__name__,legacy.run.__defaults__,legacy.run.__closure__)
    fn.__kwdefaults__=legacy.run.__kwdefaults__
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    folder=OUT/'evaluation'/label/('D0_resource_observer_'+str(time.time_ns()));folder.mkdir(parents=True)
    observer=GPUObserver('bench_natural_n_single_D0',interval=1.)
    observer.__enter__()
    try:
        result=fn(OUT,label,list(NEW_LABELS),'full',legacy.OLD.D0,estimated_peak_mib)
        if result['new_D0_forwards']!=len(new_requests(label)) or result['new_GEO_calls']!=len(new_requests(label)):raise ValueError('D0/GEO budget differs')
        verify(code);return result
    finally:
        observer.__exit__(None,None,None)
        immutable(folder/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples,
            scope='Authorized real D0/GEO only; no extra model probes'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare-panel','prepare-manifest','prepare-scores','cpu-contract','sample','d0','raw','post','collect'))
    p.add_argument('--model',choices=NEW_LABELS)
    p.add_argument('--models',nargs='+',choices=NEW_LABELS,default=list(NEW_LABELS))
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--estimated-peak-mib',type=float)
    a=p.parse_args()
    if a.action in ('prepare-panel','prepare-manifest'):result=prepare_panel()
    elif a.action=='cpu-contract':result=cpu_contract()
    elif a.action=='prepare-scores':result=prepare_scores()
    elif a.action in ('sample','d0'):
        if not a.model:p.error('--model P0, A or B required')
        result=sample(a.model,a.estimated_peak_mib or 2800) if a.action=='sample' else d0(a.model,a.estimated_peak_mib or 3500)
    elif a.action=='collect':result=collect()
    else:result=run_cpu(a.models,a.action,a.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','rows','source_rows','source_hashes','source_score_rows','metric_fields')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
