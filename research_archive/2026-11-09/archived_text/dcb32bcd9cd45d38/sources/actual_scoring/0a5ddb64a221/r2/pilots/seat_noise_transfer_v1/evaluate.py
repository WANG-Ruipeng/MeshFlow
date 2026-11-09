"""Frozen three-model, two-panel inference; no training, no implicit replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(key,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,concurrent.futures,json,multiprocessing,time,traceback
from pathlib import Path
import numpy as np
from . import common as C
LABELS=('EXP67_FROZEN','SMALL_C200','FULL_C200')
ARMS=LABELS
BASE=LABELS[0]
EV=C.OUT/'evaluation'
VERSION='r2_seat_noise_transfer_v1'
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio','base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type')

def verify(files):
    for p,h in files.items():
        if C.sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))

def manifest():
    m=C.read(C.OUT/'inputs/manifest.json');qs=m['requests']
    if len(qs)!=48 or len({q['request_id'] for q in qs})!=48 or len({q['condition_id'] for q in qs})!=8:raise ValueError('Expected 48 requests and eight C')
    groups={p:[q for q in qs if q['panel']==p] for p in ('TRAIN_FAMILIAR','DEV_HELDOUT')}
    if tuple(map(len,groups.values()))!=(32,16):raise ValueError('Panel budget changed')
    for panel,rows in groups.items():
        ids={q['parent_id'] for q in rows}
        if len(ids)!=4 or any(sum(q['parent_id']==p for q in rows)!=(8 if panel=='TRAIN_FAMILIAR' else 4) for p in ids):raise ValueError('Parent/noise budget')
    for q in qs:verify({q['input_path']:q['input_sha256'],q['condition_source']:q['condition_source_sha256']})
    return m

def code_hashes():
    paths=[Path(__file__),Path(C.__file__),C.CODE/'prepare.py',C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py',C.ROOT/'r2/pilots/condition_width_response_v1/evaluate.py',C.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py',C.ROOT/'r2/pilots/user_data_scaling_32_67_v1/training.py']
    paths+=list((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    return {str(p):C.sha(p) for p in sorted(set(paths))}

def freeze():
    m=manifest();previous=C.read(C.ROOT/'experiments/r2_seat_failure_diagnosis_v1/registration.json')
    models={BASE:previous['checkpoint']}
    for arm in LABELS[1:]:models[arm]=C.read(C.ROOT/'experiments/r2_seat_quick_learnability_v1/evaluation/bindings'/(arm+'.json'))
    verify({x['path']:x['sha256'] for x in models.values()});verify({previous['d0_checkpoint']['path']:previous['d0_checkpoint']['sha256']})
    qs=[dict(q,action='NEW') for q in m['requests']]
    for q in qs:old().read_input(q)
    blind={str(p):C.sha(p) for p in (EV/'gallery/blind_plan.json',EV/'blinding/model_key.json')}
    r=dict(status='FROZEN',version=VERSION,labels=LABELS,requests=qs,queries=m['queries'],models=models,d0_checkpoint=previous['d0_checkpoint'],manifest_sha256=C.sha(C.OUT/'inputs/manifest.json'),code_hashes=code_hashes(),blind_files=blind,
      budget=dict(new_rollouts=144,new_generator_forwards=7200,new_D0_forwards=144,new_scored_stages=288,training_updates=0,backwards=0),
      panels=dict(TRAIN_FAMILIAR='4 exact familiar TRAIN C x 8 new noises',DEV_HELDOUT='4 separate conditional-training-held-out parents x 4 new noises; official pretraining exposure UNKNOWN'),
      science='Frozen EXP67 and fixed step200 SMALL/FULL endpoints. Euler50 BF16 forward FP32 integration; all 144 RAW before single official D0 per mesh + GEO_C + source winding. No RBF, no training, no checkpoint selection.',
      primary='Within-C paired RAW C adoption / whole-shape AI_PROXY review and POST local valid indexed C/F edges, kept distinct. CF/FF/I and degradation disclosed. Panel summaries separate; 8/4 noises are repeats, not independent parents.',
      diversity='Descriptive per-C anonymous POST contact sheets, not certified mode-collapse or copy detection. Source style resemblance diagnostic only, no unique-GT quality ranking.',
      resources=dict(user_incremental_GPU_budget_MiB=6144,torch_allocator_cap_MiB=5120,context_allowance_estimate_MiB=1024,minimum_whole_device_free_MiB=2048,GPU_workers=1,CPU_score_workers_max=2),
      failure='No unresolved model-call replay. Completed cells hash-verified and reused. Resource stop only between cells; logged engineering correction can resume without repeating completed calls.')
    C.write(EV/'registration.json',r);C.write(C.OUT/'registration.json',r)
    C.resources(C.OUT/'resources_initial.json');C.event('FROZEN',budget=r['budget'],panels=r['panels'])
    return dict(status='FROZEN',requests=48,models=3,rollouts=144)

def registration():
    r=C.read(EV/'registration.json')
    if r['status']!='FROZEN' or tuple(r['labels'])!=LABELS or r['code_hashes']!=code_hashes():raise ValueError('Frozen implementation changed')
    verify({C.OUT/'inputs/manifest.json':r['manifest_sha256'],**r['blind_files']});return r

def binding(arm):return registration()['models'][arm]
def old():return C.old()

def gpu_adapter(arm):
    G=C.module('r2.pilots.seat_failure_diagnosis_v1._noise_transfer_gpu_'+arm,C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py')
    G.C=C;G.LABEL=arm;G.check_code=registration;G.registration=lambda:dict(registration(),checkpoint=binding(arm))
    G.worker_path=lambda stage,worker,workers,name:EV/'execution'/arm/f'{stage}_{worker}_{workers}_{name}.json'
    def barrier(r):
        records=[]
        for label in LABELS:
            other=gpu_adapter(label)
            records.extend(other.read_raw(q,binding(label)) for q in r['requests'])
        G.LABEL=arm
        if len(records)!=144:raise ValueError('All 144 RAW required')
        return [x for x in records if x['label']==arm]
    G.all_raw=barrier
    return G

def configure_GPU(G,phase,estimate):
    C.package()
    from meshflow_control import runtime
    import torch
    runtime.configure_stable_runtime();torch.set_num_threads(2)
    G.snap(phase+'_before_load',0,1,estimate*1.2)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((5120*1024**2)/total)
    torch.cuda.reset_peak_memory_stats()
    return torch

def memory(torch):
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='PyTorch counters only; full process VRAM unavailable under WDDM, not a 6GiB process proof')

def generate(arm):
    r=registration();b=binding(arm);G=gpu_adapter(arm);pending=[]
    for q in r['requests']:
        p=G.record_path(q,'raw')
        if p.exists():G.read_raw(q,b)
        elif p.parent.exists():raise RuntimeError('Unresolved RAW; no replay: '+q['request_id'])
        else:old().read_input(q);pending.append(q)
    if not pending:return dict(status='COMPLETE',new_rollouts=0)
    torch=configure_GPU(G,'generator_'+arm,2600.);verify({b['path']:b['sha256']})
    from meshflow_control.sampling import sample
    from meshflow_control.checkpoints import state_hash
    tick=time.perf_counter()
    if arm==BASE:
        loader=C.module('_noise_transfer_generator_loader',C.ROOT/'r2/pilots/user_data_scaling_32_67_v1/training.py')
        model,identity=loader.load_generator(C.resolve(b['path']),device='cuda',expected_sha256=b['sha256'])
        if identity['arm']!='USER_EXPANDED_67' or identity['global_step']!=6000 or identity['generator_state_sha256']!='54ba75f915bddaf3486276efec381b7b64d926f17cced6c70d5a18b4b7a9df4e':raise ValueError('Wrong EXP67 state')
    else:
        from meshflow_control.models.native import build_model
        p=torch.load(C.resolve(b['path']),map_location='cpu',weights_only=True,mmap=True)
        if (p['arm'],p['local_step'],p['global_step'])!=(arm,200,6200):raise ValueError('Wrong fixed endpoint')
        if any(k in p for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Inference export required')
        if state_hash(p['generator_state'])!=p['generator_state_sha256']:raise ValueError('State hash')
        model=build_model(p['model_config'],readout_mode='none',trainable=False).eval()
        model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False)
        identity={k:v for k,v in p.items() if k!='generator_state'}
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    C.write(G.worker_path('generator',0,1,'identity'),dict(identity,file_sha256=b['sha256']))
    G.snap('generator_resident_'+arm,0,1)
    class Budget(old().CellBudget):
        def __init__(self,*a,**k):super().__init__(*a,**k);self.velocities=[]
        def call(self,kind,fn,context=None):
            value=super().call(kind,fn,context)
            if kind=='sampling_forward':self.velocities.append(value.detach().float().cpu().numpy()[0])
            return value
    count=0;sampling_seconds=0
    try:
        for q in pending:
            G.snap('generator_cell_'+arm,0,1)
            a=old().read_input(q);G.source(q,a['C']);dest=C.checked(G.record_path(q,'raw').parent);dest.mkdir(parents=True,exist_ok=False)
            context=dict(label=arm,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=b['sha256'])
            budget=Budget(dest/'calls.jsonl',context)
            _,_,audit=sample(model,a['C'],int(q['N']),seed=q['noise_seed'],z=a['z'],budget=budget,out=dest,context=context)
            C.frozen_npz(dest/'velocities.npz',dict(velocity=np.stack(budget.velocities),source_dtype=np.array('torch.bfloat16')))
            rec=dict(q);rec.update(context);rec.update(status='COMPLETE',new_rollouts=1,new_sampling_F=50,generation_status='NEW_EULER50',velocities_path=str(dest/'velocities.npz'),velocities_sha256=C.sha(dest/'velocities.npz'),worker=0,workers=1)
            for field,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):rec[field+'_path']=str(dest/name);rec[field+'_sha256']=C.sha(dest/name)
            _,rec['contracts']=old().verify_output(q,rec,b['sha256']);C.write(dest/'receipt.json',rec);G.read_raw(q,b)
            count+=1;sampling_seconds+=audit['seconds']
            C.write(G.worker_path('generator',0,1,'progress'),dict(completed=48-len(pending)+count,new_rollouts=count,seconds=time.perf_counter()-tick,**memory(torch)),replace=True)
            print(json.dumps(dict(stage='RAW',arm=arm,completed=count,request=q['request_id'],seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=identity['generator_state_sha256']:raise ValueError('Frozen generator changed')
        G.snap('generator_finish_'+arm,0,1)
        result=dict(status='COMPLETE',new_rollouts=count,new_sampling_F=50*count,seconds=time.perf_counter()-tick,sampling_seconds=sampling_seconds,training=0,backward=0,**memory(torch))
        C.write(G.worker_path('generator',0,1,'complete'),result);return result
    except BaseException as exc:
        C.event('GENERATOR_STOPPED',arm=arm,error=repr(exc),traceback=traceback.format_exc(),returned_rollouts=count);raise
    finally:del model;torch.cuda.empty_cache()

def postprocess(arm):
    G=gpu_adapter(arm);r=registration()
    if all(G.record_path(q,'post').exists() for q in r['requests']):
        for q in r['requests']:G.read_post(q,G.read_raw(q,binding(arm)))
        return dict(status='COMPLETE',new_D0_F=0)
    G.all_raw(r)
    torch=configure_GPU(G,'D0_'+arm,3000.)
    original_source,original_partition=G.source,G.partition
    active={'cell_phase':False}
    def partition(reg,worker,workers):
        rows=original_partition(reg,worker,workers);active['cell_phase']=True;return rows
    def guarded_source(q,condition):
        # Runs before dest.mkdir / D0_started / forward reservation in old post.
        if active['cell_phase']:G.snap('D0_cell_boundary_'+arm,0,1)
        return original_source(q,condition)
    G.partition=partition;G.source=guarded_source
    try:return G.postprocess(0,1)
    finally:G.partition=original_partition;G.source=original_source

def collect():
    r=registration();raw=[];post=[]
    for arm in LABELS:
        G=gpu_adapter(arm)
        for q in r['requests']:
            a=G.read_raw(q,binding(arm));p=G.read_post(q,a)
            d=C.read(p['proposal_receipt'])
            if d['model_forwards']!=1 or d['checkpoint_sha256']!=r['d0_checkpoint']['sha256']:raise ValueError('Single registered D0 required')
            raw.append(a);post.append(p)
    for stage,rows in [('raw',raw),('post',post)]:C.write(EV/f'{stage}_index.json',dict(status='COMPLETE',rows=rows))
    return dict(status='COMPLETE',RAW=144,POST=144,model_calls=0)

def indices():
    result={}
    for stage in ('raw','post'):
        rows=C.read(EV/f'{stage}_index.json')['rows']
        if len(rows)!=144 or len({(r['label'],r['request_id']) for r in rows})!=144:raise ValueError('Index incomplete')
        result[stage]={(r['label'],r['request_id']):r for r in rows}
    return result

def cpu_scorer():
    from r2.pilots.condition_width_response_v1 import evaluate as W
    if not hasattr(W,'_transfer_original_hashes'):W._transfer_original_hashes=W.definition_hashes()
    W.E=C;W.VERSION=VERSION;W.META=META
    W.definition_hashes=lambda:{**W._transfer_original_hashes,str(Path(__file__)):C.sha(__file__),str(Path(C.__file__)):C.sha(C.__file__)}
    return W

def score_one(job):
    W=cpu_scorer();q,raw,post,query,stage,rh=job
    query=dict(query,transform_audit=dict(method='FROZEN_BASE_QUERY_UNCHANGED',new_queries=0))
    return W.score_one((q,raw,post,query,stage,rh))

def score(workers=2):
    os.environ['CUDA_VISIBLE_DEVICES']='';r=registration();idx=indices();W=cpu_scorer();done=EV/'score_completion.json'
    if done.exists():
        p=C.read(done)
        if p['definition_hashes']!=W.definition_hashes():raise ValueError('Scorer changed')
        verify({x['path']:x['sha256'] for x in p['receipts']});return p
    queries={x['condition_id']:x for x in r['queries']};rh=C.sha(EV/'registration.json')
    jobs=[(q,idx['raw'][arm,q['request_id']],idx['post'][arm,q['request_id']],queries[q['condition_id']],stage,rh) for arm in LABELS for q in r['requests'] for stage in ('RAW','POST')]
    if len(jobs)!=288:raise ValueError('Score budget changed')
    tick=time.perf_counter();first=score_one(jobs[0]);resource=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (resource.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (resource.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
    actual=2 if workers==2 and parallel else 1
    C.write(EV/('score_dispatch_'+str(time.time_ns())+'.json'),dict(workers=actual,threads=2,resources=resource,first_actual_score=first))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,x in enumerate(pool.map(score_one,jobs[1:]),2):
            receipts.append(x)
            if i%24==0:print(json.dumps(dict(stage='SCORE',completed=i,total=288)),flush=True)
    for x in receipts:x['sha256']=C.sha(x['path'])
    result=dict(status='COMPLETE',receipts=receipts,new_stages=288,seconds=time.perf_counter()-tick,definition_hashes=W.definition_hashes(),model_calls=0)
    C.write(done,result);return result

def aggregate():
    r=registration();done=C.read(EV/'score_completion.json');outputs=[];boundary=[]
    for ref in done['receipts']:
        verify({ref['path']:ref['sha256']});s=C.read(ref['path']);verify(s['files'])
        meta={k:s[k] for k in META};meta.update(label=s['label'],stage=s['stage'])
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None},**s['distance']}
        for row in s['boundary']:
            boundary.append(dict(meta,**row));vals['A_VMR__'+row['setting']]=row['VMR'];vals['PMR__'+row['setting']]=row['perfect']
        vals.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()})
        vals.update(C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],topology_status=s['topology'].get('status'),free_face_denominator=s['free_face_denominator'],C_query_weight_denominator=s['C_query_weight_denominator'],boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],source_C_connectivity_status=s['source_C_connectivity_status'],score_path=ref['path'],score_sha256=ref['sha256'])
        outputs.append(dict(meta,**vals))
    if len(outputs)!=288:raise ValueError('288 scores required')
    W=cpu_scorer();lookup={(x['label'],x['request_id'],x['stage']):x for x in outputs};paired=[]
    metrics=sorted(set(W.ORIGINAL_SCALARS)|{k for x in outputs for k in x if k.startswith('DESC__')})
    for before,after in ((BASE,LABELS[1]),(BASE,LABELS[2]),(LABELS[2],LABELS[1])):
        for q in r['requests']:
            for stage in ('RAW','POST'):
                a=lookup[before,q['request_id'],stage];b=lookup[after,q['request_id'],stage]
                for metric in metrics:
                    av,bv=a.get(metric),b.get(metric);valid=isinstance(av,(int,float,bool)) and isinstance(bv,(int,float,bool)) and np.isfinite(av) and np.isfinite(bv)
                    paired.append(dict(comparison=after+'-minus-'+before,panel=q['panel'],request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],condition_type=q['condition_type'],candidate=q['candidate'],stage=stage,metric=metric,baseline_value=av,candidate_value=bv,delta=float(bv)-float(av) if valid else None,status='PASS' if valid else 'NA'))
    parents=W.average(paired,['comparison','panel','parent_id','condition_id','condition_type','stage','metric'])
    summary=W.average(parents,['comparison','panel','stage','metric'])
    C.csvwrite(EV/'per_output.csv',outputs);C.csvwrite(EV/'AVMR_PMR_all8.csv',boundary);C.csvwrite(EV/'paired_per_output.csv',paired);C.csvwrite(EV/'paired_per_parent.csv',parents);C.csvwrite(EV/'paired_summary.csv',summary)
    C.write(EV/'aggregation.json',dict(status='COMPLETE',outputs=288,new_quality_calls=288,parents=8,separate_panels=True,model_calls=0))
    return dict(status='COMPLETE',outputs=288)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('freeze','generate','post','collect','score','aggregate','check'));p.add_argument('--arm',choices=LABELS);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage in ('generate','post'):
        if not a.arm:p.error('--arm required')
        with gpu_adapter(a.arm).owner('GPU_ALL_MODELS',0,1):result=generate(a.arm) if a.stage=='generate' else postprocess(a.arm)
    elif a.stage=='score':result=score(a.workers)
    elif a.stage=='check':result=dict(status='PASS',requests=len(registration()['requests']),model_calls=0)
    else:result=globals()[a.stage]()
    print(json.dumps(C.clean({k:v for k,v in result.items() if k!='receipts'}),allow_nan=False),flush=True)
if __name__=='__main__':main()