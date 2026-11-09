"""C-reference experiment evaluation: two cached controls and two local-coordinate states."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(key,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,concurrent.futures,json,multiprocessing,time,traceback
from pathlib import Path
import numpy as np
from . import common as C
C.package()  # Resolve maintained package before CPU RAW validation imports model.
LABELS=('EXP67_FROZEN','SMALL_C200','CREF_C0','CREF_C200')
NEW_LABELS=LABELS[2:]
PRIOR=C.ROOT/'experiments/r2_seat_noise_transfer_v1'
COMPARISONS=((LABELS[0],LABELS[2]),(LABELS[1],LABELS[3]),(LABELS[0],LABELS[3]),(LABELS[2],LABELS[3]))
ARMS=LABELS
BASE=LABELS[0]
EV=C.OUT/'evaluation'
VERSION='r2_c_reference_frame_v1'
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
    paths=list(C.CODE.glob('*.py'))+list((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    paths += [C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py',C.ROOT/'r2/pilots/condition_width_response_v1/evaluate.py',C.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py']
    # Only scientific implementations are frozen here; report/packaging remain addable.
    paths=[p for p in paths if p.parent!=C.CODE or p.name in ('common.py','model.py','prepare.py','train.py','evaluate.py','render.py')]
    return {str(p):C.sha(p) for p in sorted(set(paths))}

def freeze():
    m=manifest();prior=C.read(PRIOR/'evaluation/registration.json')
    if m!=C.read(PRIOR/'inputs/manifest.json'):raise ValueError('Evaluation requests changed')
    models={k:prior['models'][k] for k in LABELS[:2]}
    models['CREF_C0']=dict(models['EXP67_FROZEN'])
    models['CREF_C200']=dict(path=str(C.OUT/'inference/CREF_C200_generator_step200.pt'),status='EXPECTED_FIXED_STEP200')
    refs=[PRIOR/'evaluation'/name for name in ('registration.json','raw_index.json','post_index.json','per_output.csv','score_completion.json')]
    refs += [C.ROOT/'r2/pilots/seat_noise_transfer_v1'/name for name in ('evaluate.py','common.py')]
    refs += [PRIOR/'evaluation/gallery/AI_PROXY_review_unblinded.csv']
    r=dict(status='FROZEN',version=VERSION,labels=LABELS,requests=[dict(q,action='NEW') for q in m['requests']],queries=m['queries'],models=models,
      d0_checkpoint=prior['d0_checkpoint'],manifest_sha256=C.sha(C.OUT/'inputs/manifest.json'),code_hashes=code_hashes(),
      blind_files={str(p):C.sha(p) for p in (EV/'gallery/blind_plan.json',EV/'blinding/model_key.json')},protected={str(p):C.sha(p) for p in refs},
      budget=dict(new_rollouts=96,new_generator_forwards=4800,new_D0_forwards=96,new_scored_stages=192,reused_scored_stages=192,training_updates=200,training_forwards=1600,backwards=1600),
      primary='RAW C adoption NO->YES vs SMALL_C200 across multiple C and noises; disclose YES->NO, UNCERTAIN, completeness and CF/FF/I. New local0 must separate wrapper effect from adaptation.',
      science='Local backbone plus original-world C Geo; area center only from augmented C; independent Gaussian with original arrays; preOT subtract a/2 and recompute; no demeaning/rotation/scale/new parameters/new loss. Local200 starts same EXP67 generator+AdamW+RNG and copies original SMALL plan.',
      inference='48 frozen requests per new state; Euler50 local FP32 accumulation BF16 forward. Restore world F by +a and C by original bytes; ALL RAW before single D0 + GEO_C + source orientation. No RBF.',
      comparisons=COMPARISONS,resources=dict(user_budget_MiB=6144,allocator_cap_MiB=5120,headroom_MiB=2048,GPU_workers=1,CPU_workers_max=2),
      failure='No unresolved scientific-call replay; hash-verified completed cells reused. Engineering corrections logged; old artifacts readonly.')
    C.write(EV/'registration.json',r);C.resources(C.OUT/'resources_evaluation_freeze.json');C.event('EVALUATION_FROZEN',budget=r['budget'])
    return dict(status='FROZEN',requests=48,new_rollouts=96)

def registration():
    r=C.read(EV/'registration.json')
    expected=r['code_hashes']
    amendment=C.OUT/'engineering/001_import_path/amendment.json'
    if amendment.exists():
        a=C.read(amendment)
        if a['previous_code_hashes']!=expected or C.sha(a['preserved_original_path'])!=a['previous_evaluate_sha256']:raise ValueError('Amendment parent differs')
        expected=a['current_code_hashes']
    if r['status']!='FROZEN' or tuple(r['labels'])!=LABELS or expected!=code_hashes():raise ValueError('Frozen implementation changed')
    verify({C.OUT/'inputs/manifest.json':r['manifest_sha256'],**r['blind_files']});return r

def binding(arm):
    b=registration()['models'][arm]
    if arm=='CREF_C200':
        path=EV/'bindings'/f'{arm}.json'
        if path.exists():return C.read(path)
        from .train import load_binding
        b=load_binding()
        C.write(path,b)
    return b

def old():return C.old()

def gpu_adapter(arm):
    G=C.module('r2.pilots.seat_failure_diagnosis_v1._cref_gpu_'+arm,C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py')
    G.C=C;G.LABEL=arm;G.check_code=registration;G.registration=lambda:dict(registration(),checkpoint=binding(arm))
    G.worker_path=lambda stage,worker,workers,name:EV/'execution'/arm/f'{stage}_{worker}_{workers}_{name}.json'
    G.read_raw=lambda q,b:read_raw(q,arm,b)
    def barrier(r):
        for label in NEW_LABELS:
            for q in r['requests']:read_raw(q,label,binding(label))
        return [read_raw(q,arm,binding(arm)) for q in r['requests']]
    G.all_raw=barrier
    return G

def verify_local_output(q,rec,b):
    from .model import area_center
    verify({rec[f+'_path']:rec[f+'_sha256'] for f in ('raw','ply','result','velocities')})
    if rec['checkpoint_sha256']!=b['sha256']:raise ValueError('Binding mismatch')
    inp=old().read_input(q);a=C.arrays(rec['raw_path']);n,k=int(q['N']),int(q['K'])
    raw=a['raw'].reshape(n,3,3);world=a['trajectory'].reshape(51,n,3,3)
    local=a['local_trajectory'].reshape(51,n,3,3);center=a['center'].reshape(3)
    if not C.same(center,area_center(inp['C'])):raise ValueError('Center not computed only from C')
    v=C.arrays(rec['velocities_path'])['velocity'].reshape(50,n,3,3)
    if v.dtype!=np.float32 or np.any(v.view(np.uint32)&0xffff):raise ValueError('Expected lossless BF16 velocity storage')
    facts=dict(C_all51=all(C.same(x[:k],inp['C']) for x in world),local_C_all51=all(C.same(x[:k],inp['C']-center) for x in local),
      frozen_z=C.same(a['z'],inp['z']),local_start_z=C.same(local[0,k:].reshape(1,n-k,9),inp['z'][:,k:]),
      local_euler=all(C.same(local[t+1,k:],local[t,k:]+v[t,k:]*np.float32(1/50)) for t in range(50)),
      world_translation=all(C.same(w[k:],l[k:]+center) for w,l in zip(world,local)),
      terminal=C.same(raw,world[-1]),C_saved=C.same(a['C'].reshape(k,3,3),inp['C']),finite=bool(np.isfinite(world).all() and np.isfinite(local).all()),FP32=raw.dtype==world.dtype==local.dtype==np.float32)
    C.package();from meshflow_control.mesh_io import read_ply
    ply,_=read_ply(C.resolve(rec['ply_path']));facts['ply_exact']=C.same(ply,raw)
    run=C.read(rec['result_path']);run=run.get('sampling',run)
    facts['runtime']=run['nfe']==50 and run['readout_mode']=='none' and all(x in run['dtype'].upper() for x in ('BF16','FP32'))
    if not all(facts.values()):raise ValueError('Local contract failed '+str(facts))
    return facts

def read_raw(q,arm,b):
    G=gpu_adapter(arm);rec=C.read(G.record_path(q,'raw'))
    if rec['request_id']!=q['request_id'] or rec['input_sha256']!=q['input_sha256']:raise ValueError('Request identity')
    verify_local_output(q,rec,b);G.verify_ledger(G.record_path(q,'raw').parent/'calls.jsonl');G.source(q,C.arrays(rec['raw_path'])['C'])
    return rec

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
    if arm not in NEW_LABELS:raise ValueError('Cached controls may not be regenerated')
    r=registration();b=binding(arm);G=gpu_adapter(arm);pending=[]
    for q in r['requests']:
        p=G.record_path(q,'raw')
        if p.exists():G.read_raw(q,b)
        elif p.parent.exists():raise RuntimeError('Unresolved RAW; no replay: '+q['request_id'])
        else:old().read_input(q);pending.append(q)
    if not pending:return dict(status='COMPLETE',new_rollouts=0)
    torch=configure_GPU(G,'generator_'+arm,2600.);verify({b['path']:b['sha256']})
    from .model import sample, build_model
    from meshflow_control.checkpoints import state_hash
    tick=time.perf_counter()
    p=torch.load(C.resolve(b['path']),map_location='cpu',weights_only=True,mmap=True)
    if state_hash(p['generator_state'])!=p['generator_state_sha256']:raise ValueError('State hash')
    if arm=='CREF_C0' and p['generator_state_sha256']!='54ba75f915bddaf3486276efec381b7b64d926f17cced6c70d5a18b4b7a9df4e':raise ValueError('Wrong EXP67 state')
    if arm=='CREF_C200' and (p['arm'],p['local_step'],p['global_step'])!=('CREF_C200',200,6200):raise ValueError('Wrong fixed step200')
    model=build_model(p['model_config'],trainable=False).eval()
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
            rec['contracts']=verify_local_output(q,rec,b);C.write(dest/'receipt.json',rec);G.read_raw(q,b)
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
    if arm not in NEW_LABELS:raise ValueError('Do not rerun cached POST')
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
    verify(r['protected'])
    cached={s:{(x['label'],x['request_id']):x for x in C.read(PRIOR/f'evaluation/{s}_index.json')['rows']} for s in ('raw','post')}
    for arm in LABELS:
        for q in r['requests']:
            if arm in NEW_LABELS:
                G=gpu_adapter(arm);a=G.read_raw(q,binding(arm));p=G.read_post(q,a)
                d=C.read(p['proposal_receipt'])
                if d['model_forwards']!=1 or d['checkpoint_sha256']!=r['d0_checkpoint']['sha256']:raise ValueError('Single D0 required')
            else:
                a=cached['raw'][arm,q['request_id']];p=cached['post'][arm,q['request_id']]
                old().verify_output(q,a,binding(arm)['sha256']);verify(p['files'])
                for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
                    if a[key]!=q[key]:raise ValueError('Cache identity mismatch')
                verify({p['post_path']:p['post_sha256']})
            raw.append(a);post.append(p)
    for stage,rows in [('raw',raw),('post',post)]:C.write(EV/f'{stage}_index.json',dict(status='COMPLETE',rows=rows))
    return dict(status='COMPLETE',RAW=192,POST=192,new_per_stage=96)

def indices():
    result={}
    for stage in ('raw','post'):
        rows=C.read(EV/f'{stage}_index.json')['rows']
        if len(rows)!=192 or len({(r['label'],r['request_id']) for r in rows})!=192:raise ValueError('Index incomplete')
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
    jobs=[(q,idx['raw'][arm,q['request_id']],idx['post'][arm,q['request_id']],queries[q['condition_id']],stage,rh) for arm in NEW_LABELS for q in r['requests'] for stage in ('RAW','POST')]
    if len(jobs)!=192:raise ValueError('Score budget changed')
    tick=time.perf_counter();first=score_one(jobs[0]);resource=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (resource.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (resource.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
    actual=2 if workers==2 and parallel else 1
    C.write(EV/('score_dispatch_'+str(time.time_ns())+'.json'),dict(workers=actual,threads=2,resources=resource,first_actual_score=first))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,x in enumerate(pool.map(score_one,jobs[1:]),2):
            receipts.append(x)
            if i%24==0:print(json.dumps(dict(stage='SCORE',completed=i,total=192)),flush=True)
    for x in receipts:x['sha256']=C.sha(x['path'])
    cached=[]
    for ref in C.read(PRIOR/'evaluation/score_completion.json')['receipts']:
        row=C.read(ref['path'])
        if row['label'] in LABELS[:2]:
            verify({ref['path']:ref['sha256']});verify(row['files']);cached.append(ref)
    if len(cached)!=192:raise ValueError('Cache score count')
    result=dict(status='COMPLETE',receipts=receipts+cached,new_stages=192,reused_stages=192,seconds=time.perf_counter()-tick,definition_hashes=W.definition_hashes(),model_calls=0)
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
    if len(outputs)!=384:raise ValueError('384 scores required')
    W=cpu_scorer();lookup={(x['label'],x['request_id'],x['stage']):x for x in outputs};paired=[]
    metrics=sorted(set(W.ORIGINAL_SCALARS)|{k for x in outputs for k in x if k.startswith('DESC__')})
    for before,after in COMPARISONS:
        for q in r['requests']:
            for stage in ('RAW','POST'):
                a=lookup[before,q['request_id'],stage];b=lookup[after,q['request_id'],stage]
                for metric in metrics:
                    av,bv=a.get(metric),b.get(metric);valid=isinstance(av,(int,float,bool)) and isinstance(bv,(int,float,bool)) and np.isfinite(av) and np.isfinite(bv)
                    paired.append(dict(comparison=after+'-minus-'+before,panel=q['panel'],request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],condition_type=q['condition_type'],candidate=q['candidate'],stage=stage,metric=metric,baseline_value=av,candidate_value=bv,delta=float(bv)-float(av) if valid else None,status='PASS' if valid else 'NA'))
    parents=W.average(paired,['comparison','panel','parent_id','condition_id','condition_type','stage','metric'])
    summary=W.average(parents,['comparison','panel','stage','metric'])
    C.csvwrite(EV/'per_output.csv',outputs);C.csvwrite(EV/'AVMR_PMR_all8.csv',boundary);C.csvwrite(EV/'paired_per_output.csv',paired);C.csvwrite(EV/'paired_per_parent.csv',parents);C.csvwrite(EV/'paired_summary.csv',summary)
    C.write(EV/'aggregation.json',dict(status='COMPLETE',outputs=384,new_quality_calls=192,parents=8,separate_panels=True,model_calls=0))
    return dict(status='COMPLETE',outputs=384)

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