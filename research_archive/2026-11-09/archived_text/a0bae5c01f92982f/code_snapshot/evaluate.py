"""Small paired endpoint evaluation; old geometry/scoring implementations are read-only.

GPU commands are explicit. Score/render are separate CPU processes. No command
starts training, selects a checkpoint, changes a seed, or retries an open call.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, collections, concurrent.futures, csv, html, json, multiprocessing, time
from pathlib import Path
import numpy as np
from . import eval_common as C

ARMS=('SMALL_C200','FULL_C200')
BASE='EXP67_FROZEN'
LABELS=(BASE,*ARMS)
EV=C.OUT/'evaluation'
VERSION='r2_seat_quick_learnability_evaluation_v1'
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio',
      'base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type')


def verify(files):
    for p,h in files.items():
        if C.sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))


def manifest():
    m=C.read(C.OUT/'inputs/manifest.json')
    qs=m['requests']
    if len(qs)!=8 or len({q['request_id'] for q in qs})!=8 or len({q['parent_id'] for q in qs})!=4:
        raise ValueError('Expected exactly four TRAIN conditions x two saved noises')
    if any(q['split']!='TRAIN' or q['variant']!='base' for q in qs):raise ValueError('Wrong panel')
    for q in qs:
        verify({q['input_path']:q['input_sha256'],q['condition_source']:q['condition_source_sha256']})
    return m


def code_hashes():
    paths=[Path(__file__),Path(C.__file__),C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py',
           C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/rendering.py',
           C.ROOT/'r2/pilots/condition_width_response_v1/evaluate.py',
           C.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py']
    paths+=list((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    return {str(p):C.sha(p) for p in sorted(set(paths))}


def freeze():
    m=manifest();old=C.read(C.ROOT/'experiments/r2_seat_failure_diagnosis_v1/registration.json')
    queries=m['queries']
    if len(queries)!=4 or {q['condition_id'] for q in queries}!={q['condition_id'] for q in m['requests']}:
        raise ValueError('Four existing C query banks required')
    qs=[{k:v for k,v in q.items() if not k.startswith('baseline_')} for q in m['requests']]
    for q in qs:q.update(action='NEW')
    reg=dict(status='FROZEN',version=VERSION,labels=LABELS,requests=qs,queries=queries,
             manifest_sha256=C.sha(C.OUT/'inputs/manifest.json'),d0_checkpoint=old['d0_checkpoint'],
             baseline_checkpoint=old['checkpoint'],code_hashes=code_hashes(),
             budget=dict(new_rollouts=16,new_generator_forwards=800,new_D0_forwards=16,
                         new_scored_stages=32,reused_baseline_scored_stages=16,logical_scored_stages=48),
             science='Two fixed step200 endpoints; identical saved C/N/Gaussian. Euler50; all RAW precede single official D0, GEO_C and source orientation. No RBF.',
             metrics='Inherited original geometry/I/FF/CF/all8 AVMR-PMR/valid indexed edges, with unchanged scales and frozen C2048 query banks. No unique-GT score, no semantic metric proxy.',
             interpretation='Four familiar TRAIN conditions; held-out sampling noise. Small-pool learnability diagnostic, not DEV generalization or model promotion.')
    C.write(EV/'registration.json',reg)
    return dict(status='FROZEN',requests=8,new_rollouts=16,new_D0=16)


def registration():
    r=C.read(EV/'registration.json')
    if r['status']!='FROZEN' or tuple(r['labels'])!=LABELS or r['code_hashes']!=code_hashes():
        raise ValueError('Evaluation implementation changed after freeze')
    verify({C.OUT/'inputs/manifest.json':r['manifest_sha256']})
    return r


def binding(arm,create=False):
    if arm not in ARMS:raise ValueError('Unregistered endpoint')
    p=EV/'bindings'/(arm+'.json')
    if p.exists():return C.read(p)
    if not create:raise RuntimeError('Endpoint has not been bound: '+arm)
    checkpoint=C.OUT/'inference'/f'{arm}_generator_step200.pt'
    b=dict(path=str(checkpoint),sha256=C.sha(checkpoint),arm=arm,local_step=200,global_step=6200)
    C.write(p,b);return b


def gpu_adapter(arm):
    """Reuse protected receipt, Euler audit, D0 and source-export paths in memory."""
    G=C.module('r2.pilots.seat_failure_diagnosis_v1._quick_original_gpu_'+arm,C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py')
    G.C=C;G.LABEL=arm
    G.check_code=registration
    G.registration=lambda:dict(registration(),checkpoint=binding(arm))
    G.worker_path=lambda stage,worker,workers,name:EV/'execution'/arm/f'{stage}_{worker}_{workers}_{name}.json'
    def barrier(r):
        # The first D0 is blocked until BOTH new arms have all eight RAW.
        allrows=[]
        for label in ARMS:
            other=gpu_adapter(label)
            allrows.extend(other.read_raw(q,binding(label)) for q in r['requests'])
        G.LABEL=arm
        return [x for x in allrows if x['label']==arm]
    G.all_raw=barrier
    return G


def old():
    return C.module('_quick_original_io',C.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py')


def state_hash(state):
    from meshflow_control.checkpoints import state_hash as original
    return original(state)


def generate(arm):
    registration();b=binding(arm,create=True);G=gpu_adapter(arm);r=registration();pending=[]
    for q in r['requests']:
        p=G.record_path(q,'raw')
        if p.exists():G.read_raw(q,b)
        elif p.parent.exists():raise RuntimeError('Unresolved RAW; no silent replay: '+q['request_id'])
        else:old().read_input(q);pending.append(q)
    if not pending:return dict(status='COMPLETE',new_rollouts=0)
    C.package()
    from meshflow_control import runtime
    from meshflow_control.models.native import build_model
    from meshflow_control.sampling import sample
    import torch
    runtime.configure_stable_runtime();torch.set_num_threads(2)
    G.snap('generator_before_load_'+arm,0,1,2600.)
    verify({b['path']:b['sha256']})
    payload=torch.load(C.resolve(b['path']),map_location='cpu',weights_only=True,mmap=True)
    if (payload['arm'],payload['local_step'],payload['global_step'])!=(arm,200,6200):raise ValueError('Wrong endpoint')
    if any(k in payload for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Use inference export')
    if state_hash(payload['generator_state'])!=payload['generator_state_sha256']:raise ValueError('State identity mismatch')
    model=build_model(payload['model_config'],readout_mode='none',trainable=False).eval()
    model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda').set_trainable(False)
    identity={k:v for k,v in payload.items() if k!='generator_state'}
    C.write(G.worker_path('generator',0,1,'identity'),dict(identity,file_sha256=b['sha256']))
    G.snap('generator_resident_'+arm,0,1)
    class Budget(old().CellBudget):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.velocities=[]
        def call(self,kind,fn,context=None):
            value=super().call(kind,fn,context)
            if kind=='sampling_forward':self.velocities.append(value.detach().float().cpu().numpy()[0])
            return value
    tick=time.perf_counter();count=0
    try:
        for q in pending:
            a=old().read_input(q);G.source(q,a['C']);dest=C.checked(G.record_path(q,'raw').parent);dest.mkdir(parents=True,exist_ok=False)
            context=dict(label=arm,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=b['sha256'])
            budget=Budget(dest/'calls.jsonl',context)
            _,_,audit=sample(model,a['C'],int(q['N']),seed=q['noise_seed'],z=a['z'],budget=budget,out=dest,context=context)
            C.frozen_npz(dest/'velocities.npz',dict(velocity=np.stack(budget.velocities),source_dtype=np.array('torch.bfloat16')))
            rec=dict(q);rec.update(context);rec.update(status='COMPLETE',new_rollouts=1,new_sampling_F=50,generation_status='NEW_EULER50',
                velocities_path=str(dest/'velocities.npz'),velocities_sha256=C.sha(dest/'velocities.npz'),worker=0,workers=1)
            for field,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):rec[field+'_path']=str(dest/name);rec[field+'_sha256']=C.sha(dest/name)
            _,rec['contracts']=old().verify_output(q,rec,b['sha256']);C.write(dest/'receipt.json',rec);G.read_raw(q,b)
            count+=1;print(json.dumps(dict(stage='RAW',arm=arm,completed=count,seconds=audit['seconds'])),flush=True)
        if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise ValueError('Generator changed during inference')
        result=dict(status='COMPLETE',new_rollouts=count,new_sampling_F=50*count,seconds=time.perf_counter()-tick)
        C.write(G.worker_path('generator',0,1,'complete'),result);return result
    finally:
        del model;torch.cuda.empty_cache()


def postprocess(arm):
    G=gpu_adapter(arm)
    return G.postprocess(0,1)


def baseline_records():
    return manifest()['requests']


def collect():
    r=registration();raw=[];post=[]
    for q in baseline_records():
        a=q['baseline_raw_record'];b=q['baseline_post_record']
        old().verify_output(q,a,r['baseline_checkpoint']['sha256'])
        verify({b['post_path']:b['post_sha256']});verify(b['files'])
        raw.append(dict(a,label=BASE));post.append(dict(b,label=BASE))
    for arm in ARMS:
        G=gpu_adapter(arm)
        for q in r['requests']:
            a=G.read_raw(q,binding(arm));raw.append(a);post.append(G.read_post(q,a))
    for stage,rows in [('raw',raw),('post',post)]:C.write(EV/f'{stage}_index.json',dict(status='COMPLETE',rows=rows))
    return dict(status='COMPLETE',RAW=24,POST=24,model_calls=0)


def cpu_scorer():
    from r2.pilots.condition_width_response_v1 import evaluate as W
    if not hasattr(W,'_quick_original_hashes'):W._quick_original_hashes=W.definition_hashes()
    W.E=C;W.VERSION=VERSION;W.META=META
    W.definition_hashes=lambda:{**W._quick_original_hashes,str(Path(__file__)):C.sha(__file__),str(Path(C.__file__)):C.sha(C.__file__)}
    return W


def score_one(job):
    W=cpu_scorer();q,raw,post,query,stage,rh=job
    query=dict(query,transform_audit=dict(method='SAVED_BASE_C2048_UNCHANGED',new_queries=0))
    return W.score_one((q,raw,post,query,stage,rh))


def indices():
    result={}
    for stage in ('raw','post'):
        rows=C.read(EV/f'{stage}_index.json')['rows']
        if len(rows)!=24 or len({(x['label'],x['request_id']) for x in rows})!=24:raise ValueError('All RAW/POST incomplete')
        result[stage]={(x['label'],x['request_id']):x for x in rows}
    return result


def score(workers=2):
    os.environ['CUDA_VISIBLE_DEVICES']='';r=registration();idx=indices();W=cpu_scorer();done=EV/'score_completion.json'
    if done.exists():
        p=C.read(done)
        if p['definition_hashes']!=W.definition_hashes():raise ValueError('Scorer changed')
        verify({x['path']:x['sha256'] for x in p['receipts']});return p
    queries={q['condition_id']:q for q in r['queries']};rh=C.sha(EV/'registration.json')
    jobs=[(q,idx['raw'][arm,q['request_id']],idx['post'][arm,q['request_id']],queries[q['condition_id']],stage,rh)
          for arm in ARMS for q in r['requests'] for stage in ('RAW','POST')]
    if len(jobs)!=32:raise ValueError('Scoring budget changed')
    tick=time.perf_counter();first=score_one(jobs[0]);resource=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (resource.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (resource.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
    actual=2 if workers==2 and parallel else 1
    C.write(EV/('score_dispatch_'+str(time.time_ns())+'.json'),dict(workers=actual,threads=2,resources=resource,first_actual_score=first))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts+=list(pool.map(score_one,jobs[1:]))
    for x in receipts:x['sha256']=C.sha(x['path'])
    result=dict(status='COMPLETE',receipts=receipts,new_stages=32,reused_baseline_stages=16,seconds=time.perf_counter()-tick,definition_hashes=W.definition_hashes(),model_calls=0)
    C.write(done,result);return result


def _baseline_score_path(q,stage):
    # Manifest owns the exact old score; no old output is rescored.
    key='baseline_score_'+stage+'_path'
    if key in q:return q[key],q['baseline_score_'+stage+'_sha256']
    key='baseline_score_'+stage.lower()+'_path'
    if key in q:return q[key],q['baseline_score_'+stage.lower()+'_sha256']
    raise KeyError('Missing frozen baseline score '+stage)


def aggregate():
    r=registration();done=C.read(EV/'score_completion.json');refs=[dict(path=x['path'],sha256=x['sha256']) for x in done['receipts']]
    for q in baseline_records():
        for stage in ('RAW','POST'):
            p,h=_baseline_score_path(q,stage);refs.append(dict(path=p,sha256=h))
    outputs=[];boundary=[];quality=[]
    for ref in refs:
        verify({ref['path']:ref['sha256']});s=C.read(ref['path']);verify(s['files'])
        meta={k:s[k] for k in META};meta.update(label=s['label'],stage=s['stage'])
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None},**s['distance']}
        for row in s['boundary']:
            boundary.append(dict(meta,**row));vals['A_VMR__'+row['setting']]=row['VMR'];vals['PMR__'+row['setting']]=row['perfect']
        vals.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()})
        vals.update(C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],
                    C_query_weight_denominator=s['C_query_weight_denominator'],boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],
                    source_C_connectivity_status=s['source_C_connectivity_status'],score_path=ref['path'],score_sha256=ref['sha256'])
        outputs.append(dict(meta,**vals));quality.append(dict(meta,**s['metrics']))
    if len(outputs)!=48 or len({(x['label'],x['request_id'],x['stage']) for x in outputs})!=48:raise ValueError('Incomplete paired outputs')
    W=cpu_scorer();lookup={(x['label'],x['request_id'],x['stage']):x for x in outputs};paired=[]
    metrics=sorted(set(W.ORIGINAL_SCALARS)|{k for x in outputs for k in x if k.startswith('DESC__')})
    for before,after in ((BASE,ARMS[0]),(BASE,ARMS[1]),(ARMS[1],ARMS[0])):
        for q in r['requests']:
            for stage in ('RAW','POST'):
                a,b=lookup[before,q['request_id'],stage],lookup[after,q['request_id'],stage]
                for metric in metrics:
                    av,bv=a.get(metric),b.get(metric);valid=W.numeric(av) and W.numeric(bv)
                    paired.append(dict(comparison=after+'__minus__'+before,request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],
                        condition_type=q['condition_type'],candidate=q['candidate'],stage=stage,metric=metric,
                        baseline_value=av,candidate_value=bv,delta=float(bv)-float(av) if valid else None,status='PASS' if valid else 'NA'))
    parents=W.average(paired,['comparison','parent_id','condition_id','condition_type','stage','metric'])
    summary=W.average(parents,['comparison','stage','metric'])
    C.csvwrite(EV/'per_output.csv',outputs);C.csvwrite(EV/'AVMR_PMR_all8.csv',boundary);C.csvwrite(EV/'quality.csv',quality)
    C.csvwrite(EV/'paired_per_output.csv',paired);C.csvwrite(EV/'paired_per_parent.csv',parents);C.csvwrite(EV/'paired_summary.csv',summary)
    result=dict(status='COMPLETE',outputs=48,new_quality_calls=32,baseline_reuse=16,parents=4,independent_conditions=4,model_calls=0)
    C.write(EV/'aggregation.json',result);return result


def render():
    os.environ['CUDA_VISIBLE_DEVICES']='';r=registration();idx=indices()
    R=C.module('r2.pilots.seat_failure_diagnosis_v1._quick_sections',C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/rendering.py')
    G=C.module('_quick_draw',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    import matplotlib.pyplot as plt
    gallery=EV/'gallery';images=[];reviews=[];cards=[]
    for q in r['requests']:
        rid=q['request_id'];k=q['K'];cc=C.arrays(q['input_path'])['C'].reshape(k,3,3);source=C.arrays(q['condition_source'])['C_original'].reshape(k,3,3);center=R.center_of_c(cc)
        meshes={};hashes={q['input_path']:q['input_sha256'],q['condition_source']:q['condition_source_sha256']}
        for label in LABELS:
            for stage in ('RAW','POST'):
                rec=idx[stage.lower()][label,rid];field='raw' if stage=='RAW' else 'post';a=C.arrays(rec[field+'_path'])
                mesh=a['raw' if stage=='RAW' else 'output'].reshape(-1,3,3)
                if not C.same(mesh[:k],cc if stage=='RAW' else source):raise ValueError('Gallery C contract')
                meshes[label,stage]=mesh;hashes[rec[field+'_path']]=rec[field+'_sha256']
        verify(hashes);xyz=np.concatenate([a.reshape(-1,3) for a in meshes.values()]);lo=xyz.min(0);hi=xyz.max(0);cxyz=cc.reshape(-1,3)
        margins=np.array([.075,.1,.075])*C.L
        bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=float((hi-lo).max()*1.10),local_low=(cxyz.min(0)-margins).tolist(),local_high=(cxyz.max(0)+margins).tolist())
        dest=gallery/'images'/(rid+'.jpg');side=dest.with_suffix('.json');identity=C.clean(dict(request=q,hashes=hashes,bounds=bounds,code_hashes=code_hashes()))
        if side.exists():
            rec=C.read(side)
            if rec['identity']!=identity or C.sha(dest)!=rec['image_sha256']:raise ValueError('Frozen image changed')
        else:
            if dest.exists():raise RuntimeError('Incomplete image; explicit recovery needed')
            fig=plt.figure(figsize=(18,8.7),dpi=135,facecolor='white')
            for i,label in enumerate(LABELS):
                for j,stage in enumerate(('RAW','POST')):
                    mesh=meshes[label,stage];start=i*6+j*3
                    ax=fig.add_subplot(3,6,start+1,projection='3d');G.draw(ax,mesh,k,bounds,R.CAMERA)
                    ax.set_title(label+' | '+stage+f' | F={len(mesh)-k}',fontsize=9)
                    for v,view in enumerate(('front','side'),2):
                        ax=fig.add_subplot(3,6,start+v);R.section_plot(ax,mesh,k,center,bounds,view)
            fig.suptitle(q['parent_id'][:8]+' | '+q['condition_type']+f' | noise {q["candidate"]} | K={k} N={q["N"]}',fontsize=14)
            fig.text(.5,.040,'BLUE C, GRAY free | rows: frozen EXP67 / SMALL_C200 / FULL_C200 | left RAW, right one D0 + GEO_C + source orientation',ha='center',fontsize=10)
            fig.text(.5,.020,'Whole mesh plus two C-centered cuts; pale blue projected outline is not an intersection. Y is height. Same fixed scales; no mesh edits.',ha='center',fontsize=9)
            fig.text(.5,.003,'Nonblind AI_PROXY review; four familiar training parents. Visible contact does not certify topology or generalization.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.02,right=.99,top=.91,bottom=.10,wspace=.25,hspace=.43)
            C.checked(dest).parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.tmp')
            with tmp.open('xb') as f:fig.savefig(f,format='jpg',pil_kwargs={'quality':92,'optimize':True})
            plt.close(fig);os.replace(tmp,dest)
            rec=dict(status='RENDERED_NOT_REVIEWED',identity=identity,image_path=str(dest),image_sha256=C.sha(dest));C.write(side,rec)
        images.append(dict(request_id=rid,image_path=str(dest),image_sha256=rec['image_sha256']))
        for label in LABELS:
            for stage in ('RAW','POST'):
                reviews.append(dict(**{key:q[key] for key in META},label=label,stage=stage,review_status='NOT_REVIEWED',reviewer='AI_PROXY',blind=False,
                    image_path=str(dest),image_sha256=rec['image_sha256'],reviewed_image_sha256='',**{key:'UNKNOWN' for key in R.REVIEW_FIELDS},notes=''))
        rel='images/'+dest.name;cards.append('<h2>'+html.escape(rid)+'</h2><img style="width:100%" src="'+rel+'">')
    C.csvwrite(gallery/'images.csv',images);C.csvwrite(gallery/'AI_PROXY_review_template.csv',reviews)
    C._write_bytes(gallery/'index.html',('<!doctype html><meta charset="utf-8"><h1>Fixed four-condition quick learnability panel</h1>'+''.join(cards)).encode())
    result=dict(status='RENDERED_NOT_REVIEWED',images=8,review_rows=48,source_hashes=code_hashes(),model_calls=0,actual_visual_review='NOT_RUN_BY_RENDERER')
    C.write(gallery/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('freeze','generate','post','collect','score','aggregate','render','check'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage in ('generate','post'):
        if not a.arm:p.error('--arm required')
        G=gpu_adapter(a.arm)
        with G.owner(a.stage+'_'+a.arm,0,1):result=generate(a.arm) if a.stage=='generate' else postprocess(a.arm)
    elif a.stage=='score':result=score(a.workers)
    elif a.stage=='check':result=dict(status='PASS',requests=len(registration()['requests']),model_calls=0)
    else:result=globals()[a.stage]()
    print(json.dumps(C.clean(result),allow_nan=False),flush=True)


if __name__=='__main__':main()
