"""Fixed terminal evaluation of the two authorized 200-update candidates."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,sys,traceback,time,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
from .common import *
EVAL=OUT/'candidate_evaluation';TASKS=EVAL/'tasks.json'
sys.path.insert(0,str(ROOT/'r2/src'))

def prepare():
    if (EVAL/'registration.json').exists():
        r=read(EVAL/'registration.json');verify(r['protected']);return r
    inputs=read(OUT/'inputs/cells.json');assert len(inputs)==10
    tasks=[];baseline=[]
    for c in inputs:
        oq=c['original_q'];q=copy.deepcopy(c['q'])
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):q[hkey]=sha(q[key])
        representation=c['label'];baseid=representation+'__'+q['request_id']
        if representation=='ORIGINAL':
            baseline.append(dict(cell_id='EXPOSURE__'+baseid,label='EXPOSURE',representation=representation,q=q,original_q=oq,
                raw_path=c['raw_path'],score_path=c['score_path'],final_path=str(OLD/'cpu/EXPOSURE'/q['request_id']/'CLEAN_FINAL_POST/post.npz'),
                final_score_path=str(OLD/'cpu/EXPOSURE'/q['request_id']/'CLEAN_FINAL_POST/scores'),origin='REUSED_CACHE'))
        else:
            tasks.append(dict(cell_id='EXPOSURE__'+baseid,label='EXPOSURE',representation=representation,q=q,original_q=oq,
                raw_path=c['raw_path'],source_npz=q['source_npz'],baseline_origin='FROZEN_FLIP_THIS_ROUND',
                raw_score_path=str(OUT/'scores'/c['cell_id'])))
        for label in ('PLAIN200','FLIP200'):
            cid=label+'__'+baseid
            tasks.append(dict(cell_id=cid,label=label,representation=representation,q=q,original_q=oq,
                raw_path=str(EVAL/'raw'/cid/'raw.npz'),source_npz=q['source_npz'],baseline_origin='NEW_TRAINED_ENDPOINT'))
    assert len(tasks)==24 and len(baseline)==6
    write(TASKS,tasks);write(EVAL/'baseline.json',baseline)
    protected={str(OUT/'inputs/cells.json'):sha(OUT/'inputs/cells.json'),str(TASKS):sha(TASKS),
        str(EVAL/'baseline.json'):sha(EVAL/'baseline.json')}
    for t in tasks+baseline:
        for q in (t['q'],t['original_q']):
            for field in ('inference_npz','metadata_npz','source_npz','noise_npz'):protected[q[field]]=sha(q[field])
            protected[q['query_descriptor']['path']]=sha(q['query_descriptor']['path'])
    for b in baseline:
        protected[b['raw_path']]=sha(b['raw_path']);protected[b['final_path']]=sha(b['final_path'])
        for d in ('score_path','final_score_path'):
            for name in ('row.json','score.json','complete.json'):protected[str(resolve(b[d])/name)]=sha(resolve(b[d])/name)
    r=dict(status='FROZEN',schema='physical_C_edgeflip_two200_endpoint_evaluation_v1',
        labels=['EXPOSURE','PLAIN200','FLIP200'],input_geometries=5,requests_per_model=10,
        original_requests=6,internal_flip_requests=4,blocked_flip_parent='f495',
        new_trained_RAW=20,already_generated_frozen_flip_RAW=4,reused_original_RAW=6,
        new_trained_generator_forwards=1000,new_D0_forwards=24,new_GEO_calls=24,
        original_final_reused=6,main_stages=['RAW','CLEAN_FINAL_POST'],
        full_CPU_chain='single official Bench D0 -> GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE',
        new_raw_full_scores=20,reused_frozen_flip_raw_scores=4,reused_original_raw_scores=6,
        new_BASE_FINAL_full_scores=48,common_C_CF_extra_raw=8,common_C_CF_extra_final=12,
        criterion='Keep C adoption/body/support and compare actual visible fragments plus full CF/FF/I/shape/removal costs; lower raw displacement or lower FF alone is insufficient',
        fixed_endpoint='global25700/local200 of both arms; no best selection',noise='Original saved free Gaussian arrays, no new draws',
        raw_barrier='All24 new RAW must be saved before any D0 call',
        protected=protected)
    write(EVAL/'registration.json',r);return r

def sample(label,worker):
    if label not in ('PLAIN200','FLIP200'):raise ValueError(label)
    r=read(EVAL/'registration.json');verify(r['protected'])
    tasks=[t for t in read(TASKS) if t['label']==label];assert len(tasks)==10
    if all((resolve(t['raw_path']).parent/'complete.json').exists() for t in tasks):
        for t in tasks:verify(read(resolve(t['raw_path']).parent/'complete.json')['files'])
        print('COMPLETE_RECEIPTS_VERIFIED_NO_MODEL_LOAD');return
    from .training import completed
    status=completed(label)
    if not status:raise RuntimeError('Final training state not complete')
    weight=OUT/'training'/label/'generator_step200.pt'
    assert sha(weight)==status['inference_export']['sha256']
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from r2.pilots.bench_natural_n_recipe_v1.runtime import build_model,inference_layout
    from r2.pilots.bench_control_transfer_v1.sample import rollout
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays,dispatch_guard
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    stage=EVAL/'sample_workers'/worker
    if stage.exists():raise RuntimeError('Prior worker attempt, no silent retry')
    stage.mkdir(parents=True);resource=snapshot();write(stage/'resources_before.json',resource);dispatch_guard(resource,2800)
    p=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']=='bench_c_edgeflip_generator_v1' and p['global_step']==25700 and p['arm']==label
    assert state_hash(p['generator_state'])==p['generator_state_sha256']
    runtime.configure_stable_runtime();model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
    model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
    write(stage/'identity.json',dict(weight=str(weight),weight_sha256=sha(weight),tasks_sha256=sha(TASKS),
        code_sha256=sha(__file__),generator_budget=500,checkpoint_selection='fixed step200'))
    write(stage/'resources_loaded.json',snapshot());observer=GPUObserver('edgeflip_candidate_sampling',interval=1.);observer.__enter__();done=[]
    try:
        for task in tasks:
            dest=resolve(task['raw_path']).parent
            if (dest/'complete.json').exists():verify(read(dest/'complete.json')['files']);continue
            if dest.exists():raise RuntimeError('Partial sampling requires explicit recovery')
            dest.mkdir(parents=True);q=task['q'];inp=arrays(q['inference_npz']);src=arrays(task['source_npz']);noise=arrays(q['noise_npz'])['z']
            layout=inference_layout(inp['C_vertices'],inp['C_faces_original'],inp['C_faces_model'],q['N'])
            write(dest/'attempt.json',dict(cell_id=task['cell_id'],weight_sha256=sha(weight),task_manifest_sha256=sha(TASKS)))
            raw,traj,audit=rollout(model,layout,noise,'PLAIN',dest/'calls.jsonl')
            mesh=raw_arrays(raw,layout,np.full(len(layout['occupied']),-1,np.float32),src)
            npz(dest/'raw.npz',mesh);npz(dest/'trajectory.npz',dict(state=traj))
            write(dest/'complete.json',dict(status='COMPLETE',generator_forwards=50,D0_forwards=0,training_updates=0,new_noise=0,
                cell_id=task['cell_id'],audit=audit,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
            done.append(task['cell_id']);write(stage/('resources_'+str(len(done))+'.json'),snapshot())
            print(json.dumps(dict(label=label,done=len(done),cell_id=task['cell_id'],seconds=audit['seconds'])),flush=True)
        after=state_hash(model.state_dict());assert after==p['generator_state_sha256']
        write(stage/'complete.json',dict(status='COMPLETE',new_forwards=50*len(done),completed=done,model_state_sha256_after=after))
    except BaseException as e:
        write(stage/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=done));raise
    finally:
        observer.__exit__(None,None,None);write(stage/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples));write(stage/'resources_after.json',snapshot())

def raw_score(task):
    from r2.pilots.bench_c_tessellation_v1.evaluate import score_one
    r=score_one(task['q'],task['raw_path'],EVAL/'raw_scores'/task['cell_id'],task['label'],
        original_q=task['original_q'] if task['representation']=='EDGE_FLIP' else None)
    return dict(cell_id=task['cell_id'],seconds=r['seconds'],peak_RSS_KiB=r['peak_RSS_KiB'])

def final_cpu(task):
    from .candidate_post import post
    r=post(task,TASKS)
    if task['representation']=='EDGE_FLIP':
        from r2.pilots.bench_c_tessellation_v1.evaluate import _common_physical_C
        final=EVAL/'cpu'/task['cell_id']/'CLEAN_FINAL_POST'
        dest=EVAL/'final_common_C'/task['cell_id'];dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'complete.json').exists():
            _common_physical_C(task['original_q'],arrays(final/'post.npz'),int(task['q']['K']),dest)
            write(dest/'complete.json',dict(status='COMPLETE',new_CF_scans=1,new_FF_scans=0,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
        else:verify(read(dest/'complete.json')['files'])
    return dict(cell_id=task['cell_id'],status=r.get('status','COMPLETE'))

def cpu(action,workers):
    from .candidate_post import verify_raw_barrier
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    tasks,barrier=verify_raw_barrier(TASKS)
    if action=='score_raw':tasks=[t for t in tasks if t['label']!='EXPOSURE'];fn=raw_score
    else:fn=final_cpu
    if workers not in (1,2):raise ValueError('Bounded CPU dispatch')
    d=EVAL/'cpu_dispatch'/str(time.time_ns());write(d/'resources_before.json',dict(resources=snapshot(),action=action,workers=workers,
        reason='Same scorer, measured <=110MiB per RAW worker; CPU original topology cascade separately bounded at2; host24 cores and >60GiB memory'))
    t=time.perf_counter();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        fs=[pool.submit(fn,task) for task in tasks]
        for f in concurrent.futures.as_completed(fs):results.append(f.result());print(json.dumps(results[-1]),flush=True)
    write(d/'complete.json',dict(status='COMPLETE',action=action,seconds=time.perf_counter()-t,results=results,resources_after=snapshot()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','sample','score_raw','post'));p.add_argument('--label');p.add_argument('--worker',default='sample_w0');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='prepare':r=prepare();print(json.dumps({k:v for k,v in r.items() if k!='protected'}))
    elif a.action=='sample':sample(a.label,a.worker)
    else:cpu(a.action,a.workers)

if __name__=='__main__':main()
