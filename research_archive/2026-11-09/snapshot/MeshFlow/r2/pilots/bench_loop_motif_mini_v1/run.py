"""Explicit recoverable actions; no automatic retries or extra samples."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, collections, csv, hashlib, inspect, io, json, sys, time, types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_loop_motif_mini_v1'
PREV=ROOT/'experiments/r2_bench_full_prior_motif_guidance_v1'
PACKAGE='r2.pilots.bench_loop_motif_mini_v1.run'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_full_prior_motif_guidance_v1 import prepare as P, generate as G, evaluate as E, run as R

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return G.sha(p)
def write(p,x):return G.write_new(p,x)
def tasks():return read(OUT/'tasks.json')
def configure():
    P.OUT=OUT;P.TASKS=OUT/'tasks.json';P.REQUESTS=OUT/'inputs/requests.json'
    P.load_requests=lambda:read(P.REQUESTS)
    P.load_tasks=lambda resolved=False:tasks()
    P.selected_tasks=tasks
    P.verify_raw_barrier=barrier
    R.OUT=OUT
    E.VERSION='bench_loop_motif_mini_original177_v1'
    E._post_proxy=post_proxy
    old_codes=G.code_hashes
    def codes():
        d=old_codes();d[str(Path(__file__))]=sha(__file__);return d
    G.code_hashes=codes

def resources():
    x=R.full_resources();p=OUT/'workflow'/('resources_'+str(time.time_ns())+'.json');write(p,x)
    print(json.dumps({'path':str(p),'gpu':x['parsed_gpu'],'RAM_available':x.get('RAM_available'),'cpu':x.get('CPU_cores')}),flush=True)
    return x

def prepare():
    if (OUT/'registration.json').exists():
        reg=read(OUT/'registration.json');G.verify(reg['protected']);return reg
    oldrequests=read(PREV/'inputs/requests.json');oldrequests=oldrequests.get('requests',[]) if isinstance(oldrequests,dict) else oldrequests
    selected=[q for q in oldrequests if q['panel_role']=='calibration']
    assert len(selected)==8 and set(q['parent_id'] for q in selected)==set(P.CALIBRATION_PARENTS)
    oldtasks=read(PREV/'tasks.json');oldbase=read(PREV/'baseline.json')
    baseline=[]
    # Resolve existing sources before redirecting writer paths to the new experiment.
    for q in selected:
        rid=q['request_id']
        for method in ('R','G0','G1','G3'):
            if method=='R':t=next(t for t in oldbase if t['request_id']==rid)
            else:t=next(t for t in oldtasks if t['request_id']==rid and t['arm']==method and
                        (method!='G3' or t['lambda_value']==.3))
            t=dict(t);t['stage_paths']={};t['score_paths']={}
            # R already contains explicit source paths.
            src=next(b for b in oldbase if b['request_id']==rid) if method=='R' else t
            for stage in E.STAGES:
                sp=E._spec(src,stage)
                t['stage_paths'][stage]=str(sp['mesh']);t['score_paths'][stage]=str(sp['score'])
                E._score_read(sp)
            baseline.append(t)
    configure()
    qs=[];ts=[]
    for q in selected:
        q=dict(q);q['previous_panel_role']=q['panel_role'];q['panel_role']='loop_development'
        G._request_assets(q);qs.append(q)
        for label,arm,lam in [('L0','G0',None),('L3','G3',.3)]:
            t=P._base_task(q,arm,lam,suffix=label);t['main_after_lock']=True
            t['loop_method']=label;t['loop_layers_zero_based']=[4,5,6,7];ts.append(t)
    # Highest-N authorized L3 request is the first real measurement, not an extra probe.
    ts.sort(key=lambda t:(0 if t['label']=='L3' and t['request_id'].startswith('bcb') and t['q']['noise_index']==0 else 1,t['label'],t['request_id']))
    write(OUT/'inputs/requests.json',qs);write(OUT/'tasks.json',ts);write(OUT/'baseline.json',baseline)
    write(OUT/'lock.json',dict(status='FROZEN',lambda_value=.3,calibration=False,inherited_from=str(PREV/'lock.json')))
    write(OUT/'inputs/metric_fields.json',read(PREV/'inputs/metric_fields.json'))
    protected={str(p):sha(p) for p in [PREV/'registration.json',PREV/'lock.json',PREV/'REPORT.md',PREV/'tasks.json',
                PREV/'analysis_numeric/final/visual/actual_review_per_output.csv',OUT/'PROTOCOL.md',Path(__file__),
                Path(G.__file__),Path(E.__file__),Path(P.__file__),ROOT/'models/equidit.py']}
    for q in qs:protected.update(G._request_assets(q))
    scorecode=read(ROOT/'experiments/r2_bench_four_routes_v1/exposure/evaluation/score_registration.json')['source_hashes']
    G.verify(scorecode);protected.update(scorecode)
    for t in baseline:
        for p in list(t['stage_paths'].values())+[str(Path(v)/'row.json') for v in t['score_paths'].values()]:protected[p]=sha(p)
    reg=dict(status='FROZEN',experiment='R2_BENCH_LOOP_MOTIF_MINI_V1',loop_layers=[4,5,6,7],rounds=2,
        loop_order=[0,1,2,3,4,4,5,5,6,6,7,7,8,9,10,11],all_50_steps=True,
        new_RAW=16,sampling_vector_fields=800,sampling_VJP=360,precheck_vector_fields=2,precheck_VJP=2,
        CFG_sample_equivalents_sampling=1600,D0=16,new_stage_scores=48,new_noise=0,training_updates=0,
        actual_layer_evaluations_per_sampling_field=16,precheck='same saved bcb G3 lambda0.3 state025; base and direct-bypass r1',
        no_automatic_retries=True,protected=protected,inputs_sha256=sha(OUT/'inputs/requests.json'),tasks_sha256=sha(OUT/'tasks.json'))
    write(OUT/'registration.json',reg)
    write(OUT/'workflow/initial_resources.json',R.full_resources())
    return reg

def install_loop(model, rounds):
    if not hasattr(model,'_loop_originals'):
        model._loop_originals=[layer.forward for layer in model.layers]
    model._loop_count=rounds;model._layer_call_order=[]
    def make(index,original):
        def forward(layer,x,c,mask=None):
            n=model._loop_count if index in (4,5,6,7) else 1
            if n==1:
                model._layer_call_order.append(index)
                return original(x,c,mask=mask)
            for _ in range(n):
                model._layer_call_order.append(index)
                full=original(x,c,mask=mask)
                x=x+(full-x)/n
            return x
        return forward
    for i,layer in enumerate(model.layers):layer.forward=types.MethodType(make(i,model._loop_originals[i]),layer)

def precheck(model):
    import torch,numpy as np
    from meshflow_control import runtime
    from r2.pilots.bench_full_prior_motif_guidance_v1.guidance import endpoint_energy
    dest=OUT/'checks/r1';done=dest/'complete.json'
    if done.exists():G.verify(read(done)['files']);return
    if dest.exists():raise RuntimeError('Incomplete r1 check: no silent replay')
    dest.mkdir(parents=True);ledger=dest/'calls.jsonl'
    t=tasks()[0];q=t['q'];rid=q['request_id']
    old=next(t for t in read(PREV/'tasks.json') if t['request_id']==rid and t['arm']=='G3' and t['lambda_value']==.3)
    sp=Path(old['output_dir'])/'states/state_025.npz'
    c=G.arrays(q['inference_npz']);condition=torch.from_numpy(c['C_vertices'][c['C_faces_model']]).cuda()
    results=[]
    write(dest/'attempt.json',dict(source=str(sp),sha256=sha(sp),vector_fields_budget=2,VJP_budget=2))
    for name in ('original','r1'):
        if name=='r1':install_loop(model,1)
        x=torch.from_numpy(G.arrays(sp)['state']).cuda().requires_grad_()
        with runtime.stable_execution(model):
            v=G._accounted_call(ledger,'vector_field',25,lambda:G.cfg_velocity(model,x,25))
            pred,err,e,_=endpoint_energy(x,v,condition,25)
            grad=G._accounted_call(ledger,'input_VJP',25,lambda:torch.autograd.grad(e,x)[0])
        results.append((v.detach().cpu().numpy(),grad.detach().cpu().numpy()))
        del x,v,grad,pred,err,e
    assert np.array_equal(results[0][0],results[1][0]) and np.array_equal(results[0][1],results[1][1]),'r1 function/VJP mismatch'
    assert model._layer_call_order==list(range(12))
    G.save_npz(dest/'comparison.npz',dict(velocity_original=results[0][0],velocity_r1=results[1][0],VJP_original=results[0][1],VJP_r1=results[1][1]))
    write(done,dict(status='PASS',exact_velocity=True,exact_input_VJP=True,vector_fields=2,input_VJP=2,parameter_updates=0,
                    files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))

def generate(indices,first=False):
    configure();G.verify(read(OUT/'registration.json')['protected'])
    chosen=[tasks()[i] for i in indices];pending=[]
    for t in chosen:
        if Path(t['raw_receipt_path']).exists():G.verify_receipt(t['output_dir'],registration_sha256=sha(OUT/'registration.json'),job_id=t['job_id'])
        else:pending.append(t)
    if not pending:return dict(status='COMPLETE_NO_MODEL_LOAD')
    if not first and not (OUT/'checks/r1/complete.json').exists():raise RuntimeError('r1 check required')
    model,identity=G.load_official_model()
    if first:precheck(model)
    install_loop(model,2)
    identity.update(loop_layers=[4,5,6,7],loop_rounds=2,loop_strength=1.,loop_scope='whole layer residual; each layer repeated before advancing',
                    frozen_weights_changed_function=True,wrapper_sha256=sha(__file__))
    original=G.cfg_velocity
    def counted(m,s,k):
        m._layer_call_order=[]
        v=original(m,s,k)
        assert m._layer_call_order==read(OUT/'registration.json')['loop_order'],'Wrong loop order'
        return v
    G.cfg_velocity=counted
    for t in pending:
        result=G.run_request(model,t['q'],t['arm'],t['lambda_value'],t['output_dir'],sha(OUT/'registration.json'),
                            job_id=t['job_id'],model_identity=identity)
        print(json.dumps(dict(completed=t['cell_id'],audit=result['audit'])),flush=True)
    G.verify(read(OUT/'registration.json')['protected'])
    return dict(status='COMPLETE',outputs=len(chosen))

def barrier(tasks_path=None):
    ts=tasks();assert len(ts)==16
    files={}
    for t in ts:
        P.verify_raw(t)
        for p in (t['raw_path'],t['raw_receipt_path']):files[p]=sha(p)
    for p in (OUT/'tasks.json',OUT/'registration.json'):files[str(p)]=sha(p)
    return ts,dict(status='PASS',outputs=16,generator_forwards=800,D0_before_barrier=0,files=files)

def post_proxy():
    from r2.pilots.bench_c_edgeflip_v1 import candidate_post as old
    ns=dict(vars(old))
    for name,value in vars(old).items():
        if inspect.isfunction(value) and value.__globals__ is vars(old):
            clone=types.FunctionType(value.__code__,ns,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;ns[name]=clone
    def identity(t,tasks_path=None):
        return dict(version='loop_mini_frozen_backend_v1',cell_id=t['cell_id'],label=t['label'],
                    tasks_sha256=sha(OUT/'tasks.json'),raw_sha256=sha(t['raw_path']),source_sha256=sha(t['source_npz']),
                    input_sha256=sha(t['q']['inference_npz']),query_sha256=t['q']['query_descriptor']['sha256'],
                    adapter_sha256=sha(__file__),delegated_code_sha256=sha(old.__file__))
    ns.update(ROOT=OUT,TASKS=OUT/'tasks.json',VERSION='loop_mini_frozen_backend_v1',load_tasks=lambda tasks_path=None:tasks(),
              verify_raw_barrier=barrier,_task_identity=identity,_raw_receipt=P.raw_receipt)
    return types.SimpleNamespace(**ns)

def cpu(indices,phase):
    configure();E.freeze_evaluation_code()
    for i in indices:
        t=tasks()[i]
        if phase in ('raw','both'):E.score_raw(t)
        if phase in ('post','both'):E.post(t)
        print(json.dumps(dict(completed=t['cell_id'],phase=phase)),flush=True)
    return dict(status='COMPLETE',count=len(indices),phase=phase)

def render(indices):
    configure();base=read(OUT/'baseline.json')
    qs=read(OUT/'inputs/requests.json')
    for i in indices:
        q=qs[i];rid=q['request_id'];group=[t for t in base if t['request_id']==rid]+[t for t in tasks() if t['request_id']==rid]
        group.sort(key=lambda t:['R','G0','G1','G3','L0','L3'].index(t['label']))
        for stage in ('RAW','CLEAN_FINAL_POST'):
            cells=[]
            for t in group:
                p=t['stage_paths'][stage] if 'stage_paths' in t else str(E._spec(t,stage)['mesh'])
                cells.append(dict(display_label=t['label'],cell_id=t['cell_id'],q=t['q'],path=p))
            page=dict(page_id=stage+'__'+rid,stage=stage,mode='loop_development',reference_q=q,cells=cells)
            E._render_page(page)
        print(json.dumps(dict(rendered=rid)),flush=True)
    return dict(status='COMPLETE',requests=len(indices))

def collect():
    configure();fields=E._fields();allrows=[]
    for t in read(OUT/'baseline.json')+tasks():
        for stage in E.STAGES:
            spec=dict(task=t,stage=stage,mesh=Path(t['stage_paths'][stage]),score=Path(t['score_paths'][stage])) if 'stage_paths' in t else E._spec(t,stage)
            r=E._score_read(spec);meta={k:t['q'][k] for k in ('request_id','parent_id','condition_id','noise_index','N','K')}
            meta.update(method=t['label'],stage=stage,origin='CACHE' if 'stage_paths' in t else 'NEW',I_status=t['q']['I_original']['status'] if 'I_original' in t['q'] else '')
            meta.update({f:r.get(f) for f in fields});allrows.append(meta)
    assert len(allrows)==144
    E._csv(OUT/'analysis/per_output_original177.csv',allrows)
    # Preserve fixed I domains from prior independent declarations.
    info=read(PREV/'registration.json')
    write(OUT/'analysis/complete.json',dict(status='COMPLETE',rows=144,new_rows=48,reused_rows=96,original_fields=177,
           files={str(OUT/'analysis/per_output_original177.csv'):sha(OUT/'analysis/per_output_original177.csv')}))
    return dict(status='COMPLETE',rows=144)

def launch(args):
    configure();res=resources();gpu=res['parsed_gpu'];head=max(2048,gpu['total_mib']*.1)
    if args.phase in ('first','generate','d0'):
        if gpu['free_mib']<head+1.2*args.peak:raise RuntimeError('Insufficient measured headroom')
    cmd=[sys.executable,'-B','-u','-m',PACKAGE,args.phase,'--indices']+list(map(str,args.indices))
    if args.phase=='d0':cmd+=['--worker',args.worker,'--peak',str(args.peak)]
    return R.observe_child(cmd,args.phase,args.worker)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','resources','first','generate','d0','raw','post','both','render','collect','launch','status'])
    p.add_argument('--indices',nargs='+',type=int,default=[]);p.add_argument('--worker',default='worker0')
    p.add_argument('--phase',choices=['first','generate','d0','raw','post','both','render']);p.add_argument('--peak',type=float,default=0)
    a=p.parse_args()
    if a.action=='prepare':r=prepare()
    elif a.action=='resources':r=resources()
    elif a.action=='launch':r=launch(a)
    elif a.action in ('first','generate'):r=generate(a.indices,a.action=='first')
    elif a.action=='d0':
        configure();E.freeze_evaluation_code();r=post_proxy().d0_batch(OUT/'tasks.json',a.indices,a.worker,a.peak)
    elif a.action in ('raw','post','both'):r=cpu(a.indices,a.action)
    elif a.action=='render':r=render(a.indices)
    elif a.action=='collect':r=collect()
    else:r=dict(raw=len(list(OUT.glob('raw/*/receipt.json'))),D0=len(list(OUT.glob('post/*/receipt.json'))),
                CPU=len(list(OUT.glob('cpu/*/complete.json'))),resources=R.gpu_snapshot())
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','files')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
