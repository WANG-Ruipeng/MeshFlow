"""Fixed existing-state readout diagnostic. No training and no new rollout."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","4")
from pathlib import Path
import argparse,contextlib,gc,hashlib,json,time,threading
import numpy as np
from meshflow_control import runtime
from meshflow_control.accounting import resource_snapshot
from meshflow_control.artifacts import file_sha256,atomic_json
from meshflow_control.checkpoints import load_checkpoint,state_hash,rng_state
from meshflow_control.models import native
from meshflow_control.models.readout import corner_indices
import torch
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parent
ART=ROOT.parents[1]
REPO=Path('/mnt/e/MeshFlow-Control')
LAYERS=(3,6,9,12)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rng_hash():
    s=rng_state()
    h=hashlib.sha256()
    for k in ('python','numpy'):h.update(repr(s[k]).encode())
    h.update(s['torch_cpu'].numpy().tobytes())
    for t in s['torch_cuda']:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()
def event(event_type,**data):
    p=ROOT/'budget/calls.jsonl';p.parent.mkdir(exist_ok=True)
    row=dict(event=event_type,time_unix=time.time(),**data)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def cell_id(label,request,step,mode):return f'{label}__{request}__s{step:02d}__{mode}'
def save_arrays(p,arrays):
    temp=p.with_suffix('.partial')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def source_checks(plan,include_checkpoints=True):
    bad=[]
    for p,h in plan['protected_files'].items():
        if not include_checkpoints and p.endswith('.pt'):continue
        if file_sha256(p)!=h:bad.append(p)
    if bad:raise RuntimeError('Read-only dependency changed: '+str(bad))
def prepare():
    if (ROOT/'plan.json').exists():raise RuntimeError('Frozen plan already exists')
    allrows=read(ART/'outputs/all_completed.json')['rows']
    parents=sorted({r['parent_id'] for r in allrows if r['panel']=='P1'})
    ids=['P1__'+pid+'_C1__n4' for pid in (parents[0],parents[-1])]
    ids += ['P2_seat2__n0','P2_seat_back4__n0']
    rows=[r for r in allrows if r['label'] in ('B1000','C1000') and r['request_id'] in ids]
    assert len(rows)==8
    checkpoints={}
    for label in ('B1000','C1000'):
        p=ART/('checkpoints/'+label[0]+'_step1000.pt')
        expected={r['checkpoint_sha256'] for r in rows if r['label']==label};assert len(expected)==1
        h=file_sha256(p);assert h==next(iter(expected))
        checkpoints[label]=dict(path=str(p),sha256=h)
    protected={str(REPO/r['path']):r['sha256'] for r in __import__('csv').DictReader((ART/'audit/final_source_hashes.csv').open(encoding='utf-8-sig'))}
    for p in [ART/'EXPERIMENT_REPORT.md',ART/'outputs/all_completed.json',ART/'audit/budget/calls.jsonl',ART/'audit/budget/gpu_occupancy.jsonl']:
        protected[str(p)]=file_sha256(p)
    for v in checkpoints.values():protected[v['path']]=v['sha256']
    streams=[]
    for r in sorted(rows,key=lambda x:(x['label'],ids.index(x['request_id']))):
        for name,hname in (('raw_path','raw_sha256'),('input_path','input_sha256')):
            assert file_sha256(r[name])==r[hname];protected[r[name]]=r[hname]
        with np.load(r['raw_path'],allow_pickle=False) as z:
            trajectory=z['trajectory'];assert trajectory.shape==(51,r['N'],9) and trajectory.dtype==np.float32
            assert trajectory[-1].tobytes()==z['raw'].tobytes()
        with np.load(r['input_path'],allow_pickle=False) as z:C=z['C'].reshape(r['K'],9);gaussian=z['z']
        assert all(t[:r['K']].tobytes()==C.tobytes() for t in trajectory)
        assert trajectory[0,r['K']:].tobytes()==gaussian[0,r['K']:].tobytes()
        streams.append({**{k:r[k] for k in ('label','request_id','panel','N','K','candidate','parent_id','input_path','input_sha256','raw_path','raw_sha256')},
            'state_hashes':{str(s):array_hash(trajectory[s]) for s in (0,25,45)},
            'next_state_hashes':{str(s):array_hash(trajectory[s+1]) for s in (0,25,45)}})
    reqs=[next({k:r[k] for k in ('request_id','panel','N','K','candidate','parent_id')} for r in rows if r['request_id']==i) for i in ids]
    plan=dict(schema='readout_effective_injection_v1',created_unix=time.time(),labels=['B1000','C1000'],requests=reqs,steps=[0,25,45],
        times=[0.,.5,.9],modes=['on','off'],layers=list(LAYERS),checkpoints=checkpoints,streams=streams,
        selection='P1 sorted parent identity first/last, NEW n4; P2 seat2 and seat_back4 n0. Fixed before new forward; not a representative random sample.',
        state_basis='Each checkpoint uses its own pre-existing ON trajectory; ON/OFF share identical state within checkpoint. No cross-model same-state causal comparison.',
        forwards=48,backwards=0,optimizer_updates=0,new_gaussians=0,new_rollouts=0,quality_rescoring=0,
        gpu_owner_limit_seconds=600,worker_count=1,cpu_threads=4,
        concurrency_reason='Small fixed 48-forward inspection using one shared POSIX GPU lock, preserving original single-worker protocol. Independent CPU plan/code review parallel. Models loaded sequentially to avoid concurrent loading peaks.',
        protected_files=protected,runner_sha256=file_sha256(__file__),
        metrics=['cast error','addition rounding error','effective/residual norm ratio (unbounded above)','fraction nonzero residual with zero effective change','cosine/projection','per-corner distribution',
                 'free prediction ON-OFF and isolated Euler-one-step difference','exact ON next-state replay'],
        limits='No quality/usability verdict, no automatic code precision change, no retraining, no full OFF rollout.',
        output_storage_budget_bytes=1024**3)
    source_checks(plan)
    atomic_json(ROOT/'plan.json',plan)
    atomic_json(ROOT/'resources_before.json',resource_snapshot())
    atomic_json(ROOT/'status.json',dict(status='REGISTERED',forward_limit=48,actual_forwards=0))
    print(json.dumps(dict(status='REGISTERED',plan_sha256=file_sha256(ROOT/'plan.json'),requests=ids,forward_limit=48)))
class Probe:
    def __init__(self,mode):self.mode=mode;self.arrays={};self.checks=[];self.visited=[]
    def __enter__(self):
        self.original=native.apply_readout
        def wrapped(hidden,known,valid,module,**kwargs):
            layer=kwargs['layer'];self.visited.append(layer)
            if self.mode=='off':return hidden
            assert hidden.shape[0]==1
            captured={}
            def hook(mod,inputs,result):
                captured['query']=inputs[0].detach();captured['residual']=result.detach()
                return None
            handle=module.register_forward_hook(hook)
            try:after=self.original(hidden,known,valid,module,**kwargs)
            finally:handle.remove()
            idx=corner_indices(valid[0]&~known[0]);before=captured['query'];residual=captured['residual']
            actual=after[0].index_select(0,idx);cast=residual.to(hidden.dtype)
            mask=(~(valid[0]&~known[0])).repeat_interleave(3)
            check=dict(layer=layer,hidden_dtype=str(hidden.dtype),residual_dtype=str(residual.dtype),
                       output_dtype=str(after.dtype),query_is_actual_free=torch.equal(before,hidden[0].index_select(0,idx)),
                       known_padding_unchanged=torch.equal(after[:,mask],hidden[:,mask]),
                       actual_add_matches=torch.equal(actual,before+cast),free_corners=len(idx))
            assert all(check[k] for k in ('query_is_actual_free','known_padding_unchanged','actual_add_matches'))
            for name,value in [('before',before),('residual',residual),('cast',cast),('after',actual)]:
                self.arrays[f'l{layer}_{name}']=value.float().cpu().numpy().copy()
            self.checks.append(check)
            return after
        native.apply_readout=wrapped
        return self
    def __exit__(self,*exc):native.apply_readout=self.original
def load_events():
    p=ROOT/'budget/calls.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def check_memory(where):
    snap=resource_snapshot()
    parts=snap['gpu']['stdout'].strip().splitlines()[1].split(',')
    total=float(parts[1].strip().split()[0]);free=float(parts[3].strip().split()[0])
    if free<max(2048.,.15*total):raise RuntimeError('GPU memory headroom breached '+where)
    event('RESOURCE',where=where,gpu=snap['gpu'],gpu_processes=snap['gpu_processes'],memory=snap['memory'])
    return snap
def run():
    plan=read(ROOT/'plan.json');assert file_sha256(__file__)==plan['runner_sha256']
    source_checks(plan)
    (ROOT/'cells').mkdir(exist_ok=True)
    events=load_events();reserved={e['cell_id'] for e in events if e['event']=='RESERVE'}
    committed={e['cell_id'] for e in events if e['event']=='COMMIT'}
    if reserved-committed:raise RuntimeError('Uncommitted attempted cell; preserve evidence and resolve without silent forward replay')
    done={}
    for p in (ROOT/'cells').glob('*.json'):
        c=read(p);assert c['status']=='COMPLETE' and file_sha256(c['array_path'])==c['array_sha256']
        assert c['cell_id'] in committed;done[c['cell_id']]=c
    count=len(reserved)
    target={cell_id(l,r['request_id'],s,m) for l in plan['labels'] for r in plan['requests'] for s in plan['steps'] for m in plan['modes']}
    assert reserved<=target
    if len(done)==48:print('All 48 completed cells reused; zero new calls');return
    atomic_json(ROOT/'status.json',dict(status='RUNNING',actual_forwards=count,forward_limit=48,plan_sha256=file_sha256(ROOT/'plan.json')))
    start=time.monotonic();arm_checks=[];monitor_stop=threading.Event()
    def monitor():
        while not monitor_stop.is_set():
            try:
                snap=resource_snapshot()
                with (ROOT/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
            except Exception as ex:
                with (ROOT/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps(dict(error=repr(ex)))+'\n')
            monitor_stop.wait(2.)
    thread=None
    try:
        with runtime.gpu_worker(ROOT/'budget','effective_injection') as owner:
            runtime._GPU_DEADLINE=min(runtime._GPU_DEADLINE,time.monotonic()+plan['gpu_owner_limit_seconds'])
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            for label in plan['labels']:
                if all(cell_id(label,r['request_id'],s,m) in done for r in plan['requests'] for s in plan['steps'] for m in plan['modes']):continue
                check_memory('before_load_'+label)
                model,optimizer,payload=load_checkpoint(plan['checkpoints'][label]['path'],device='cuda',trainable=False)
                assert optimizer is None and payload['pilot_step']==1000 and model.readout_mode=={'B1000':'mixed','C1000':'clean'}[label]
                expected_state=payload['model_state_sha256'];del payload
                before_state=state_hash(model.state_dict());assert before_state==expected_state
                before_rng=rng_hash()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                for request in plan['requests']:
                    row=next(r for r in plan['streams'] if r['label']==label and r['request_id']==request['request_id'])
                    N,K=row['N'],row['K']
                    with np.load(row['raw_path'],allow_pickle=False) as d:trajectory=d['trajectory'].copy()
                    with np.load(row['input_path'],allow_pickle=False) as d:C=d['C'].reshape(K,9).copy()
                    known=torch.zeros((1,N),dtype=torch.bool,device='cuda');known[:,:K]=True
                    valid=torch.ones_like(known);y=torch.tensor([N],device='cuda',dtype=torch.long);cond=torch.from_numpy(C).to('cuda')
                    for step in plan['steps']:
                        x=torch.from_numpy(trajectory[step:step+1].copy()).to('cuda')
                        t=torch.full((1,),step/50,dtype=torch.float32,device='cuda')
                        assert array_hash(trajectory[step])==row['state_hashes'][str(step)]
                        for mode in plan['modes']:
                            ident=cell_id(label,row['request_id'],step,mode)
                            if ident in done:continue
                            runtime.check_gpu_deadline()
                            if count>=48:raise RuntimeError('48-forward hard cap reached')
                            if time.monotonic()-start>600:raise RuntimeError('10-minute wall guard')
                            event('RESERVE',cell_id=ident,kind='diagnostic_forward',ordinal=count+1,label=label,mode=mode,step=step)
                            count+=1;tick=time.perf_counter()
                            try:
                                with torch.no_grad(),native.native_execution(model,coordinates=(x,t)),Probe(mode) as probe:
                                    velocity=model(x,t,y,valid,known)
                                torch.cuda.synchronize()
                                assert velocity.dtype==torch.bfloat16 and bool(torch.isfinite(velocity).all())
                                assert probe.visited==list(LAYERS)
                                event('RETURN',cell_id=ident,kind='diagnostic_forward',seconds=time.perf_counter()-tick)
                                with torch.no_grad():
                                    nxt=x.clone();nxt[:,K:]=x[:,K:]+velocity.float()[:,K:]*(1/50);nxt[:,:K]=cond[None]
                                arrays=dict(velocity=velocity.float().cpu().numpy().copy(),state=x.cpu().numpy().copy(),
                                            next_state=nxt.cpu().numpy().copy(),**probe.arrays)
                                assert arrays['state'].tobytes()==trajectory[step:step+1].tobytes()
                                assert arrays['next_state'][0,:K].tobytes()==C.tobytes()
                                replay=arrays['next_state'][0].tobytes()==trajectory[step+1].tobytes() if mode=='on' else None
                                p=ROOT/'cells'/(ident+'.npz');save_arrays(p,arrays)
                                receipt=dict(cell_id=ident,label=label,request_id=row['request_id'],panel=row['panel'],N=N,K=K,step=step,t=step/50,mode=mode,
                                    status='COMPLETE' if mode=='off' or replay else 'REPLAY_MISMATCH',
                                    array_path=str(p),array_sha256=file_sha256(p),trajectory_replay_exact=replay,
                                    input_state_sha256=array_hash(arrays['state']),checkpoint_sha256=plan['checkpoints'][label]['sha256'],
                                    layer_checks=probe.checks,readout_sites_visited=probe.visited,known_coordinates_next_exact=True,
                                    prediction_dtype='torch.bfloat16 stored as float32 values',seconds_instrumented=time.perf_counter()-tick)
                                atomic_json(ROOT/'cells'/(ident+'.json'),receipt)
                                if mode=='on' and not replay:raise RuntimeError('ON failed exact next-state replay; saved evidence, no further calls')
                                event('COMMIT',cell_id=ident,array_sha256=receipt['array_sha256']);done[ident]=receipt
                                if mode=='off':print(json.dumps(dict(label=label,request_id=row['request_id'],step=step,completed_forwards=count,on_replay=done[cell_id(label,row['request_id'],step,'on')]['trajectory_replay_exact'])),flush=True)
                                atomic_json(ROOT/'status.json',dict(status='RUNNING',actual_forwards=count,completed_cells=len(done),forward_limit=48))
                            except BaseException as exc:
                                event('ERROR',cell_id=ident,error=repr(exc));raise
                            finally:
                                if 'probe' in locals():probe.arrays.clear()
                            del velocity,nxt,arrays
                        check_memory(label+'__s'+str(step))
                torch.cuda.synchronize()
                after_rng=rng_hash();after_state=state_hash(model.state_dict())
                assert before_state==after_state and before_rng==after_rng
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                result=dict(label=label,model_state_sha256_before=before_state,model_state_sha256_after=after_state,
                    RNG_sha256_before=before_rng,RNG_sha256_after=after_rng,unchanged=True,optimizer_created=False,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='One loaded model plus actual fixed forwards and readout capture; phase peak reset after load. Instrumented timing not a speed benchmark.')
                arm_checks.append(result);atomic_json(ROOT/(label+'_integrity.json'),result)
                del model;gc.collect();torch.cuda.empty_cache();check_memory('after_'+label)
    except BaseException as exc:
        atomic_json(ROOT/'status.json',dict(status='STOPPED_ERROR',actual_forwards=count,completed_cells=len(done),error=repr(exc)))
        raise
    finally:
        monitor_stop.set()
        if thread:thread.join(timeout=10)
    source_checks(plan)
    assert len(done)==count==48
    size=sum(p.stat().st_size for p in (ROOT/'cells').glob('*.npz'))
    assert size<=plan['output_storage_budget_bytes']
    atomic_json(ROOT/'status.json',dict(status='COMPLETE',actual_forwards=48,on_forwards=24,off_forwards=24,
        backwards=0,optimizer_updates=0,new_gaussians=0,new_rollouts=0,geometry_rescoring=0,
        completed_cells=48,layer_records=96,all_on_exact_replay=True,all_dependencies_unchanged=True,
        scientific_source_files_modified=0,model_integrity=arm_checks,compressed_arrays_bytes=size,wall_seconds=time.monotonic()-start))
    print(json.dumps(read(ROOT/'status.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.stage=='prepare' else run()
