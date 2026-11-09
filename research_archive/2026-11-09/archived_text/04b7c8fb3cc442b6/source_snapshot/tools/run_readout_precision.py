"""Bounded READOUT_PRECISION_PATH_AND_ROLLOUT_V1, external artifacts only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import argparse,contextlib,csv,gc,hashlib,json,subprocess,time,threading,sys,difflib
import numpy as np
from meshflow_control import runtime
from meshflow_control.artifacts import atomic_json as _atomic_json,file_sha256
from meshflow_control.accounting import resource_snapshot
from meshflow_control.checkpoints import load_checkpoint,state_hash,rng_state
from meshflow_control.mesh_io import write_ply,read_ply
import torch
torch.set_num_threads(4)
REPO=Path(__file__).resolve().parents[1]
ART=Path('/mnt/e/MeshFlow-Control-artifacts')
OLD=ART/'followups/readout_effective_injection_v1'
ROOT=ART/'followups/readout_precision_path_v1'
PRECISIONS=('LEGACY_BF16','FP32_REFERENCE')
MODES=('on','off');LAYERS=(3,6,9,12)

def atomic_json(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True);return _atomic_json(p,value)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def cid(label,req,step,mode):return f'{label}__{req}__s{step:02d}__{mode}'
def rng_hash():
    v=rng_state();h=hashlib.sha256()
    for k in ('python','numpy'):h.update(repr(v[k]).encode())
    h.update(v['torch_cpu'].numpy().tobytes())
    for x in v['torch_cuda']:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()
def save_npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def event(event_type,**kw):
    p=ROOT/'budget/calls.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(event=event_type,time_unix=time.time(),**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def events():
    p=ROOT/'budget/calls.jsonl';return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def budget_snapshot():
    es=events();counts={k:sum(e['event']=='RESERVE' and e['kind']==k for e in es) for k in ('preflight','fixed','sampling')}
    attempts=sum(e['event']=='ROLLOUT_START' for e in es)
    returns=sum(e['event']=='RETURN' for e in es);commits=sum(e['event']=='COMMIT' for e in es)
    errors=[e for e in es if e['event']=='ERROR']
    occ=ROOT/'budget/gpu_occupancy.jsonl';orows=[json.loads(x) for x in occ.read_text().splitlines()] if occ.exists() else []
    used=sum(x['seconds'] for x in orows if x['event']=='END')
    return dict(actual_forwards=sum(counts.values()),by_kind=counts,returned_forwards=returns,committed_forwards=commits,
        new_rollout_attempts=attempts,new_rollouts_complete=sum(e['event']=='ROLLOUT_COMPLETE' for e in es),cached_rollouts=8,cached_fixed_cells=48,
        backward=0,optimizer_updates=0,new_gaussians=0,new_conditions=0,errors=errors,gpu_completed_owner_seconds=used,
        limits=dict(total_forwards=1664,fixed=48,preflight=16,sampling=1600,rollout_attempts=32,gpu_seconds=2700))
def persist_budget():atomic_json(ROOT/'budget.json',budget_snapshot())
def validate_ledger():
    es=events();groups={k:{} for k in ('RESERVE','RETURN','COMMIT')}
    for i,e in enumerate(es):
        if e['event'] not in groups:continue
        group=groups[e['event']]
        if e['id'] in group:raise RuntimeError('Duplicate ledger cell '+e['id'])
        group[e['id']]=(i,e)
    for ident,(i,e) in groups['RESERVE'].items():
        if ident not in groups['RETURN'] or ident not in groups['COMMIT']:raise RuntimeError('Unclosed attempt; preserve and recover returned arrays, never silent retry: '+ident)
        assert i<groups['RETURN'][ident][0]<groups['COMMIT'][ident][0]
        receipt=read(groups['COMMIT'][ident][1]['receipt_path'])
        assert receipt['call_id']==ident and receipt['status']=='COMPLETE'
        assert file_sha256(receipt['array_path'])==receipt['array_sha256']
    assert set(groups['RESERVE'])==set(groups['RETURN'])==set(groups['COMMIT'])
    return set(groups['COMMIT'])
def old_cell(label,req,step,mode):return OLD/'cells'/(cid(label,req,step,mode)+'.json')
def check_sources(plan,weights=True):
    for p,h in plan['protected_files'].items():
        if not weights and p.endswith('.pt'):continue
        if file_sha256(p)!=h:raise RuntimeError('Protected file changed '+p)
def input_arrays(row):
    assert file_sha256(row['input_path'])==row['input_sha256']
    with np.load(row['input_path'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    N,K=row['N'],row['K'];assert int(d['N'])==N and int(d['K'])==K
    assert d['known_mask'].dtype==d['valid_mask'].dtype==bool
    assert same(d['known_mask'],np.arange(N)<K) and bool(d['valid_mask'].all())
    assert d['z'].dtype==d['C'].dtype==np.float32 and d['z'].shape==(1,N,9)
    return d

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists() and (ROOT/'outputs/manifest.json').exists():
        p=read(ROOT/'plan.json');check_sources(p);print('Prepared identity verified; no model calls');return
    old=read(OLD/'plan.json');assert old['labels']==['B1000','C1000']
    allrows=read(ART/'outputs/all_completed.json')['rows'];reqs=old['requests'];ids=[x['request_id'] for x in reqs]
    rows=[next(r for r in allrows if r['label']==l and r['request_id']==q) for l in old['labels'] for q in ids]
    protected={p:h for p,h in old['protected_files'].items() if not p.startswith(str(REPO)+'/')}
    source_before=[];before_dir=ROOT/'source_before';before_dir.mkdir(exist_ok=True)
    for row in csv.DictReader((ART/'audit/final_source_hashes.csv').open(encoding='utf-8-sig')):
        source_before.append(row);src=REPO/row['path'];dst=before_dir/row['path']
        if not dst.exists():
            assert file_sha256(src)==row['sha256'],str(src)
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
        assert file_sha256(dst)==row['sha256']
    atomic_json(ROOT/'source_before_manifest.json',dict(files=source_before))
    git={}
    for name,path in [('original',Path('/mnt/e/MeshFlow')),('control',REPO)]:
        git[name]={cmd:subprocess.run(['git','-C',str(path)]+cmd.split(),capture_output=True,text=True).__dict__ for cmd in ['status --short','rev-parse HEAD']}
        for row in git[name].values():row.pop('args',None)
    atomic_json(ROOT/'git_before.json',git)
    for name in ['plan.json','analysis/metric_definitions.json','analysis/layer_metrics.csv','analysis/prediction_pairs.csv','analysis/cell_audit.csv','analysis/summary.json','derived_layer_ratios.csv','B1000_integrity.json','C1000_integrity.json','run_check.py','analyze.py']:
        p=OLD/name;protected[str(p)]=file_sha256(p)
    fixed_cache=[]
    for row in rows:
        d=input_arrays(row);N,K=row['N'],row['K']
        for k in ['raw_path','input_path','reference_path']:assert file_sha256(row[k])==row[k.replace('_path','_sha256')]
        with np.load(row['raw_path'],allow_pickle=False) as z:
            tr=z['trajectory'];raw=z['raw'];assert tr.shape==(51,N,9) and tr.dtype==np.float32
            assert same(tr[-1],raw) and same(tr[0,K:],d['z'][0,K:])
            assert all(x[:K].tobytes()==d['C'].tobytes() for x in tr)
        for p in [Path(row['result_path']),Path(row['raw_path']).with_suffix('.ply'),Path(row['reference_path'])]:protected[str(p)]=file_sha256(p)
        for s in old['steps']:
            for mode in MODES:
                jp=old_cell(row['label'],row['request_id'],s,mode);c=read(jp)
                assert c['status']=='COMPLETE' and c['checkpoint_sha256']==row['checkpoint_sha256']
                assert c['N']==N and c['K']==K and c['t']==s/50 and file_sha256(c['array_path'])==c['array_sha256']
                with np.load(c['array_path'],allow_pickle=False) as z:
                    assert same(z['state'][0],tr[s])
                    if mode=='on':assert same(z['next_state'][0],tr[s+1]) and c['trajectory_replay_exact'] is True
                protected[str(jp)]=file_sha256(jp);protected[c['array_path']]=c['array_sha256'];fixed_cache.append(dict(**c,receipt_path=str(jp)))
    plan=dict(schema='READOUT_PRECISION_PATH_AND_ROLLOUT_V1',labels=old['labels'],requests=reqs,steps=[0,25,45],precisions=list(PRECISIONS),modes=list(MODES),layers=list(LAYERS),
        checkpoints=old['checkpoints'],source_rows=rows,old_followup=str(OLD),fixed_cache=fixed_cache,
        protected_files=protected,new_fixed_forwards=48,planned_preflight_forwards=12,preflight_cap=16,
        planned_new_rollouts=24,cached_rollouts=8,sampling_cap=1600,total_forward_cap=1664,gpu_seconds_cap=2700,
        worker_count=1,cpu_score_workers=2,cpu_threads_per_score=2,
        fixture_requests=[ids[0],ids[2]],fixture_step=25,
        full_layer_capture='B/C x first P1/P2 x t=.5 x four layers =16 FP32 layer arrays; remaining80 FP32 layer records statistics+hash only',
        zero_norm='UNDEFINED; absolute norms/RMS retained; no clipping norm ratios',
        local_replay='CPU same old BF16 h/r -> FP32 addition; not downstream model result',
        preflight_order='4 legacy replay +8 zero-residual probes; 4 of48 planned FP32_OFF references evaluated within preflight and reused; remaining44 fixed calls only after gate PASS',
        rollout_order='all eight caches plus 24 new independent Euler50; no quality-based early stop',
        created_unix=time.time())
    check_sources(plan);atomic_json(ROOT/'plan.json',plan)
    atomic_json(ROOT/'checkpoint_and_input_manifest.json',dict(checkpoints=old['checkpoints'],rows=rows,old_fixed_cache=fixed_cache))
    atomic_json(ROOT/'resources_prepare.json',resource_snapshot());persist_budget()
    manifest=[]
    for row in rows:
        for precision,mode in [('LEGACY_BF16','ON'),('LEGACY_BF16','OFF'),('FP32_REFERENCE','ON'),('FP32_REFERENCE','OFF')]:
            label=row['label']+'__'+precision+'__'+mode
            if precision=='LEGACY_BF16' and mode=='ON':
                manifest.append(dict(row,model_label=row['label'],label=label,precision=precision,readout=mode,generation_status='VERIFIED_CACHE',ply_path=str(Path(row['raw_path']).with_suffix('.ply'))))
    atomic_json(ROOT/'outputs/manifest.json',dict(rows=manifest))
    print(json.dumps(dict(status='PREPARED',cached_fixed=len(fixed_cache),cached_rollouts=len(manifest),planned_new_forwards=1260)))

class Capture:
    def __init__(self,save_full):self.full=save_full;self.arrays={};self.metrics=[];self.dtypes=[]
    def __call__(self,record):
        if record['site']!='readout_addition':return
        from readout_precision_analysis import layer_numbers
        layer=record['layer'];values={k:record[k].detach().float().cpu().numpy().copy() for k in ('before','residual','cast','after')}
        self.metrics.append(dict(layer=layer,hidden_dtype=str(record['before'].dtype),residual_dtype=str(record['residual'].dtype),cast_dtype=str(record['cast'].dtype),output_dtype=str(record['after'].dtype),
            array_hashes={k:ah(v) for k,v in values.items()},arrays_saved=self.full,**layer_numbers(**values)))
        if self.full:
            self.arrays.update({f'l{layer}_{k}':v for k,v in values.items()})

class DtypeProbe:
    def __init__(self,model):self.model=model;self.rows=[];self.handles=[]
    def tensors(self,x):
        if isinstance(x,torch.Tensor):return [x]
        if isinstance(x,(tuple,list)):return [v for a in x for v in self.tensors(a)]
        if isinstance(x,dict):return [v for a in x.values() for v in self.tensors(a)]
        return []
    def note(self,name,inp,out):
        f=lambda v:sorted({str(t.dtype) for t in self.tensors(v) if t.is_floating_point()})
        self.rows.append(dict(site=name,input_float_dtypes=f(inp),output_float_dtypes=f(out)))
    def __enter__(self):
        for name,module in self.model.named_modules():
            if name and (isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.Embedding)) or any(s in name for s in ['norm','x_embedder','t_embedder','y_embedder','attn','context_encoder','readouts','final_layer'])):
                self.handles.append(module.register_forward_hook(lambda m,i,o,n=name:self.note(n,i,o)))
        return self
    def observer(self,record):
        self.note('native_'+str(record['site'])+'_'+str(record.get('layer','')),record,record)
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
    def assert_precision(self,p):
        if p=='FP32_REFERENCE':
            bad=[r for r in self.rows if any(d!='torch.float32' for d in r['input_float_dtypes']+r['output_float_dtypes'])]
            assert not bad,str(bad[:3])
        assert self.rows

def reserve_call(ident,kind):
    b=budget_snapshot();assert b['actual_forwards']<1664
    assert b['by_kind'][kind]<{'fixed':48,'preflight':16,'sampling':1600}[kind]
    runtime.check_gpu_deadline();event('RESERVE',id=ident,kind=kind)
def call_forward(model,x,t,y,valid,known,*,ident,kind,precision,mode,path,capture_full=False,inspect_dtype=False,metadata=None):
    from meshflow_control.precision import precision_execution
    path=Path(path);jp=path.with_suffix('.json')
    if jp.exists():
        c=read(jp);assert c['status']=='COMPLETE' and c['call_id']==ident and file_sha256(c['array_path'])==c['array_sha256'];return c
    reserve_call(ident,kind);tick=time.perf_counter();capture=Capture(capture_full) if kind=='fixed' and mode=='on' else None
    probe=DtypeProbe(model) if inspect_dtype else None
    try:
        with torch.no_grad(),precision_execution(model,precision=precision,readout=mode,coordinates=(x,t),capture=capture), (probe if probe else contextlib.nullcontext()):
            velocity=model(x,t,y,valid,known,observer=probe.observer if probe else None)
        torch.cuda.synchronize();forward_seconds=time.perf_counter()-tick
        expected=torch.bfloat16 if precision=='LEGACY_BF16' else torch.float32
        assert velocity.dtype==expected and velocity.shape==x.shape and bool(torch.isfinite(velocity).all())
        if probe:probe.assert_precision(precision)
        K=int(known[0].sum());nxt=x.clone();nxt[:,K:]=x[:,K:]+velocity.float()[:,K:]*(1/50);nxt[:,:K]=x[:,:K]
        arrays=dict(velocity=velocity.float().cpu().numpy().copy(),state=x.cpu().numpy().copy(),next_state=nxt.cpu().numpy().copy())
        assert same(arrays['next_state'][:,:K],arrays['state'][:,:K]) and np.isfinite(arrays['next_state']).all()
        if capture:arrays.update(capture.arrays);assert [m['layer'] for m in capture.metrics]==list(LAYERS)
        event('RETURN',id=ident,kind=kind,seconds=forward_seconds)
        save_npz(path,**arrays)
        c=dict(call_id=ident,status='COMPLETE',precision=precision,mode=mode,array_path=str(path),array_sha256=file_sha256(path),
            state_sha256=ah(arrays['state']),next_state_sha256=ah(arrays['next_state']),velocity_dtype=str(velocity.dtype),
            layer_metrics=capture.metrics if capture else [],dtype_records=probe.rows if probe else [],seconds=forward_seconds,**(metadata or {}))
        atomic_json(jp,c);event('COMMIT',id=ident,kind=kind,receipt_path=str(jp));persist_budget();return c
    except BaseException as ex:event('ERROR',id=ident,kind=kind,error=repr(ex));persist_budget();raise

def frozen_state(row,step):
    d=input_arrays(row)
    with np.load(row['raw_path'],allow_pickle=False) as z:state=z['trajectory'][step:step+1].copy()
    return tensors(d,state,step)
def tensors(d,state,step):
    return (torch.from_numpy(state).cuda(),torch.full((1,),step/50,dtype=torch.float32,device='cuda'),
        torch.tensor([int(d['N'])],dtype=torch.long,device='cuda'),torch.from_numpy(d['valid_mask'][None].copy()).cuda(),torch.from_numpy(d['known_mask'][None].copy()).cuda())
def metadata(row,step):return {k:row[k] for k in ('label','request_id','panel','N','K','checkpoint_sha256','input_sha256')}|dict(step=step,t=step/50,cell_id=cid(row['label'],row['request_id'],step,'off'))
def fixed_call(model,row,step,mode,inspect=False):
    ident=cid(row['label'],row['request_id'],step,mode);meta=metadata(row,step);meta['cell_id']=ident
    full=mode=='on' and step==25 and row['request_id'] in read(ROOT/'plan.json')['fixture_requests']
    return call_forward(model,*frozen_state(row,step),ident='fixed__'+ident,kind='fixed',precision='FP32_REFERENCE',mode=mode,
        path=ROOT/'fixed/cells'/(ident+'.npz'),capture_full=full,inspect_dtype=inspect or full,metadata=meta)
def extract(c,key='velocity'):
    with np.load(c['array_path'],allow_pickle=False) as z:return z[key].copy()
def monitor_loop(stop,stage):
    while not stop.is_set():
        snap=resource_snapshot();snap['stage']=stage
        with (ROOT/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
        stop.wait(2)
def sources_freeze():
    paths=[REPO/r['path'] for r in read(ROOT/'source_before_manifest.json')['files']]
    paths += [p for p in [REPO/'src/meshflow_control/precision.py',Path(__file__),REPO/'tools/readout_precision_analysis.py'] if p not in paths]
    current={str(p):file_sha256(p) for p in paths}
    p=ROOT/'execution_source_hashes.json'
    if p.exists():assert read(p)==current,'Executed sources changed; explicit version recovery required'
    else:atomic_json(p,current)
def run_gpu(stage):
    plan=read(ROOT/'plan.json');validate_ledger();check_sources(plan);sources_freeze()
    if stage!='preflight':assert read(ROOT/'preflight/status.json')['status']=='PASS'
    if stage=='preflight' and (ROOT/'preflight/status.json').exists():assert read(ROOT/'preflight/status.json')['status']=='PASS';print('Preflight PASS reused; zero forwards');return
    if stage=='fixed-state' and len(list((ROOT/'fixed/cells').glob('*.json')))==48:print('48 fixed cells verified and reused; zero forwards');return
    if stage=='rollout' and len(read(ROOT/'outputs/manifest.json')['rows'])==32:print('32 outputs reused; zero forwards');return
    used=budget_snapshot()['gpu_completed_owner_seconds'];assert used<2700
    stop=threading.Event();thread=None;stage_tick=time.perf_counter();checks=[]
    try:
        with runtime.gpu_worker(ROOT/'budget',stage):
            runtime._GPU_DEADLINE=min(runtime._GPU_DEADLINE,time.monotonic()+2700-used)
            thread=threading.Thread(target=monitor_loop,args=(stop,stage),daemon=True);thread.start()
            for label in plan['labels']:
                rs=[r for r in plan['source_rows'] if r['label']==label]
                snap=resource_snapshot();atomic_json(ROOT/('resources_'+stage+'_'+label+'.json'),snap)
                line=snap['gpu']['stdout'].strip().splitlines()[1].split(',');assert float(line[3].split()[0])>=max(2048,.15*float(line[1].split()[0]))
                load_tick=time.perf_counter();model,opt,payload=load_checkpoint(plan['checkpoints'][label]['path'],device='cuda',trainable=False)
                assert opt is None and payload['pilot_step']==1000 and model.readout_mode=={'B1000':'mixed','C1000':'clean'}[label]
                expected=payload['model_state_sha256'];del payload
                before=state_hash(model.state_dict());assert before==expected;before_rng=rng_hash();torch.cuda.reset_peak_memory_stats()
                event('MODEL_LOAD',stage=stage,label=label,seconds=time.perf_counter()-load_tick)
                if stage=='preflight':
                    for row in rs:
                        if row['request_id'] not in plan['fixture_requests']:continue
                        stem=label+'__'+row['request_id'];meta=metadata(row,25)
                        c=call_forward(model,*frozen_state(row,25),ident='preflight__replay__'+stem,kind='preflight',precision='LEGACY_BF16',mode='on',path=ROOT/'preflight/cells'/('replay__'+stem+'.npz'),inspect_dtype=True,metadata=meta)
                        previous=read(old_cell(label,row['request_id'],25,'on'));assert same(extract(c),extract(previous))
                        checks.append(dict(check='legacy_replay',label=label,request_id=row['request_id'],bitwise=True))
                        fp_off=fixed_call(model,row,25,'off',True)
                        for precision in PRECISIONS:
                            c=call_forward(model,*frozen_state(row,25),ident='preflight__zero__'+precision+'__'+stem,kind='preflight',precision=precision,mode='zero',path=ROOT/'preflight/cells'/('zero__'+precision+'__'+stem+'.npz'),inspect_dtype=True,metadata=meta)
                            ref=read(old_cell(label,row['request_id'],25,'off')) if precision=='LEGACY_BF16' else fp_off
                            assert same(extract(c),extract(ref))
                            checks.append(dict(check='zero_matches_OFF',label=label,request_id=row['request_id'],precision=precision,bitwise=True))
                elif stage=='fixed-state':
                    for row in rs:
                        for step in plan['steps']:
                            for mode in MODES:fixed_call(model,row,step,mode)
                        print(json.dumps(dict(stage=stage,label=label,request_id=row['request_id'],budget=budget_snapshot()['by_kind'])),flush=True)
                else:
                    for row in rs:
                        for precision,mode in [('LEGACY_BF16','off'),('FP32_REFERENCE','on'),('FP32_REFERENCE','off')]:rollout(model,row,precision,mode)
                after_rng=rng_hash();after=state_hash(model.state_dict());assert before==after and before_rng==after_rng
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                atomic_json(ROOT/'integrity'/(stage+'__'+label+'.json'),dict(before=before,after=after,RNG_before=before_rng,RNG_after=after_rng,unchanged=True,optimizer=False,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='phase reset after load; not standalone process or training VRAM'))
                del model;gc.collect();torch.cuda.empty_cache()
                atomic_json(ROOT/('resources_after_'+stage+'_'+label+'.json'),resource_snapshot())
        if stage=='preflight':atomic_json(ROOT/'preflight/status.json',dict(status='PASS',checks=checks,new_preflight_calls=12,planned_fixed_OFF_references=4))
        atomic_json(ROOT/(stage+'_status.json'),dict(status='COMPLETE',seconds=time.perf_counter()-stage_tick))
    except BaseException as ex:
        atomic_json(ROOT/(stage+'_status.json'),dict(status='ERROR',error=repr(ex),seconds=time.perf_counter()-stage_tick));raise
    finally:
        stop.set()
        if thread:thread.join(timeout=10)
        persist_budget()
    check_sources(plan);print(json.dumps(budget_snapshot()),flush=True)

def rollout(model,row,precision,mode):
    name=row['label']+'__'+precision+'__'+mode.upper();ident=name+'__'+row['request_id'];dst=ROOT/'outputs'/name/row['request_id'];rp=dst/'receipt.json'
    if rp.exists():
        r=read(rp);assert file_sha256(r['raw_path'])==r['raw_sha256'];return
    dst.mkdir(parents=True,exist_ok=True);started=dst/'started.json'
    if not started.exists():
        assert budget_snapshot()['new_rollout_attempts']<32
        event('ROLLOUT_START',id=ident,label=name,request_id=row['request_id'])
        atomic_json(started,dict(label=name,checkpoint_sha256=row['checkpoint_sha256'],input_sha256=row['input_sha256']))
    d=input_arrays(row);N,K=row['N'],row['K'];state=d['z'].copy();state[:,:K]=d['C'].reshape(1,K,9)
    path=[state[0].copy()];velocities=[];stepchecks=[];timings=[];tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(50):
        callid='rollout__'+ident+'__s'+str(step).zfill(2)
        c=call_forward(model,*tensors(d,state,step),ident=callid,kind='sampling',precision=precision,mode=mode,path=dst/'steps'/f's{step:02d}.npz',metadata=dict(label=row['label'],request_id=row['request_id'],step=step,t=step/50,N=N,K=K))
        with np.load(c['array_path'],allow_pickle=False) as z:
            assert same(z['state'],state);state=z['next_state'].copy();velocity=z['velocity'].copy()
        assert state[0,:K].tobytes()==d['C'].tobytes() and np.isfinite(state).all()
        velocities.append(velocity[0]);path.append(state[0].copy());timings.append(c['seconds'])
        stepchecks.append(dict(step=step,finite=True,known_bitwise=True,N=N,K=K,state_sha256=ah(state),velocity_dtype=c['velocity_dtype']))
    tr=np.stack(path);raw=state[0];v=np.stack(velocities)
    save_npz(dst/'raw.npz',raw=raw,free=raw[K:].reshape(-1,3,3),C=d['C'],z=d['z'],trajectory=tr,velocities=v,
        known_mask=d['known_mask'],valid_mask=d['valid_mask'],known_face_indices=np.arange(K,dtype=np.int64),source_corner_to_output_vertex=np.arange(3*K,dtype=np.int64))
    plytemp=dst/'raw.partial.ply';write_ply(plytemp,raw);assert same(read_ply(plytemp)[0].reshape(N,9),raw);os.replace(plytemp,dst/'raw.ply')
    audit=dict(nfe=50,known_all51_bitwise=True,known_coordinates_bitwise=True,NPZ_C_bitwise=True,PLY_C_bitwise=True,export_npz_ply_bitwise=True,
        N=N,K=K,seconds=time.perf_counter()-tick,model_forward_seconds=sum(timings),precision=precision,readout=mode.upper(),prediction_parameterization='velocity; no extra conversion',
        actual_velocity_dtype='torch.bfloat16' if precision=='LEGACY_BF16' else 'torch.float32',integration_dtype='torch.float32',seed=row['noise_seed'],frozen_gaussian_reused=True,
        checkpoint_sha256=row['checkpoint_sha256'],input_sha256=row['input_sha256'],known_state_checks=51,step_checks=stepchecks,output_stage='RAW',postprocessing=None,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='rollout with loaded weights and actual journaling; not pure deployment time',human_usability='NOT_RUN')
    atomic_json(dst/'run.json',audit)
    out=dict(row,model_label=row['label'],label=name,precision=precision,readout=mode.upper(),raw_path=str(dst/'raw.npz'),raw_sha256=file_sha256(dst/'raw.npz'),
        result_path=str(dst/'run.json'),ply_path=str(dst/'raw.ply'),ply_sha256=file_sha256(dst/'raw.ply'),generation_status='GENERATED')
    atomic_json(rp,out);event('ROLLOUT_COMPLETE',id=ident,receipt_path=str(rp),nfe=50,seconds=audit['seconds'])
    m=read(ROOT/'outputs/manifest.json');m['rows']=[r for r in m['rows'] if (r['label'],r['request_id'])!=(name,row['request_id'])]+[out]
    atomic_json(ROOT/'outputs/manifest.json',m);persist_budget();print(json.dumps(dict(stage='rollout',completed=budget_snapshot()['new_rollouts_complete'],label=name,request_id=row['request_id'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','preflight','fixed-state','rollout','score','render','summarize','verify']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage in ('preflight','fixed-state','rollout'):run_gpu(a.stage)
    elif a.stage in ('score','render'):
        subprocess.run([sys.executable,'-B',str(REPO/'tools/readout_precision_evaluation.py'),'--stage',a.stage,'--out',str(ROOT)],check=True)
    elif a.stage=='summarize':
        subprocess.run([sys.executable,'-B',str(REPO/'tools/readout_precision_analysis.py'),'--root',str(ROOT)],check=True)
        subprocess.run([sys.executable,'-B',str(REPO/'tools/readout_precision_evaluation.py'),'--stage','summarize','--out',str(ROOT)],check=True)
    else:
        validate_ledger();check_sources(read(ROOT/'plan.json'));sources_freeze();persist_budget()
        b=budget_snapshot();assert b['by_kind']['fixed']==48 and b['by_kind']['sampling']==1200 and b['new_rollouts_complete']==24 and not b['errors']
        assert len(read(ROOT/'outputs/manifest.json')['rows'])==32
        atomic_json(ROOT/'verification.json',dict(status='PASS',budget=b,source_hashes_unchanged=True,model_calls_in_verification=0))
        print(json.dumps(b))
if __name__=='__main__':main()