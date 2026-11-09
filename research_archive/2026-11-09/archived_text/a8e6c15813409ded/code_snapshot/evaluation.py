"""Versioned fixed-panel evaluator. No model import until an owned GPU stage.

Original sampling/denoise/scoring mathematics are imported read-only. This
adapter binds the new data-scale panel and keeps all writes below this OUT.
"""
from __future__ import annotations
import argparse, collections, copy, csv, datetime, hashlib, importlib.util
import json, os, sys, time, traceback
from pathlib import Path
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/r2_user_data_scaling_12h_v1'
VERSION = 'r2_user_scaling_evaluation_v1'
L = 9.208138264587333
HIST = 'HIST_USER_MIX_FROZEN'
HIST_SHA = '8c1287b13541b59fa7b1e70974a70b06fbe0d67160449c225e5bdc7fdd0897cd'
HIST_STATE = '1ee83ae12c370ddaf21c7aa83aabd5cf44692d09eba345464b3c6b5bb943cd89'
D0_SHA = '6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
HISTORY = ROOT / 'experiments/r2_user_mix_from_official_v1'
META = ('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio')

def resolve(value):
    value = str(value)
    if os.name != 'nt' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\','/')
    return Path(value)

def checked(value):
    p = resolve(value).absolute()
    if any(a.is_symlink() for a in (p, *p.parents)):
        raise ValueError('Output symlink forbidden: ' + str(p))
    p = p.resolve()
    if p == OUT.resolve() or not p.is_relative_to(OUT.resolve()):
        raise ValueError('Output outside fixed OUT: ' + str(p))
    return p

def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def same(a,b): return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
def read(path): return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as a: return {k:a[k].copy() for k in a.files}
def clean(a):
    if isinstance(a,dict): return {str(k):clean(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)): return [clean(v) for v in a]
    if isinstance(a,np.ndarray): return clean(a.tolist())
    if isinstance(a,np.generic): return clean(a.item())
    return a
def write(path,value,replace=False):
    p=checked(path);p.parent.mkdir(parents=True,exist_ok=True);value=clean(value)
    if p.exists() and not replace:
        if read(p) != value: raise ValueError('Frozen JSON differs: '+str(p))
        return
    temp=checked(p.with_name(p.name+'.tmp'))
    with temp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def save_npz(path,value):
    p=checked(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        prior=arrays(p)
        if set(prior)!=set(value) or any(not same(prior[k],np.asarray(v)) for k,v in value.items()): raise ValueError('Frozen NPZ differs')
        return
    tmp=checked(p.with_name(p.name+'.tmp'))
    with tmp.open('xb') as f: np.savez_compressed(f,**value);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def csvwrite(path,rows):
    rows=clean(rows);p=checked(path);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def module(name,path):
    if name not in sys.modules:
        env=os.environ.copy()
        spec=importlib.util.spec_from_file_location(name,resolve(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m
        try: spec.loader.exec_module(m)
        finally: os.environ.clear();os.environ.update(env)
    return sys.modules[name]
def old(): return module('_scaling_original_evaluation',ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py')
def package():
    p=str(ROOT/'r2/src')
    if p not in sys.path: sys.path.insert(0,p)
def event(name,**fields):
    p=checked(OUT/'evaluation/ledger.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:
        f.write(json.dumps(clean(dict(event=name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def deadline():
    end=datetime.datetime.fromisoformat(read(OUT/'registration.json')['deadline_utc'])
    if datetime.datetime.now(datetime.timezone.utc)>=end: raise RuntimeError('TASK_DEADLINE_NO_NEW_SCIENTIFIC_CALLS')
def gpu_resources(path,estimated_increment_MiB):
    package();from meshflow_control.denoise import resources
    snapshot=resources();path=checked(path)
    if path.exists():path=path.with_name(path.stem+'__'+str(time.time_ns())+path.suffix)
    write(path,snapshot)
    if snapshot['gpu']['returncode']!=0:raise RuntimeError('nvidia-smi unavailable; no GPU dispatch')
    first=snapshot['gpu']['stdout'].strip().splitlines()[0].split(',')
    total,free=float(first[1]),float(first[3])
    if free-max(2048.,.15*total)<estimated_increment_MiB:
        raise RuntimeError('Insufficient GPU headroom for this stage; no model forward dispatched')
    return snapshot

def verify(mapping):
    for p,h in mapping.items():
        if sha(p)!=h: raise ValueError('Frozen file changed: '+p)
def source(row,C):
    if sha(row['condition_source'])!=row['condition_source_sha256']: raise ValueError('Source metadata changed')
    package();from meshflow_control.source_winding import validate_source
    a=arrays(row['condition_source']);v=validate_source(a,C)
    return a,v
def load_registration():
    p=OUT/'evaluation/registration.json';r=read(p)
    if r['version']!=VERSION or r['source_hashes']!=source_hashes(): raise ValueError('Evaluation implementation changed after freeze')
    if len(r['requests'])!=140 or len({q['request_id'] for q in r['requests']})!=140: raise ValueError('Fixed request count differs')
    return r
def source_hashes():
    files=[CODE/'evaluation.py',CODE/'scoring.py',CODE/'rendering.py',ROOT/'r2/src/meshflow_control/sampling.py',
           ROOT/'r2/src/meshflow_control/denoise_model.py',ROOT/'r2/src/meshflow_control/geo_c.py',
           ROOT/'r2/src/meshflow_control/source_winding.py',ROOT/'r2/src/meshflow_control/winding_solver.py',
           ROOT/'r2/tools/avmr_pmr_v1/run.py',ROOT/'r2/pilots/anchor_aware_denoiser_v1/evaluation.py',
           ROOT/'r2/tools/c_restore_coupling_v1/evaluation.py']
    return {**old().scorer_sources(),**{str(p):sha(p) for p in files}}

def inventory():
    r=read(HISTORY/'evaluation/registration.json');c=read(ROOT/'experiments/r2_c_restore_coupling_v1/registration.json')
    p=HISTORY/'inference/R2_USER_MIX_generator_step6000.pt';available=p.is_file() and p.with_suffix('.json').is_file()
    receipt=read(p.with_suffix('.json')) if available else {}
    available=available and receipt.get('sha256')==HIST_SHA and receipt.get('generator_state_sha256')==HIST_STATE
    value=dict(status='AVAILABLE' if available else 'ASSET_UNAVAILABLE',historical_checkpoint=str(p),historical_bytes=p.stat().st_size if p.is_file() else None,recorded_sha256=HIST_SHA,
               recorded_state_sha256=HIST_STATE,hash_status='RECORDED_RECEIPT; file/state verified by sole GPU load before use',
               historical_registration=str(HISTORY/'evaluation/registration.json'),requests=r['requests'],queries=r['queries'],
               cached_proposals=c['records'],new_model_calls=0)
    write(OUT/'evaluation/asset_inventory.json',value);return value

def prepare(panel_manifest,large_label,include_history=True):
    deadline();package();regpath=OUT/'evaluation/registration.json'
    if regpath.exists(): return load_registration()
    if any((OUT/'evaluation/raw').glob('*/*/calls.jsonl')): raise ValueError('Cannot freeze after sampling')
    panel=read(panel_manifest);items=panel['conditions'] if isinstance(panel,dict) else panel
    if len(items)!=64 or len({x['parent_id'] for x in items})!=32: raise ValueError('DATA_SCALE_BLOCKED: require 32 new parents and 64 distinct conditions')
    if any(n!=2 for n in collections.Counter(x['parent_id'] for x in items).values()) or len({x['condition_id'] for x in items})!=64: raise ValueError('Duplicate/missing per-parent conditions')
    train_path=OUT/'data/train_manifest.json';train=read(train_path)
    if train.get('status')!='FROZEN':raise ValueError('Training data must be frozen before evaluation inputs')
    train_ids={uid for values in train['pools'].values() for uid in values}
    historical_training=read(HISTORY/'data/input_registration.json')
    historical_ids=set(historical_training['parents'])
    new_ids={x['parent_id'] for x in items}
    if new_ids&(train_ids|historical_ids):raise ValueError('New evaluation overlaps a scaling or historical conditional training parent')
    labels=['USER_SMALL',large_label]+([HIST] if include_history else [])
    if not large_label.startswith('USER_LARGE_'): raise ValueError('Actual LARGE size must be named')
    history=inventory();past=read(HISTORY/'evaluation/registration.json');requests=[];queries=[]
    if include_history and history['status']!='AVAILABLE':raise RuntimeError('HIST_USER_MIX asset unavailable; explicitly register --no-history, never substitute a model')
    for item in sorted(items,key=lambda x:(x['parent_id'],x['condition_id'])):
        row=copy.deepcopy(item);row['panel']='NEW32';row.setdefault('ratio',row['K']/row['N'])
        if sha(row['condition_source'])!=row['condition_source_sha256']: raise ValueError('Condition hash changed')
        cond=arrays(row['condition_source'])['C'];source(row,cond)
        if cond.shape!=(row['K'],3,3) or not 128<=row['N']<=256 or not 0<row['K']<row['N']: raise ValueError('Invalid N/K')
        query,seed=old().query_bank(cond,row['condition_id']);qp=OUT/'evaluation/queries'/(row['condition_id']+'.npz');save_npz(qp,query)
        queries.append(dict(condition_id=row['condition_id'],parent_id=row['parent_id'],panel='NEW32',path=str(qp),sha256=sha(qp),
            array_sha256={k:array_sha(v) for k,v in query.items()},query_count=2048,weight_denominator=float(query['C_weights'].sum()),
            identity=dict(condition_id=row['condition_id'],C_seed=int(seed),C_sha256=array_sha(cond),L0=L)))
        for candidate in (0,1):
            rid='NEW32__'+row['condition_id']+'__n'+str(candidate)
            seed=int.from_bytes(hashlib.sha256(('R2_USER_DATA_SCALING_12H_V1/EVAL_GAUSSIAN/'+rid).encode()).digest()[:4],'little')
            z=np.zeros((1,row['N'],9),np.float32);z[:,row['K']:]=np.random.RandomState(seed).randn(row['N']-row['K'],3,3).astype(np.float32).reshape(1,-1,9)
            ip=OUT/'evaluation/inputs'/(rid+'.npz');save_npz(ip,dict(C=cond,z=z,N=np.asarray(row['N'],np.int64),K=np.asarray(row['K'],np.int64)))
            requests.append(dict(row,request_id=rid,candidate=candidate,noise_seed=seed,input_path=str(ip),input_sha256=sha(ip),
                C_sha256=array_sha(cond),Gaussian_sha256=array_sha(z),noise_provenance='FROZEN_MT19937_NAMESPACED_REQUEST_ARRAY'))
    for q in past['requests']:
        q=copy.deepcopy(q);q['panel']='OLD6';old().read_input(q);source(q,arrays(q['input_path'])['C']);requests.append(q)
    for q in past['queries']:
        q=copy.deepcopy(q);q['panel']='OLD6';old().load_queries(q);queries.append(q)
    if set(x['parent_id'] for x in items)&set(x['parent_id'] for x in past['requests']): raise ValueError('New panel intersects old six parents')
    value=dict(version=VERSION,status='FROZEN',panel_manifest=str(resolve(panel_manifest)),panel_manifest_sha256=sha(panel_manifest),labels=labels,
        requests=requests,queries=queries,source_hashes=source_hashes(),definition=dict(L=L,query_count=2048,sampling='original Euler50',
        POST='one official D0, canonical C restore, original GEO_C, source C restore, fixed-C free orientation',RBF=False,
        quality='original full pair scans; final actual free denominators; frozen C2048 weights',boundary='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',
        required_interface='NA_NOT_REGISTERED',S='NA',E='NA',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',
        aggregation='per request paired; two noises within C, two C within parent, equal parents; NEW32 and OLD6 separate'),
        training_manifest_sha256=sha(train_path),historical_training_exclusion_sha256=sha(HISTORY/'data/input_registration.json'),
        budget=dict(logical_RAW=len(labels)*140,logical_POST=len(labels)*140,generator_F_max=len(labels)*7000,D0_F_max=len(labels)*140,quality_max=len(labels)*280))
    write(regpath,value);csvwrite(OUT/'evaluation/requests.csv',requests);event('EVALUATION_FROZEN',requests=len(requests),labels=labels);return value

def bind(label,checkpoint):
    reg=load_registration()
    if label not in reg['labels']: raise ValueError('Unknown arm')
    p=resolve(checkpoint).resolve();h=sha(p)
    if label==HIST and h!=HIST_SHA: raise ValueError('Wrong USER_MIX asset')
    value=dict(label=label,path=str(p),sha256=h,evaluation_registration_sha256=sha(OUT/'evaluation/registration.json'))
    if label!=HIST:
        receipt=read(p.with_suffix('.json'));step=int(receipt['global_step']);lock=read(OUT/'training_budget_lock.json')
        if receipt['sha256']!=h or step not in (4000,6000) or p!=(OUT/'inference'/f'{label}_generator_step{step}.pt').resolve():raise ValueError('Wrong fixed inference endpoint')
        if step!=lock['updates_per_arm']:
            fallback=read(OUT/'evaluation/time_budget_fallback.json')
            cutoff=datetime.datetime.fromisoformat(read(OUT/'registration.json')['training_cutoff_utc'])
            if fallback.get('status')!='TIME_BUDGET_FALLBACK' or fallback.get('selected_step')!=4000 or step!=4000 or datetime.datetime.now(datetime.timezone.utc)<cutoff:raise ValueError('Unregistered fixed-endpoint fallback')
        value.update(global_step=step,common_initialization_sha256=read(OUT/'initialization/common_step0000.json')['sha256'])
        for other in reg['labels']:
            otherp=OUT/'evaluation/checkpoints'/(other+'.json')
            if other not in (label,HIST) and otherp.exists():
                prior=read(otherp)
                if any(prior[k]!=value[k] for k in ('global_step','common_initialization_sha256')):raise ValueError('SMALL/LARGE endpoints or initialization differ')
    write(OUT/'evaluation/checkpoints'/(label+'.json'),value);return value

def sample_arm(label,controller_owned=False):
    if not controller_owned: raise ValueError('GPU dispatch requires controller ownership')
    reg=load_registration();binding=read(OUT/'evaluation/checkpoints'/(label+'.json'));package()
    if label not in reg['labels'] or binding['evaluation_registration_sha256']!=sha(OUT/'evaluation/registration.json'): raise ValueError('Wrong binding')
    receipts=[];pending=[];oldcache={r['request_id']:r for r in read(OUT/'evaluation/asset_inventory.json')['cached_proposals']}
    for q in reg['requests']:
        dest=OUT/'evaluation/raw'/label/q['request_id'];rp=dest/'receipt.json'
        if rp.exists():
            r=read(rp);old().verify_output(q,r,binding['sha256']);receipts.append(r);continue
        if label==HIST and q['panel']=='OLD6':
            cached=oldcache[q['request_id']]
            if any(cached[k]!=q[k] for k in ('C_sha256','Gaussian_sha256','N','K','input_sha256')): raise ValueError('Historical RAW input differs')
            origin=resolve(cached['raw_path']).parent;prior=read(origin/'receipt.json');old().verify_output(q,prior,HIST_SHA)
            r=dict(prior,label=label,panel='OLD6',status='VERIFIED_CACHE',new_rollouts=0,generation_status='VERIFIED_HISTORICAL_RAW_REUSE')
            write(rp,r);receipts.append(r)
        else:
            if dest.exists(): raise RuntimeError('Incomplete sampling requires explicit recovery; never replay automatically: '+str(dest))
            pending.append(q)
    if not pending: return dict(status='COMPLETE',new_rollouts=0,rows=receipts)
    deadline();from meshflow_control import runtime
    import torch
    torch.set_num_threads(2);runtime.configure_stable_runtime()
    from meshflow_control import denoise
    gpu_resources(OUT/'evaluation/sampling'/label/'resources_before.json',2600.)
    if label==HIST: loader=module('_scaling_historical_trainer',ROOT/'r2/pilots/user_mix_from_official_v1/trainer.py')
    else: loader=module('_scaling_new_trainer',CODE/'training.py')
    model,identity=loader.load_generator(binding['path'],device='cuda',expected_sha256=binding['sha256'])
    if label==HIST and identity['generator_state_sha256']!=HIST_STATE: raise ValueError('Wrong USER_MIX state')
    if label!=HIST and (identity['arm']!=label or identity['global_step']!=binding['global_step'] or identity['lineage']['common_initialization_sha256']!=binding['common_initialization_sha256']):raise ValueError('Wrong new arm, endpoint or common initialization')
    write(OUT/'evaluation/sampling'/label/'identity.json',identity)
    gpu_resources(OUT/'evaluation/sampling'/label/'resources_model_resident.json',0.)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    from meshflow_control.sampling import sample
    class Budget(old().CellBudget):
        def __init__(self,*a,**kw): super().__init__(*a,**kw);self.velocities=[]
        def reserve(self,kind,context=None): deadline();return super().reserve(kind,context)
        def call(self,kind,fn,context=None):
            result=super().call(kind,fn,context)
            if kind=='sampling_forward':self.velocities.append(result.detach().float().cpu().numpy()[0])
            return result
    tick=time.perf_counter()
    try:
        for q in pending:
            deadline();b=old().read_input(q);dest=checked(OUT/'evaluation/raw'/label/q['request_id']);ctx=dict(label=label,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=binding['sha256'])
            budget=Budget(dest/'calls.jsonl',ctx);_,trajectory,audit=sample(model,b['C'],q['N'],seed=q['noise_seed'],z=b['z'],budget=budget,out=dest,context=ctx)
            path=trajectory.reshape(51,q['N'],9);K=q['K']
            recurrence=len(budget.velocities)==50 and all(same(path[s+1,K:],path[s,K:]+v[K:]*np.float32(1/50)) for s,v in enumerate(budget.velocities))
            if not recurrence: raise ValueError('Actual velocity Euler recurrence mismatch')
            write(dest/'euler_audit.json',dict(status='PASS',actual_velocity_checks=50,FP32_recurrence_bitwise=True,C_all51_bitwise=True))
            r=dict(q,**ctx,status='COMPLETE',new_rollouts=1,generation_status='NEW_EULER50')
            for field,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):r[field+'_path']=str(dest/name);r[field+'_sha256']=sha(dest/name)
            _,r['contracts']=old().verify_output(q,r,binding['sha256']);write(dest/'receipt.json',r);receipts.append(r)
            write(OUT/'evaluation/sampling'/label/'progress.json',dict(completed=len(receipts),registered=140,seconds=time.perf_counter()-tick,last_audit=audit),replace=True)
        result=dict(status='COMPLETE',new_rollouts=len(pending),sampling_F=len(pending)*50,rows=receipts,seconds=time.perf_counter()-tick)
        write(OUT/'evaluation/sampling'/label/'completion.json',result);return result
    except BaseException as exc:
        event('SAMPLING_STOPPED',label=label,error=repr(exc),completed=len(receipts));raise
    finally:
        del model;torch.cuda.empty_cache()

def collect_raw(require_complete=True):
    reg=load_registration();rows=[];missing=[]
    for label in reg['labels']:
        binding=read(OUT/'evaluation/checkpoints'/(label+'.json'))
        for q in reg['requests']:
            p=OUT/'evaluation/raw'/label/q['request_id']/'receipt.json'
            if not p.exists():missing.append(dict(label=label,request_id=q['request_id'],status='NOT_RUN'));continue
            r=read(p);old().verify_output(q,r,binding['sha256']);rows.append(r)
    result=dict(status='COMPLETE' if not missing else 'PARTIAL',rows=rows,missing=missing)
    write(OUT/'evaluation/raw_index.json',result,replace=True)
    if missing and require_complete:raise RuntimeError('All registered RAW must finish before D0')
    return result

def postprocess(checkpoint,controller_owned=False):
    if not controller_owned:raise ValueError('D0 dispatch requires controller ownership')
    reg=load_registration();records=collect_raw()['rows'];package()
    from meshflow_control import denoise,geo_c,geo_io,source_winding
    cached={r['request_id']:r for r in read(OUT/'evaluation/asset_inventory.json')['cached_proposals']};model=None;rows=[];calls=0;tick=time.perf_counter()
    try:
        for r in records:
            deadline();dest=OUT/'evaluation/post'/r['label']/r['request_id'];rp=dest/'receipt.json'
            if rp.exists():
                rec=read(rp)
                if rec['raw_sha256']!=r['raw_sha256'] or rec['source_sha256']!=r['condition_source_sha256']:raise ValueError('POST identity drift')
                verify(rec['files']);rows.append(rec);continue
            raw=arrays(r['raw_path'])['raw'].reshape(-1,3,3);K=int(r['K']);src,sv=source(r,raw[:K]);pp=dest/'proposal.npz';pr=dest/'proposal_receipt.json'
            if (dest/'geometry_started.json').exists():raise RuntimeError('Incomplete CPU geometry/export; explicit recovery required')
            if pr.exists():proposal=denoise.cached_proposal(pp,pr,raw,r['raw_sha256']);newcall=0
            elif dest.exists():raise RuntimeError('Incomplete D0 forward; no silent retry')
            elif r['label']==HIST and r['panel']=='OLD6':
                oldr=cached[r['request_id']];proposal=denoise.cached_proposal(oldr['D0_denoised_path'],oldr['D1_receipt_path'],raw,r['raw_sha256']);newcall=0
                pp=resolve(oldr['D0_denoised_path']);pr=resolve(oldr['D1_receipt_path'])
            else:
                if model is None:
                    deadline();from meshflow_control import denoise_model
                    import torch
                    torch.set_num_threads(2);gpu_resources(OUT/'evaluation/D0/resources_before.json',2600.)
                    model,mi=denoise_model.load_denoiser(checkpoint,device='cuda',expected_sha256=D0_SHA);write(OUT/'evaluation/D0/identity.json',mi)
                    gpu_resources(OUT/'evaluation/D0/resources_model_resident.json',0.)
                dest.mkdir(parents=True,exist_ok=False);write(dest/'D0_started.json',dict(raw_sha256=r['raw_sha256'],checkpoint_sha256=D0_SHA,status='STARTED'))
                deadline();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();event('D0_FORWARD_RESERVED',label=r['label'],request_id=r['request_id'])
                proposal=denoise_model.predict_proposal(model,raw);torch.cuda.synchronize();calls+=1;newcall=1
                save_npz(pp,dict(official_unconstrained_proposal=proposal));write(pr,dict(status='PASS',raw_sha256=r['raw_sha256'],checkpoint_sha256=D0_SHA,output_sha256=sha(pp),model_forwards=1,
                    seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            denoise.validate_proposal(proposal,raw);write(dest/'geometry_started.json',dict(status='STARTED',new_D0_F=newcall))
            den=proposal.copy();den[:K]=raw[:K];save_npz(dest/'pre_geo.npz',dict(output=den,C=raw[:K],known_faces=np.asarray(K)))
            geometry=geo_c.process(den,K);canonical=geo_io._validate_result(den,K,geometry);save_npz(dest/'post_canonical.npz',canonical)
            result=source_winding.export_source_winding(canonical,src);post=result['oriented_arrays'];save_npz(dest/'post.npz',post)
            geo_io.write_indexed_ply(checked(dest/'post.ply'),post['vertices'],post['faces'])
            v,f=geo_io.read_indexed_ply(dest/'post.ply')
            if not same(v,post['vertices']) or not same(f,post['faces']):raise ValueError('PLY readback differs')
            write(dest/'geometry.json',{k:v for k,v in geometry.items() if not isinstance(v,np.ndarray)})
            write(dest/'source_restoration.json',result['restoration_summary']);write(dest/'orientation.json',{k:v for k,v in result['orientation_result'].items() if not isinstance(v,np.ndarray)})
            rec={k:r[k] for k in (*META,'label')};rec.update(status='COMPLETE',raw_path=r['raw_path'],raw_sha256=r['raw_sha256'],source_sha256=r['condition_source_sha256'],
                proposal_path=str(pp),proposal_receipt=str(pr),post_path=str(dest/'post.npz'),post_sha256=sha(dest/'post.npz'),canonical_path=str(dest/'post_canonical.npz'),
                pre_geo_path=str(dest/'pre_geo.npz'),source_restoration=result['restoration_summary'],orientation=result['orientation_result']['summary'],new_D0_F=newcall,GEO_calls=1,
                files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
            write(rp,rec);rows.append(rec)
            write(OUT/'evaluation/D0/progress.json',dict(completed=len(rows),registered=len(records),new_D0_F=calls,seconds=time.perf_counter()-tick),replace=True)
        result=dict(status='COMPLETE',rows=rows,new_D0_F_this_invocation=calls,seconds=time.perf_counter()-tick);write(OUT/'evaluation/post_index.json',result);return result
    except BaseException as exc:event('POST_STOPPED',error=repr(exc),completed=len(rows),new_D0_F_returned=calls);raise
    finally:
        if model is not None:
            del model;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['inventory','prepare','bind','sample','postprocess','collect'],required=True)
    p.add_argument('--panel-manifest');p.add_argument('--large-label');p.add_argument('--label');p.add_argument('--checkpoint');p.add_argument('--controller-owned',action='store_true');p.add_argument('--no-history',action='store_true');a=p.parse_args()
    if a.stage=='inventory':r=inventory()
    elif a.stage=='prepare':r=prepare(a.panel_manifest,a.large_label,not a.no_history)
    elif a.stage=='bind':r=bind(a.label,a.checkpoint)
    elif a.stage=='sample':r=sample_arm(a.label,a.controller_owned)
    elif a.stage=='postprocess':r=postprocess(a.checkpoint,a.controller_owned)
    else:r=collect_raw(False)
    print(json.dumps(dict(status=r['status'] if 'status' in r else 'BOUND',stage=a.stage),ensure_ascii=False))
if __name__=='__main__':main()
