"""Frozen EXP67 -> FULL_SEAT continuation, with detached interface diagnostics.

Only the original world-coordinate Native/T1+role+Geo FM is differentiated.
This module never dispatches itself: the root controller owns resource checks,
cleanup completion and interleaved evaluation at fixed 0/1000/3000/6000 steps.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
from pathlib import Path
import sys,copy,json,time,signal,traceback,hashlib,argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from . import common as C
C.package()
from meshflow_control import runtime
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.models.native import build_model,native_execution
from meshflow_control.checkpoints import state_hash,rng_state,restore_rng
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy,_check_generator
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.dataset import collate,model_inputs,training_time
from meshflow_control.data.io import read_json,atomic_json,freeze_json,append_event,digest,save_npz,json_hash,array_hash
from r2.pilots.user_data_scaling_32_67_v1 import training_data as olddata
from r2.pilots.user_condition_finetune_v1 import trainer as support

ARM=C.TRAIN_LABEL
BASE_STEP=6000
UPDATES=6000
FIXED=(1000,3000,6000)
SCHEMA='r2_full_seat_learning_curve_training_v1'


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([C.VERSION,'training',*parts],separators=(',',':')).encode()).digest()[:4],'little')


def registration():
    r=C.load_registration()
    if r.get('status')!='FROZEN' or (r['updates_per_arm'],r['batch'],r['microbatch'])!=(UPDATES,8,1):
        raise ValueError('Only frozen 6000-update batch8/micro1 registration accepted')
    if r.get('source_global_step',BASE_STEP)!=BASE_STEP:raise ValueError('EXP67 step6000 required')
    if digest(C.resolve(r['source_data']))!=r['source_data_sha256']:raise ValueError('Source dataset changed')
    if digest(C.resolve(r['interface_manifest_path']))!=r['interface_manifest_sha256']:raise ValueError('Interface label manifest changed')
    bins=np.asarray(r['diagnostic_time_bin_edges'],np.float64)
    if bins.ndim!=1 or len(bins)<2 or bins[0]!=0 or bins[-1]!=1 or np.any(np.diff(bins)<=0):raise ValueError('Invalid frozen time bins')
    for path,expected in r.get('training_code_hashes',{}).items():
        if digest(C.resolve(path))!=expected:raise ValueError('Frozen training code changed: '+path)
    return r


def _source_payload(reg,path=None,expected=None):
    path=C.resolve(path or reg['source_checkpoint']);expected=expected or reg['source_checkpoint_sha256']
    if digest(path)!=expected:raise ValueError('Full source/resume checkpoint file changed')
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if p['model_config']!=MODEL_CONFIG:raise ValueError('Wrong model architecture')
    for key,h,keyhash in [('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')]:
        if h(p[key])!=p[keyhash]:raise ValueError('Full checkpoint internal hash mismatch: '+key)
    if path==C.resolve(reg['source_checkpoint']):
        if p['global_step']!=BASE_STEP or p.get('arm')!='USER_EXPANDED_67' or p.get('update_in_progress') is not False:
            raise ValueError('Need completed original EXP67 live generator/AdamW step6000')
        for key,rkey in [('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')]:
            if p[key]!=reg[rkey]:raise ValueError('Frozen EXP67 source identity differs: '+key)
        if p['protocol'].get('EMA') is not False or any(k in p for k in ('ema','ema_state','generator_ema')):
            raise ValueError('Source protocol is live weights with EMA=false; no state substitution')
    return p


class InterfaceMasks:
    def __init__(self,reg,dataset):
        self.path=C.resolve(reg['interface_manifest_path']);self.meta=C.read(self.path);self.dataset=dataset
        tasks=self.meta.get('tasks',self.meta.get('records'))
        if isinstance(tasks,list):tasks={t['task_id']:t for t in tasks}
        if not isinstance(tasks,dict):raise ValueError('Interface manifest tasks mapping required')
        self.rows=tasks
        if not set(reg['train_task_ids']).issubset(tasks):raise ValueError('Training task lacks interface record')
    @lru_cache(maxsize=128)
    def source(self,task_id):
        row=self.rows[task_id];value=row.get('interface_mask',row)
        path=value.get('mask_path',value.get('path',value.get('npz',value.get('interface_path'))))
        expected=value.get('mask_sha256',value.get('sha256',value.get('interface_sha256')))
        if path is None or expected is None:raise ValueError('Interface path/hash absent for '+task_id)
        p=C.resolve(path)
        if not p.is_absolute():p=self.path.parent/p
        if digest(p)!=expected:raise ValueError('Source interface mask changed')
        key=value.get('mask_key',value.get('array_key','source_interface_corner_mask'))
        with np.load(p,allow_pickle=False) as z:mask=z[key].copy()
        task=self.dataset.tasks[task_id];n,k=task['N'],task['K']
        if mask.dtype!=np.bool_ or mask.shape!=(n,3):raise ValueError('Interface mask must be bool[N,3]')
        if mask[np.asarray(task['source_face_ids'],np.int64)].any():raise ValueError('Interface mask contains known-C corners')
        count=value.get('mask_count',value.get('corner_count'))
        if count is not None and int(mask.sum())!=count:raise ValueError('Interface mask count changed')
        return mask
    def attach(self,s):
        original=self.source(s['task_id']);ids=s['source_face_ids'];corners=s['corner_permutations'];n=len(ids)
        # These source IDs/corner maps already include free pre-permutation,
        # nested face/corner OT and the final all-face shuffle.
        mapped=original[ids][np.arange(n)[:,None],corners]
        if mapped[s['known_mask']].any() or int(mapped.sum())!=int(original.sum()):raise ValueError('Interface label OT transport failed')
        s['interface_corner_mask']=mapped;s['interface_mask_sha256']=array_hash(mapped)
        s['interface_source_mask_sha256']=array_hash(original)
        return s


def prepare_plan():
    reg=registration();d=olddata.Dataset(C.resolve(reg['source_data']));tasks=sorted(reg['train_task_ids'])
    if len(tasks)!=len(set(tasks)) or len(tasks)<8:raise ValueError('At least eight distinct frozen FULL tasks required')
    by={}
    masks=InterfaceMasks(reg,d)
    for tid in tasks:
        t=d.tasks[tid]
        if t['condition_type']!='SEAT_FULL' or t['K']<2:raise ValueError('Only K>=2 SEAT_FULL tasks')
        if t['uid'] in by:raise ValueError('This arm requires exactly one FULL condition per parent')
        d.verify_condition(tid);masks.source(tid);by[t['uid']]=tid
    parents=sorted(by);counts=Counter({u:0 for u in parents});prng=np.random.RandomState(seed('parent_balance'));rows=[]
    for b in range(UPDATES):
        shuffle=prng.permutation(len(parents));picked=sorted(shuffle,key=lambda j:counts[parents[int(j)]])[:8]
        arng=np.random.RandomState(seed('augmentation',b));alpha=np.r_[np.ones(4,np.float32),arng.uniform(.9,1.1,4).astype(np.float32)];alpha=alpha[arng.permutation(8)]
        if np.count_nonzero(alpha==1)!=4:raise ValueError('Ambiguous augmentation; no redraw')
        for slot,j in enumerate(picked):
            uid=parents[int(j)];tid=by[uid];t=d.tasks[tid];counts[uid]+=1;i=b*8+slot;ts=seed('time',i)
            rows.append(dict(sample_index=i,effective_batch_index=b,step=b+1,global_step=BASE_STEP+b+1,slot_index=slot,
                             arm=ARM,uid=uid,task_id=tid,N=t['N'],K=t['K'],ratio=t['K']/t['N'],alpha=float(alpha[slot]),
                             t_seed=ts,t=float(training_time(ts)),gaussian_seed=seed('gaussian',i),epsilon_seed=seed('epsilon',i),
                             permutation_seed=seed('permutation',i),condition_type='SEAT_FULL',region_type='seat'))
    if max(counts.values())-min(counts.values())>1:raise ValueError('Parent exposure imbalance')
    plan=dict(schema='r2_full_seat_training_plan_v1',arm=ARM,records=rows,parent_counts=dict(counts),
              condition_counts=dict(Counter(r['task_id'] for r in rows)),registration_sha256=digest(C.OUT/'registration.json'),
              interface_manifest_sha256=reg['interface_manifest_sha256'],source_data_sha256=reg['source_data_sha256'],
              updates=UPDATES,samples=len(rows),free_coordinate_count=sum(9*(r['N']-r['K']) for r in rows),
              semantics='Balanced distinct parents; one fixed coherent FULL condition per parent; original R2 noise/time/width distributions')
    freeze_json(C.checked(C.OUT/'plans'/f'{ARM}.json'),plan)
    return dict(status='PASS',plan_path=str(C.OUT/'plans'/f'{ARM}.json'),plan_sha256=digest(C.OUT/'plans'/f'{ARM}.json'),
                parents=len(parents),samples=len(rows),updates=UPDATES,model_calls=0)


def batch(d,masks,records,index,plan_sha):
    folder=C.checked(C.OUT/'ot'/ARM);folder.mkdir(parents=True,exist_ok=True);path=folder/f'{index:05d}.npz'
    key=json_hash(records);receipt=path.with_suffix('.json');attempt=path.with_suffix('.attempt.json')
    identity=dict(records_sha256=key,plan_sha256=plan_sha,batch_index=index)
    if receipt.exists():
        r=read_json(receipt)
        if r['identity']!=identity or r['sha256']!=digest(path):raise ValueError('OT cache changed')
        with np.load(path,allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    else:
        if attempt.exists() or path.exists():raise RuntimeError('Incomplete OT attempt; explicit repair required')
        freeze_json(attempt,dict(identity=identity,unix=time.time()));maps=None
    builder=olddata.Builder(d,maps)
    samples=[masks.attach(olddata.validate_sample(builder.sample(r),r)) for r in records]
    if maps is None:
        save_npz(path,**builder.new_maps);atomic_json(receipt,dict(identity=identity,sha256=digest(path),actual_OT=8,status='PASS'))
    return samples


def _validate_optimizer(model,opt,global_step,hyper):
    if not BASE_STEP<=global_step<=BASE_STEP+UPDATES:raise ValueError('Continuation clock outside6000..12000')
    base.validate_optimizer(model,None,opt,'COMMON',global_step)
    if support.optimizer_hyperparameters(opt)!=hyper:raise ValueError('Inherited AdamW hyperparameters changed')


def diagnostic(sample,prediction,target,bins):
    """Detached normal-forward errors; no new loss or model call.

    Endpoint values are the registered (1-t)^2 scaled velocity errors. They
    do not claim an additional rollout or exact FP32 endpoint reconstruction.
    """
    with torch.no_grad():
        squared=(prediction.detach()-target.detach()).float().square()[0].reshape(int(sample['y']),3,3).cpu().numpy()
    if not np.isfinite(squared).all():raise FloatingPointError('Nonfinite diagnostic error')
    free=sample['valid_mask']&~sample['known_mask'];inter=sample['interface_corner_mask']&free[:,None]
    other=free[:,None]&~inter;percorner=squared.sum(axis=-1,dtype=np.float64)
    one_minus=float(np.float32(1)-sample['t']);factor=one_minus**2;t=float(sample['t']);bi=min(len(bins)-2,int(np.searchsorted(bins,t,side='right')-1))
    row=dict(sample_index=int(sample['sample_index']),uid=sample['object_id'],task_id=sample['task_id'],
             t=t,time_bin=bi,time_bin_lower=float(bins[bi]),time_bin_upper=float(bins[bi+1]),endpoint_error_factor=factor,
             interface_mask_sha256=sample['interface_mask_sha256'],interface_source_mask_sha256=sample['interface_source_mask_sha256'])
    for name,mask in [('interface',inter),('other',other)]:
        count=int(mask.sum())*3;sse=float(percorner[mask].sum());row.update({name+'_scalar_count':count,name+'_velocity_sse':sse,
            name+'_velocity_mse':sse/count if count else None,name+'_endpoint_scaled_sse':factor*sse,
            name+'_endpoint_scaled_mse':factor*sse/count if count else None,name+'_status':'PASS' if count else 'NA_NO_COORDINATES'})
    if row['interface_scalar_count']+row['other_scalar_count']!=sample['free_coordinate_count']:raise ValueError('Interface/free diagnostic partition differs')
    row['all_free_velocity_sse']=row['interface_velocity_sse']+row['other_velocity_sse']
    return row


def update(model,opt,samples,ledger,step,hyper,bins):
    den=support.validate_samples(samples,ARM,step,effective_batch=True)
    if any(s['K']<2 or s['condition_type']!='SEAT_FULL' for s in samples):raise ValueError('Non-FULL training exposure')
    g=BASE_STEP+step;_validate_optimizer(model,opt,g-1,hyper)
    torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    result=dict(FM=0.,total_loss=0.,free_coordinate_denominator=den,forward_seconds=0.,backward_seconds=0.,sample_losses=[])
    diagnostics=[]
    for micro,s in enumerate(samples):
        b=collate([s],'cuda');ctx=dict(local_step=step,global_step=g,microbatch=micro,sample_index=s['sample_index'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'])
        with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
            v,sec=ledger.call('train_forward',lambda:model(*model_inputs(b)).float(),ctx);result['forward_seconds']+=sec
            loss=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=den)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
            value=float(loss.detach());_,sec=ledger.call('train_backward',loss.backward,ctx);result['backward_seconds']+=sec
        di=diagnostic(s,v,b['u'],bins);di.update(local_step=step,global_step=g,FM_contribution=value,batch_free_coordinate_denominator=den)
        if not np.isclose(di['all_free_velocity_sse']/den,value,rtol=2e-6,atol=1e-8):raise ValueError('Detached diagnostic does not partition original FM')
        diagnostics.append(di);result['FM']+=value;result['sample_losses'].append(dict(sample_index=s['sample_index'],FM_contribution=value));del v,loss,b
    params=list(model.parameters())
    if any(p.grad is None for p in params):raise RuntimeError('Unused generator parameter')
    result['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    _,result['optimizer_seconds']=ledger.call('optimizer_update',opt.step,dict(local_step=step,global_step=g))
    if any(not bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite updated generator')
    _validate_optimizer(model,opt,g,hyper)
    result.update(total_loss=result['FM'],update_seconds=time.perf_counter()-tick,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result,diagnostics


def save(path,payload):
    path=C.checked(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        r=read_json(path.with_suffix('.json'))
        if digest(path)!=r['sha256']:raise ValueError('Existing checkpoint file changed')
        for key in ('generator_state_sha256','optimizer_state_sha256','rng_sha256','local_step','global_step','arm'):
            if key in payload and r.get(key)!=payload[key]:raise ValueError('Existing checkpoint state differs: '+key)
        return r
    temp=path.with_suffix('.partial')
    with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    r=dict(path=str(path),sha256=digest(path),**{k:payload[k] for k in ('generator_state_sha256','optimizer_state_sha256','rng_sha256','local_step','global_step','arm') if k in payload})
    atomic_json(path.with_suffix('.json'),r);return r


def bind(label,receipt,step):
    binding=dict(path=receipt['path'],sha256=receipt['sha256'],generator_state_sha256=receipt['generator_state_sha256'],
                 local_step=step,global_step=BASE_STEP+step,arm=ARM,label=label,EMA=False,checkpoint_selection='FIXED_PREREGISTERED_STEP')
    freeze_json(C.checked(C.OUT/'evaluation/bindings'/f'{label}.json'),binding)
    return binding


def export_initial():
    reg=registration();path=C.resolve(reg.get('step0_inference_path',C.OLD/'inference/USER_EXPANDED_67_generator_step6000.pt'))
    receipt=read_json(path.with_suffix('.json'));expected=reg.get('step0_inference_sha256',receipt['sha256'])
    if digest(path)!=expected:raise ValueError('Existing EXP67 inference file changed')
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if p['global_step']!=BASE_STEP or p['generator_state_sha256']!=reg['source_generator_sha256'] or state_hash(p['generator_state'])!=reg['source_generator_sha256']:
        raise ValueError('Step0 inference does not match live continuation generator')
    if any(k in p for k in ('optimizer','rng','ema')):raise ValueError('Unexpected state in inference export')
    return bind('FULL_S0000',dict(path=str(path),sha256=expected,generator_state_sha256=p['generator_state_sha256']),0)


def _resource():
    r=C.resources()
    try:r['free_mib']=float(r['gpu']['stdout'].splitlines()[0].split(',')[3])
    except (ValueError,IndexError,KeyError):r['free_mib']=None
    return r


def train(stop_step,attempt=None):
    if stop_step not in FIXED:raise ValueError('Only fixed evaluation segment endpoints allowed')
    reg=registration();cleanup=C.resolve(reg['cleanup_receipt_path'])
    if digest(cleanup)!=reg['cleanup_receipt_sha256'] or C.read(cleanup).get('status') not in ('COMPLETE','PASS'):
        raise ValueError('User requested old unused checkpoint cleanup before any training; completion receipt missing')
    planpath=C.OUT/'plans'/f'{ARM}.json';plan=read_json(planpath);plan_sha=digest(planpath)
    if plan['registration_sha256']!=digest(C.OUT/'registration.json') or plan['arm']!=ARM or len(plan['records'])!=UPDATES*8:raise ValueError('Plan/registration differs')
    folder=C.checked(C.OUT/'training'/ARM);folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json';previous=read_json(status) if status.exists() else None
    if previous and previous['status']=='COMPLETE':return previous
    if previous and previous['status'] not in ('SAFE_SEGMENT','INTERRUPTED_SAFE'):raise RuntimeError('Unsafe prior attempt; no rollback/replay')
    step=int(previous['local_step']) if previous else 0
    if step>=stop_step:raise ValueError('Segment already completed; no duplicate training dispatch')
    r=_resource();C.write(folder/f'resources_dispatch_{int(time.time())}.json',r)
    if r['free_mib'] is None or r['free_mib']<1.2*4096+2048:raise RuntimeError('RESOURCE_WAIT: >=6963MiB free required before model load')
    attempt=attempt or 'attempt_'+str(int(time.time()));attempt=support.safe_attempt_id(attempt)
    owner=folder/'worker_owner.json'
    with owner.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),attempt=attempt,stop_step=stop_step),f)
    adir=folder/'attempts'/attempt;adir.mkdir(parents=True,exist_ok=False)
    ledger=support.CallLedger(folder/'calls.jsonl',dict(train_forward=8*UPDATES,train_backward=8*UPDATES,optimizer_update=UPDATES),ARM,attempt)
    state=dict(status='LOADING',arm=ARM,local_step=step,global_step=BASE_STEP+step,attempt=attempt,stop_step=stop_step)
    d=olddata.Dataset(C.resolve(reg['source_data']));masks=InterfaceMasks(reg,d);bins=reg['diagnostic_time_bin_edges']
    observer=support.GPUObserver('full_seat_train',interval=3.);pending={};pool=ThreadPoolExecutor(max_workers=2)
    flag={'stop':False};start=time.perf_counter();handlers={}
    def interrupt(*unused):flag['stop']=True
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
    try:
        torch.set_num_threads(2);runtime.configure_stable_runtime()
        torch.cuda.set_per_process_memory_fraction(5*1024**3/torch.cuda.get_device_properties(0).total_memory)
        with observer:
            source=C.resolve(previous['checkpoint']['path']) if previous else C.resolve(reg['source_checkpoint'])
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            p=_source_payload(reg,source,expected);ledger.require_exact_cursor(step)
            if previous and (p.get('schema')!=SCHEMA or p['local_step']!=step or p['protocol']!=reg or p['plan_sha256']!=plan_sha):raise ValueError('Resume branch/protocol/stream differs')
            if p['global_step']!=BASE_STEP+step:raise ValueError('Source optimizer clock differs')
            model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
            if p['optimizer_parameter_names']!=base.optimizer_names(model,None):raise ValueError('Optimizer parameter order differs')
            model.load_state_dict(p['generator_state'],strict=True);model.to('cuda');opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
            if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:
                raise ValueError('Exact live generator/AdamW/RNG restoration failed')
            hyper=support.optimizer_hyperparameters(opt);_validate_optimizer(model,opt,BASE_STEP+step,hyper)
            lineage=dict(source_file_sha256=reg['source_checkpoint_sha256'],source_generator_sha256=reg['source_generator_sha256'],source_optimizer_sha256=reg['source_optimizer_sha256'],source_rng_sha256=reg['source_rng_sha256'],
                         optimizer_restored=True,rng_restored=True,EMA=False,EMA_status='NOT_PRESENT_IN_SOURCE',source_stream_provenance_only=True,new_full_seat_plan_sha256=plan_sha)
            C.write(adir/'initialization.json',dict(**lineage,loaded_generator_sha256=p['generator_state_sha256'],loaded_optimizer_sha256=p['optimizer_state_sha256'],loaded_rng_sha256=p['rng_sha256'],restored_global_step=BASE_STEP+step,hyperparameters=hyper))
            del p
            state.update(status='TRAINING');atomic_json(status,state)
            def checkpoint():
                ledger.require_exact_cursor(step);_validate_optimizer(model,opt,BASE_STEP+step,hyper)
                g,o,rng=_cpu_copy(model.state_dict()),_cpu_copy(opt.state_dict()),_cpu_copy(rng_state());_check_generator(g)
                p=dict(schema=SCHEMA,experiment=C.VERSION,arm=ARM,local_step=step,global_step=BASE_STEP+step,update_in_progress=False,
                       generator_state=g,generator_state_sha256=state_hash(g),optimizer=o,optimizer_state_sha256=base.tree_hash(o),rng=rng,rng_sha256=base.tree_hash(rng),
                       optimizer_parameter_names=base.optimizer_names(model,None),model_config=copy.deepcopy(MODEL_CONFIG),protocol=reg,lineage=lineage,plan_sha256=plan_sha,
                       stream_state=dict(batch_index=step,next_sample_index=step*8,parent_exposure_counts=dict(Counter(x['uid'] for x in plan['records'][:step*8]))),execution=ledger.snapshot(),EMA=False)
                state['checkpoint']=save(C.OUT/'checkpoints'/f'{ARM}_step{step}.pt',p)
                if step in FIXED:
                    label=f'FULL_S{step:04d}';q={k:p[k] for k in ('experiment','arm','local_step','global_step','generator_state','generator_state_sha256','model_config','lineage')}
                    q.update(schema='r2_full_seat_learning_curve_generator_v1',inference_contract=dict(readout_mode='none',forward='BF16',parameters='FP32',integration='FP32'),training_checkpoint_sha256=state['checkpoint']['sha256'],EMA=False)
                    state['inference_export']=save(C.OUT/'inference'/f'{label}_generator.pt',q);bind(label,state['inference_export'],step)
                append_event(adir/'events.jsonl',dict(event='CHECKPOINT',**state['checkpoint']))
            segment_start=step
            for current in range(step+1,stop_step+1):
                if current==segment_start+1 or current%5==0 or flag['stop']:
                    rr=_resource();append_event(folder/'resource_checks.jsonl',rr)
                    if flag['stop'] or rr['free_mib'] is None or rr['free_mib']<2048:
                        checkpoint();state['status']='INTERRUPTED_SAFE';state['reason']='signal_or_resource_headroom';break
                began=time.perf_counter()
                for bidx in range(current-1,min(stop_step,current+1)):
                    if bidx not in pending:pending[bidx]=pool.submit(batch,d,masks,plan['records'][bidx*8:(bidx+1)*8],bidx,plan_sha)
                samples=pending.pop(current-1).result();input_s=time.perf_counter()-began
                if flag['stop']:
                    checkpoint();state['status']='INTERRUPTED_SAFE';state['reason']='signal_after_input_before_model';break
                for s in samples:
                    append_event(folder/'exposure.jsonl',dict(local_step=current,global_step=BASE_STEP+current,sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],condition_type=s['condition_type'],
                        N=int(s['y']),K=s['K'],alpha=s['alpha'],t=float(s['t']),input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key'],
                        free_coordinate_count=s['free_coordinate_count'],interface_corner_count=int(s['interface_corner_mask'].sum()),interface_mask_sha256=s['interface_mask_sha256']))
                result,di=update(model,opt,samples,ledger,current,hyper,bins);step=current
                append_event(folder/'training_log.jsonl',dict(local_step=step,global_step=BASE_STEP+step,input_seconds=input_s,**result))
                for row in di:append_event(folder/'interface_diagnostics.jsonl',row)
                state.update(local_step=step,global_step=BASE_STEP+step,last_FM=result['FM'],peak_reserved_bytes=result['cuda_peak_reserved_bytes'],wall_seconds=time.perf_counter()-start)
                if step in FIXED:checkpoint()
                atomic_json(status,state)
                if step==1 or step%25==0:print(ARM,step,'FM',result['FM'],'seconds',round(result['update_seconds'],2),flush=True)
            else:state['status']='COMPLETE' if step==UPDATES else 'SAFE_SEGMENT'
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        C.write(adir/'failure.json',state);raise
    finally:
        pool.shutdown(wait=True)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state.update(local_step=step,global_step=BASE_STEP+step,wall_seconds=time.perf_counter()-start,calls=ledger.snapshot(),resources=observer.summary())
        C.write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()));atomic_json(status,state);C.write(adir/'final_status.json',state)
        owner.unlink()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare-plan','export-initial','train'],required=True);p.add_argument('--stop-step',type=int,choices=FIXED);p.add_argument('--attempt')
    a=p.parse_args();value=prepare_plan() if a.stage=='prepare-plan' else export_initial() if a.stage=='export-initial' else train(a.stop_step,a.attempt)
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()