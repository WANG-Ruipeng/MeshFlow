"""Paired EXP67 -> source-contour readout continuation, with original FM diagnostics.

The full world-coordinate generator and new contour branch receive the original FM.
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
from .model import build_candidate
from .topology import TopologyInputs
from meshflow_control.checkpoints import state_hash,rng_state,restore_rng
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy,_check_generator as _check_old_generator
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.dataset import collate,model_inputs,training_time
from meshflow_control.data.io import read_json,atomic_json,freeze_json,append_event,digest,save_npz,json_hash,array_hash
from r2.pilots.user_data_scaling_32_67_v1 import training_data as olddata
from r2.pilots.user_condition_finetune_v1 import trainer as support

ARM=C.TRAIN_LABEL
BASE_STEP=6000
UPDATES=6000
FIXED=(1000,3000,6000)
SCHEMA='r2_full_contour_readout_training_v1'


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([C.VERSION,'training',*parts],separators=(',',':')).encode()).digest()[:4],'little')


def registration():
    r=C.load_registration()
    if r.get('status')!='FROZEN' or (r['updates_per_arm'],r['batch'],r['microbatch'])!=(UPDATES,8,1):
        raise ValueError('Only frozen 6000-update batch8/micro1 registration accepted')
    if r.get('source_global_step',BASE_STEP)!=BASE_STEP:raise ValueError('EXP67 step6000 required')
    if digest(C.resolve(r['source_data']))!=r['source_data_sha256']:raise ValueError('Source dataset changed')
    if digest(C.resolve(r['interface_manifest_path']))!=r['interface_manifest_sha256']:raise ValueError('Interface label manifest changed')
    for path_key,hash_key in (('replay_exposure_path','replay_exposure_sha256'),('topology_manifest_path','topology_manifest_sha256'),('baseline_registration_path','baseline_registration_sha256')):
        if digest(C.resolve(r[path_key]))!=r[hash_key]:raise ValueError('Frozen replay/topology asset changed: '+path_key)
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
    raise RuntimeError('Plan regeneration forbidden; reuse original frozen bytes')


def batch(d,masks,records,index,plan_sha,topology,expected_exposures):
    folder=C.BASELINE/'ot'/'FULL_SEAT';path=folder/f'{index:05d}.npz'
    receipt=path.with_suffix('.json')
    if not path.is_file() or not receipt.is_file():raise RuntimeError('Missing original OT cache; no recomputation')
    identity=dict(records_sha256=json_hash(records),plan_sha256=plan_sha,batch_index=index)
    r=read_json(receipt)
    if r['identity']!=identity or r['sha256']!=digest(path):raise ValueError('Read-only original OT cache changed')
    with np.load(path,allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    builder=olddata.Builder(d,maps)
    samples=[masks.attach(olddata.validate_sample(builder.sample(row),row)) for row in records]
    if builder.new_maps:raise ValueError('Unexpected new matching')
    for sample in samples:
        old=expected_exposures[int(sample['sample_index'])]
        for key in ('input_bytehash','label_bytehash','interface_mask_sha256','ot_cache_key'):
            if sample[key]!=old[key]:raise ValueError('Paired original exposure mismatch: '+key)
        sample['boundary_edge_mask']=topology.for_training(sample)
        sample['boundary_edge_mask_sha256']=array_hash(sample['boundary_edge_mask'])
    return samples


def _check_generator(state):
    old={n:v for n,v in state.items() if not n.startswith('contour_readout.')}
    new={n:v for n,v in state.items() if n.startswith('contour_readout.')}
    _check_old_generator(old)
    if sum(v.numel() for v in new.values())!=412032 or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in new.values()):raise ValueError('Candidate branch shape/dtype/finite contract differs')


def parameter_groups(model):
    old=[(n,p) for n,p in model.named_parameters() if not n.startswith('contour_readout.')]
    new=[(n,p) for n,p in model.named_parameters() if n.startswith('contour_readout.')]
    if len(old)!=194 or sum(p.numel() for _,p in new)!=412032:raise ValueError('Unexpected old/new parameter ownership')
    return old,new


def optimizer_names(model):
    return [['generator.'+n for n,p in group] for group in parameter_groups(model)]


def _validate_optimizer(model,opt,global_step,hyper):
    local=global_step-BASE_STEP
    if not 0<=local<=UPDATES or model.training or len(opt.param_groups)!=2:raise ValueError('Invalid dual-clock optimizer')
    if support.optimizer_hyperparameters(opt)!=hyper:raise ValueError('Inherited AdamW hyperparameters changed')
    for gi,((named),group) in enumerate(zip(parameter_groups(model),opt.param_groups)):
        if [id(p) for n,p in named]!=[id(p) for p in group['params']]:raise ValueError('Parameter group identity/order differs')
        if (group['lr'],tuple(group['betas']),group['eps'],group['weight_decay'])!=(1e-5,(.9,.95),1e-8,0.):raise ValueError('Changed AdamW hyperparameters')
        expected_step=global_step if gi==0 else local
        for name,p in named:
            state=opt.state.get(p,{})
            if p.dtype!=torch.float32 or not p.requires_grad:raise ValueError('Nontrainable or nonFP32 '+name)
            if expected_step==0 and not state:continue
            if set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=expected_step:raise ValueError('Actual AdamW clock differs '+name)
            if any(state[k].shape!=p.shape or state[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq')):raise ValueError('Moment identity differs '+name)
    return True


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
    den=support.validate_samples(samples,'FULL_SEAT',step,effective_batch=True)
    if any(s['K']<2 or s['condition_type']!='SEAT_FULL' for s in samples):raise ValueError('Non-FULL training exposure')
    g=BASE_STEP+step;_validate_optimizer(model,opt,g-1,hyper)
    torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    result=dict(FM=0.,total_loss=0.,free_coordinate_denominator=den,forward_seconds=0.,backward_seconds=0.,sample_losses=[])
    diagnostics=[]
    for micro,s in enumerate(samples):
        b=collate([s],'cuda');ctx=dict(local_step=step,global_step=g,microbatch=micro,sample_index=s['sample_index'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'])
        with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
            edge=torch.from_numpy(s['boundary_edge_mask'][None]).to('cuda')
            v,sec=ledger.call('train_forward',lambda:model(*model_inputs(b),boundary_edge_mask=edge).float(),ctx);result['forward_seconds']+=sec
            loss=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=den)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
            value=float(loss.detach());_,sec=ledger.call('train_backward',loss.backward,ctx);result['backward_seconds']+=sec
        di=diagnostic(s,v,b['u'],bins);di.update(local_step=step,global_step=g,FM_contribution=value,batch_free_coordinate_denominator=den)
        if not np.isclose(di['all_free_velocity_sse']/den,value,rtol=2e-6,atol=1e-8):raise ValueError('Detached diagnostic does not partition original FM')
        diagnostics.append(di);result['FM']+=value;result['sample_losses'].append(dict(sample_index=s['sample_index'],FM_contribution=value));del v,loss,b
    params=list(model.parameters())
    if any(p.grad is None for p in params):raise RuntimeError('Unused generator parameter')
    old_params,new_params=parameter_groups(model)
    def gradnorm(named):return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for _,p in named])))
    result['old_gradient_norm']=gradnorm(old_params);result['new_gradient_norm']=gradnorm(new_params)
    result['new_parameter_gradient_norms']={n:float(torch.linalg.vector_norm(p.grad.detach().float())) for n,p in new_params} if step<=2 else {}
    result['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    result['actual_clip_coefficient']=min(1.,1./(result['gradient_norm']+1e-6))
    if step==1:
        prior=json.loads((C.BASELINE/'training/FULL_SEAT/training_log.jsonl').open().readline())
        match=result['sample_losses']==prior['sample_losses'] and result['FM']==prior['FM']
        zero=getattr(model,'initial_zero_observations',[])
        if not match or len(zero)!=8 or any(not x['zero_residual_exact'] or not x['unchanged_hidden_exact'] for x in zero):raise ValueError('Normal first-batch zero initialization comparison failed')
        C.write(C.OUT/'audits/zero_initialization.json',dict(status='PASS',normal_training_microbatches=8,zero_residual_exact=True,unchanged_hidden_exact=True,
            first_batch_FM_matches=match,old_state_sha256_matches=True,baseline_first_batch_FM=prior['FM'],candidate_first_batch_FM=result['FM'],observations=zero,
            interpretation='No extra forward; exact first normal training batch loss + zero injection. Step0 rollouts inferred equivalent and reused, not newly measured.'))
        model.initial_zero_observations=None
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
    planpath=C.resolve(reg['replay_plan_path']);plan=read_json(planpath);plan_sha=digest(planpath)
    if plan_sha!=reg['replay_plan_sha256'] or plan['arm']!='FULL_SEAT' or len(plan['records'])!=UPDATES*8:raise ValueError('Original replay plan differs')
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
    topology=TopologyInputs();expected_exposures=[json.loads(line) for line in (C.BASELINE/'training/FULL_SEAT/exposure.jsonl').read_text().splitlines()]
    if len(expected_exposures)!=UPDATES*8:raise ValueError('Expected 48000 original exposures')
    observer=support.GPUObserver('full_contour_train',interval=3.);pending={};pool=ThreadPoolExecutor(max_workers=2)
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
            model=build_candidate(copy.deepcopy(MODEL_CONFIG),trainable=True,seed=reg['new_module_seed']).eval()
            old_named,new_named=parameter_groups(model)
            if previous:
                if p['optimizer_parameter_names']!=optimizer_names(model):raise ValueError('Candidate optimizer order differs')
                model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
                opt=torch.optim.AdamW([dict(params=[p for n,p in old_named]),dict(params=[p for n,p in new_named])],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
                opt.load_state_dict(p['optimizer'])
                if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256']:raise ValueError('Candidate exact resume failed')
            else:
                if p['optimizer_parameter_names']!=[['generator.'+n for n,v in old_named]]:raise ValueError('Old optimizer identity/order differs')
                incompatible=model.load_state_dict(p['generator_state'],strict=False)
                if incompatible.unexpected_keys or set(incompatible.missing_keys)!={n for n,v in new_named}:raise ValueError('Unexpected incomplete source load')
                model.to('cuda')
                inherited={n:v for n,v in model.state_dict().items() if not n.startswith('contour_readout.')}
                if state_hash(inherited)!=p['generator_state_sha256']:raise ValueError('Old weights not exact')
                opt=torch.optim.AdamW([dict(params=[v for n,v in old_named])],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
                opt.load_state_dict(p['optimizer'])
                if base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256']:raise ValueError('Old AdamW restore not exact')
                opt.add_param_group(dict(params=[v for n,v in new_named],**{k:v for k,v in opt.param_groups[0].items() if k!='params'}))
                model.initial_zero_observations=[]
                C.write(adir/'new_parameter_initialization.json',dict(new_parameter_names=[n for n,v in new_named],
                    new_parameter_count=sum(v.numel() for n,v in new_named),new_state_sha256=state_hash({n:v for n,v in model.state_dict().items() if n.startswith('contour_readout.')}),
                    old_state_sha256=state_hash(inherited),old_optimizer_sha256=p['optimizer_state_sha256'],new_optimizer_state_empty=all(not opt.state.get(v) for n,v in new_named)))
            if step==0:model.initial_zero_observations=[]
            restore_rng(p['rng'])
            if base.tree_hash(rng_state())!=p['rng_sha256']:raise ValueError('Exact original RNG restoration failed')
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
                       optimizer_parameter_names=optimizer_names(model),model_config=copy.deepcopy(MODEL_CONFIG),protocol=reg,lineage=lineage,plan_sha256=plan_sha,
                       stream_state=dict(batch_index=step,next_sample_index=step*8,parent_exposure_counts=dict(Counter(x['uid'] for x in plan['records'][:step*8]))),execution=ledger.snapshot(),EMA=False)
                state['checkpoint']=save(C.OUT/'checkpoints'/f'{ARM}_step{step}.pt',p)
                if step in FIXED:
                    label=f'CONTOUR_S{step}';q={k:p[k] for k in ('experiment','arm','local_step','global_step','generator_state','generator_state_sha256','model_config','lineage')}
                    q.update(schema='r2_full_contour_readout_generator_v1',inference_contract=dict(readout_mode='finite_surface_source_contour',forward='BF16 with FP32 branch/add',parameters='FP32',integration='FP32',source_topology_required=True),training_checkpoint_sha256=state['checkpoint']['sha256'],EMA=False)
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
                    if bidx not in pending:pending[bidx]=pool.submit(batch,d,masks,plan['records'][bidx*8:(bidx+1)*8],bidx,plan_sha,topology,expected_exposures)
                samples=pending.pop(current-1).result();input_s=time.perf_counter()-began
                if flag['stop']:
                    checkpoint();state['status']='INTERRUPTED_SAFE';state['reason']='signal_after_input_before_model';break
                for s in samples:
                    append_event(folder/'exposure.jsonl',dict(local_step=current,global_step=BASE_STEP+current,sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],condition_type=s['condition_type'],
                        N=int(s['y']),K=s['K'],alpha=s['alpha'],t=float(s['t']),input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key'],
                        free_coordinate_count=s['free_coordinate_count'],interface_corner_count=int(s['interface_corner_mask'].sum()),interface_mask_sha256=s['interface_mask_sha256'],boundary_edge_mask_sha256=s['boundary_edge_mask_sha256'],paired_baseline_hashes_match=True))
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stop-step',type=int,choices=FIXED,required=True);p.add_argument('--attempt')
    a=p.parse_args();print(json.dumps(train(a.stop_step,a.attempt),indent=2))

if __name__=='__main__':main()
