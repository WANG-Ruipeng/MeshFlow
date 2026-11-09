"""Two exactly1000-update arms, live PLAIN_S1000 + matched AdamW/RNG.

No model construction or calls on import. Root owns resource-aware dispatch.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'r2/src'))
import argparse,copy,json,signal,time,traceback
import numpy as np
from meshflow_control import runtime
import torch
from meshflow_control.models.native import build_model,native_execution
from meshflow_control.seam.model_adapter import seam_forward,corner_masked_fm_loss
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash,rng_state,restore_rng
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy,_check_generator
from meshflow_control.data.io import atomic_json,append_event
from r2.pilots.user_condition_finetune_v1 import trainer as historical
from r2.pilots.seam_pool47_couple_v1.training import resource,save_payload,tensors
from r2.pilots.seam_quality_wave1_wave2_v1.training import weighted_loss,restore_optimizer,parameter_names
from . import training_data as D

EXPERIMENT='R2_SEAM_HINGE_QUALITY_V1'
SCHEMA='r2_seam_hinge_quality_training_v1'

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);atomic_json(path,value)

def validate_optimizer(model,optimizer,step,hyper):
    base.validate_optimizer(model,None,optimizer,'COMMON',D.BASE_STEP+step)
    if historical.optimizer_hyperparameters(optimizer)!=hyper:raise ValueError('AdamW settings changed')

def update(model,optimizer,samples,arm,step,ledger,hyper,bins):
    plain,weighted,edges=D.batch_denominators(samples)
    denominator=weighted if arm=='CLEAN_EDGE_COH' else plain
    validate_optimizer(model,optimizer,step-1,hyper)
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);started=time.perf_counter()
    result=dict(FM=0.,edge_coherence=0.,total_loss=0.,forward_seconds=0.,backward_seconds=0.,
        free_coordinate_denominator=plain,weighted_free_coordinate_denominator=weighted,loss_denominator=denominator,
        edge_denominator=edges,edge_lambda=.1 if arm=='CLEAN_EDGE_COH' else 0.,prior_rho=.25 if arm=='SMOOTH_PRIOR' else 1.,
        head_status='FROZEN_NOT_LOADED_OR_CALLED',occupancy_BCE=None)
    diagnostics=[]
    for micro,sample in enumerate(samples):
        b=tensors(sample);context=dict(local_step=step,global_step=D.BASE_STEP+step,microbatch=micro,
            sample_index=sample['sample_index'],uid=sample['uid'],alpha=sample['alpha'])
        with native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
            prediction,seconds=ledger.call('train_forward',lambda:seam_forward(model,b['x'],b['t'],b['y'],b['valid'],b['C'],b['corner']).float(),context)
            result['forward_seconds']+=seconds
            if arm=='CLEAN_EDGE_COH':
                weights=torch.as_tensor(sample['quality_weight'][None],device='cuda')
                fm=weighted_loss(prediction,b['target'],weights,denominator)
                edge,facts=D.edge_coherence(prediction,sample,edges);loss=fm+D.EDGE_LAMBDA*edge
            else:
                fm=corner_masked_fm_loss(prediction,b['target'],b['valid'],b['corner'],denominator=plain)
                edge=prediction.sum()*0.;facts=dict(edge_count=sample['edge_count'],status='LOGGED_NOT_OPTIMIZED');loss=fm
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM/edge loss')
            value=float(fm.detach());edge_value=float(edge.detach());total=float(loss.detach())
            _,seconds=ledger.call('train_backward',loss.backward,context);result['backward_seconds']+=seconds
        row=D.diagnostic(sample,prediction.detach().cpu().numpy()[0],bins)
        sse=row['quality_weighted_velocity_sse'] if arm=='CLEAN_EDGE_COH' else row['all_free_velocity_sse']
        if not np.isclose(sse/denominator,value,atol=1e-7,rtol=3e-6):raise ValueError('Detached FM disagrees')
        row.update(**context,FM_contribution=value,edge_contribution=edge_value,total_contribution=total,
            edge_facts=facts,batch_loss_denominator=denominator,batch_edge_denominator=edges,
            source_edge_categories=sample['edge_categories'])
        diagnostics.append(row);result['FM']+=value;result['edge_coherence']+=edge_value;result['total_loss']+=total
        del prediction,loss,fm,edge,b
    if any(p.grad is None for p in model.parameters()):raise RuntimeError('Unused generator parameter')
    result['generator_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    _,result['optimizer_seconds']=ledger.call('optimizer_update',optimizer.step,dict(local_step=step,global_step=D.BASE_STEP+step))
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameters')
    validate_optimizer(model,optimizer,step,hyper)
    for category in ('quality_clean_bulk','quality_dirty_bulk','quality_seam_third'):
        count=sum(r[category+'_scalar_count'] for r in diagnostics);sse=sum(r[category+'_velocity_sse'] for r in diagnostics)
        result.update({category+'_scalar_count':count,category+'_velocity_sse':sse,category+'_velocity_mse':sse/count if count else None})
    result['all_free_unweighted_velocity_mse']=sum(r['all_free_velocity_sse'] for r in diagnostics)/plain
    result.update(update_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='One actual formal update incl forward/backward/AdamW; not process VRAM')
    return result,diagnostics

def check_registration(root):
    reg=D.read(root/'training_registration.json')
    if (reg['status'],reg['arms'],reg['updates_per_arm'],reg['batch'],reg['microbatch'])!=('FROZEN',list(D.ARMS),1000,8,1):raise ValueError('Training contract mismatch')
    if (reg['source_global_step'],reg['final_global_step'],reg['edge_lambda'],reg['prior_rho'],reg['source_window'])!=(14000,15000,.1,.25,[1001,2000]):raise ValueError('Science changed')
    for path,sha in reg['protected_hashes'].items():
        if D.digest(path)!=sha:raise ValueError('Protected input/code changed '+path)
    go=D.read(root/'training_dispatch_go.json')
    if not go.get('authorized') or go['training_registration_sha256']!=D.digest(root/'training_registration.json'):raise ValueError('Parent release required')
    for path,sha in go['code_hashes'].items():
        if D.digest(path)!=sha:raise ValueError('Released code changed '+path)
    manifest=D.read(D.resolve(reg['shared_data_manifest']))
    if go['shared_data_manifest_sha256']!=D.digest(D.resolve(reg['shared_data_manifest'])):raise ValueError('Input stream changed')
    if manifest['status']!='COMPLETE' or len(manifest['steps'])!=1000 or manifest['samples']!=8000:raise ValueError('Exact8000 sample stream required')
    if [x['step'] for x in manifest['steps']]!=list(range(1,1001)) or [x['source_continuation_step'] for x in manifest['steps']]!=list(range(1001,2001)):raise ValueError('Stream window changed')
    return reg,manifest,D.read(D.resolve(reg['source_labels']))['labels_by_parent']

def require_cursor(ledger,step):
    expected=dict(train_forward=step*8,train_backward=step*8,optimizer_update=step)
    if ledger.pending or ledger.failed:raise RuntimeError('No replay of failed/pending calls')
    if any(ledger.attempted[k]!=v or ledger.successful[k]!=v for k,v in expected.items()):raise RuntimeError('No implicit rollback')
    seq=[r['context']['global_step'] for r in ledger.rows if r['event']=='CALL_BEGIN' and r['kind']=='optimizer_update']
    if seq!=list(range(14001,14001+step)):raise RuntimeError('Optimizer gap/replay')

def train(arm,root,attempt=None):
    root=D.resolve(root).resolve()
    if arm not in D.ARMS or not root.is_relative_to((ROOT/'experiments').resolve()):raise ValueError('Independent experiment root/arm required')
    reg,manifest,labels=check_registration(root)
    regsha=D.digest(root/'training_registration.json');manifestsha=D.digest(D.resolve(reg['shared_data_manifest']))
    folder=root/'training'/arm;folder.mkdir(parents=True,exist_ok=True);statuspath=folder/'status.json'
    previous=D.read(statuspath) if statuspath.exists() else None
    if previous and previous['status']=='COMPLETE':
        for key in ('checkpoint','inference_export'):
            if D.digest(previous[key]['path'])!=previous[key]['sha256']:raise ValueError('Complete artifact changed')
        return previous
    if previous and previous['status'] not in ('INTERRUPTED_SAFE','SAFE_SEGMENT'):raise RuntimeError('No automatic failed retry')
    step=previous['local_step'] if previous else 0;attempt=historical.safe_attempt_id(attempt or 'attempt_'+str(time.time_ns()))
    adir=folder/'attempts'/attempt;adir.mkdir(parents=True,exist_ok=False);snapshot=resource();write(adir/'resources_dispatch.json',snapshot)
    policy=D.read(root/'dispatch_policy.json');minimum=float(policy['minimum_free_mib_before_load']);write(adir/'dispatch_policy.json',policy)
    if minimum<2048 or snapshot['free_mib'] is None or snapshot['free_mib']<minimum:
        write(adir/'resource_wait.json',dict(status='RESOURCE_WAIT',resource=snapshot));raise RuntimeError('RESOURCE_WAIT no model loaded')
    ledger=historical.CallLedger(folder/'calls.jsonl',dict(train_forward=8000,train_backward=8000,optimizer_update=1000),arm,attempt)
    require_cursor(ledger,step);state=dict(status='LOADING',arm=arm,local_step=step,global_step=14000+step,attempt=attempt,
        registration_sha256=regsha,data_manifest_sha256=manifestsha)
    started=time.perf_counter();stop={'requested':False};owned=False
    def interrupt(*args):stop['requested']=True
    handlers={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    observer=historical.GPUObserver('hinge_'+arm,interval=2.);bins=np.asarray(reg['diagnostic_time_bin_edges'])
    try:
        with historical.exclusive_writer(folder),observer:
            owned=True;write(statuspath,state);torch.set_num_threads(2);runtime.configure_stable_runtime()
            checkpoint=D.resolve(previous['checkpoint']['path'] if previous else reg['source_checkpoint'])
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            if D.digest(checkpoint)!=expected:raise ValueError('Checkpoint bytes changed')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            if payload['model_config']!=MODEL_CONFIG or payload.get('update_in_progress') is not False or payload['global_step']!=14000+step or payload.get('EMA') is not False:raise ValueError('Live state/cursor differs')
            for key,fn,hkey in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')):
                if fn(payload[key])!=payload[hkey]:raise ValueError('Internal checkpoint hash failed '+key)
            if previous:
                if payload['schema']!=SCHEMA or payload['arm']!=arm or payload['registration_sha256']!=regsha or payload['data_manifest_sha256']!=manifestsha:raise ValueError('Resume input mismatch')
            else:
                if (payload['schema'],payload['arm'],payload['local_step'],payload['seam_total_step'])!=('r2_seam_pool47_couple_training_v1','PLAIN3000',1000,2000):raise ValueError('Only livePLAIN_S1000 allowed')
                for pk,rk in (('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')):
                    if payload[pk]!=reg[rk]:raise ValueError('Registered source differs')
            model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval();model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda')
            model._quality_arm='CLEAN_BULK_FM' # helper requests exact all-parameter AdamW restore, not its objective.
            optimizer,receipt=restore_optimizer(model,payload,resume=bool(previous));restore_rng(payload['rng'])
            if state_hash(model.state_dict())!=payload['generator_state_sha256'] or base.tree_hash(rng_state())!=payload['rng_sha256']:raise ValueError('Exact source restore failed')
            hyper=historical.optimizer_hyperparameters(optimizer);validate_optimizer(model,optimizer,step,hyper)
            write(adir/'initialization.json',dict(generator_state_sha256=payload['generator_state_sha256'],optimizer_restore=receipt,
                rng_sha256=payload['rng_sha256'],global_step=payload['global_step'],all_parameters_trainable=True,head_loaded=False,
                scientific_probe_calls=0,trainable_parameters=sum(p.numel() for p in model.parameters()),EMA=False));del payload
            state['status']='TRAINING';write(statuspath,state)
            def checkpoint_save(final=False):
                if state.get('checkpoint',{}).get('local_step')==step:return
                require_cursor(ledger,step);validate_optimizer(model,optimizer,step,hyper)
                gen=_cpu_copy(model.state_dict());_check_generator(gen);opt=_cpu_copy(optimizer.state_dict());rng=_cpu_copy(rng_state())
                data=dict(schema=SCHEMA,experiment=EXPERIMENT,arm=arm,local_step=step,source_continuation_step=1000+step,
                    seam_total_step=2000+step,global_step=14000+step,update_in_progress=False,generator_state=gen,
                    generator_state_sha256=state_hash(gen),optimizer=opt,optimizer_state_sha256=base.tree_hash(opt),
                    optimizer_parameter_names=parameter_names(model),rng=rng,rng_sha256=base.tree_hash(rng),model_config=copy.deepcopy(MODEL_CONFIG),
                    EMA=False,head_state=None,head_state_sha256=None,head_optimizer=None,head_optimizer_state_sha256=None,
                    registration_sha256=regsha,data_manifest_sha256=manifestsha,source_checkpoint_sha256=reg['source_checkpoint_sha256'],
                    stream_state=dict(batch_index=1000+step,next_sample_index=(1000+step)*8),execution=ledger.snapshot())
                dest=root/'checkpoints'/(f'{arm}_step{step}.pt' if final else f'{arm}_rolling.pt');state['checkpoint']=save_payload(dest,data,replace=not final)
                append_event(folder/'checkpoint_history.jsonl',state['checkpoint'])
                if final:
                    exp={k:data[k] for k in ('experiment','arm','local_step','seam_total_step','global_step','generator_state','generator_state_sha256','model_config','EMA')}
                    exp.update(schema='r2_seam_hinge_quality_generator_v1',training_checkpoint_sha256=state['checkpoint']['sha256'],prior_rho=.25 if arm=='SMOOTH_PRIOR' else 1.)
                    state['inference_export']=save_payload(root/'inference'/f'{arm}_generator.pt',exp)
                write(statuspath,state)
            resume_step=step
            for current in range(step+1,1001):
                if current==resume_step+1 or current%10==1 or stop['requested']:
                    rr=resource();append_event(folder/'resource_checks.jsonl',rr)
                    if stop['requested'] or rr['free_mib'] is None or rr['free_mib']<2048:
                        checkpoint_save();state.update(status='INTERRUPTED_SAFE',reason='signal_or_resource_headroom');break
                tick=time.perf_counter();samples=[D.load_sample(r,root,arm,current,m,labels) for m,r in enumerate(manifest['steps'][current-1]['samples'])];input_seconds=time.perf_counter()-tick
                for m,s in enumerate(samples):append_event(folder/'exposure.jsonl',dict(local_step=current,source_continuation_step=1000+current,global_step=14000+current,
                    microbatch=m,uid=s['uid'],sample_index=s['sample_index'],alpha=s['alpha'],t=float(s['t']),N=s['N'],K=s['K'],path=s['path'],
                    input_sha256=manifest['steps'][current-1]['samples'][m]['sha256'],edge_count=s['edge_count'],prior_rho=.25 if arm=='SMOOTH_PRIOR' else 1.))
                result,diagnostics=update(model,optimizer,samples,arm,current,ledger,hyper,bins);step=current
                append_event(folder/'training_log.jsonl',dict(local_step=step,global_step=14000+step,input_seconds=input_seconds,**result))
                for row in diagnostics:append_event(folder/'geometry_diagnostics.jsonl',row)
                state.update(local_step=step,global_step=14000+step,last_FM=result['FM'],last_edge=result['edge_coherence'],
                    peak_reserved_bytes=result['cuda_peak_reserved_bytes'],peak_allocated_bytes=result['cuda_peak_allocated_bytes'],wall_seconds=time.perf_counter()-started)
                if step==1:write(folder/'first_formal_backward.json',dict(status='PASS',arm=arm,no_extra_forward=True,generator_gradient_norm=result['generator_gradient_norm'],
                    edge_coherence=result['edge_coherence'],loss_denominator=result['loss_denominator'],edge_denominator=result['edge_denominator']))
                if step%100==0:checkpoint_save(final=step==1000)
                write(statuspath,state)
                if step==1 or step%10==0:print(json.dumps(dict(arm=arm,step=step,FM=result['FM'],edge=result['edge_coherence'],update_seconds=result['update_seconds'],
                    peak_reserved_mib=result['cuda_peak_reserved_bytes']/2**20,wall_seconds=state['wall_seconds'])),flush=True)
            else:state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(adir/'failure.json',state);raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state.update(local_step=step,global_step=14000+step,wall_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
        write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        if owned:write(statuspath,state)
        write(adir/'final_status.json',state)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=D.ARMS,required=True);p.add_argument('--root',default='experiments/r2_seam_hinge_quality_v1');p.add_argument('--attempt');a=p.parse_args()
    print(json.dumps(train(a.arm,a.root,a.attempt),indent=2))
