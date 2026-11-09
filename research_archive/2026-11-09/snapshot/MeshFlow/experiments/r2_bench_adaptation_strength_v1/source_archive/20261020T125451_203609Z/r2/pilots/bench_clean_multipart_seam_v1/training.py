"""Fixed +1500 Bench valid-interface multipart SEAM; no forward on import.

--freeze-cpu / --check-cpu perform file/CPU preparation only. GPU work requires
--execute and a parent release bound to the frozen registration. The generator
inherits complete COMMON12000 live AdamW/RNG; the C-only head is newly defined.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
from pathlib import Path
import argparse, copy, csv, json, signal, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
from meshflow_control import runtime
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.data.io import digest, append_event
from r2.pilots.user_condition_finetune_v1 import trainer as historical
from r2.pilots.seam_pool47_couple_v1.training import resource
from r2.pilots.bench_parts_convergence_v1.worker_entry import ScopedIO

DEFAULT_ROOT=ROOT/'experiments/r2_bench_clean_multipart_seam_v1'
EXPERIMENT='R2_BENCH_CLEAN_MULTIPART_SEAM_V1'
SCHEMA='bench_clean_multipart_seam_training_v1'
ARM='BENCH_CLEAN_MULTIPART'
UPDATES=1500
SOURCE_GLOBAL=12000
SOURCE_SEAM_TOTAL=0
_IO=None

def resolve(path):
    path=str(path)
    if len(path)>2 and path[1]==':':path='/mnt/'+path[0].lower()+path[2:].replace('\\','/')
    return Path(path) if Path(path).is_absolute() else ROOT/path

def _plain_read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read(path):
    return (_IO.metadata_reader(_plain_read)(path) if _IO is not None else _plain_read(path))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    (_IO.replace if _IO is not None else os.replace)(tmp,path)

def install_io(root):
    global _IO
    root=Path(root);(root/'training').mkdir(parents=True,exist_ok=True)
    _IO=ScopedIO(root,os.replace,log_path=root/'training/io_guard.jsonl')

def freeze(root):
    from . import head as H
    root=resolve(root).resolve();install_io(root)
    path=root/'training_registration.json'
    if path.exists():return checked(root)[0]
    audit=read(root/'training_audit/start_checkpoint.json')
    if audit['status']!='PASS' or audit['global_step']!=SOURCE_GLOBAL or audit['head_state_present']:
        raise ValueError('Audited full COMMON12000 source required')
    manifest=root/'data/manifest.json'
    candidates=(root/'data/schedule.json',root/'schedule.json')
    schedule=next((p for p in candidates if p.exists()),None)
    if schedule is None:raise FileNotFoundError('Prepared 1500-update data schedule missing')
    evaluation=root/'eval_manifest.json'
    code=[ROOT/'r2/pilots/bench_clean_multipart_seam_v1'/n for n in ('training.py','data.py','head.py','audit_start.py')]
    code += [Path(x) for x in (base.__file__,runtime.__file__)]
    from meshflow_control.seam import model_adapter
    from meshflow_control.models import native
    from r2.pilots.bench_parts_convergence_v1 import worker_entry
    from r2.pilots.bench_parts_convergence_v1 import data as original_data
    from r2.pilots.chair_multipart_seam_v1 import head as chair_head
    from r2.pilots.seam_abc_v1 import slot_data_extension
    from meshflow_control.training import coupling
    code += [Path(model_adapter.__file__),Path(native.__file__),Path(worker_entry.__file__),Path(historical.__file__),
             Path(original_data.__file__),Path(chair_head.__file__),Path(slot_data_extension.__file__),Path(coupling.__file__)]
    reg=dict(schema='bench_clean_multipart_seam_registration_v1',status='FROZEN',experiment=EXPERIMENT,
        arm=ARM,updates=UPDATES,source_global_step=SOURCE_GLOBAL,final_global_step=SOURCE_GLOBAL+UPDATES,
        source_seam_total=SOURCE_SEAM_TOTAL,source_checkpoint=audit['path'],source_checkpoint_sha256=audit['checkpoint_sha256'],
        source_generator_sha256=audit['generator_state_sha256'],source_optimizer_sha256=audit['optimizer_state_sha256'],source_rng_sha256=audit['rng_sha256'],
        batch=8,microbatch=1,hybrid_lambda=.25,objective='original free-only corner FM; no couple or other generator loss',
        generator_optimizer='exact inherited live AdamW, lr1e-5 betas(.9,.95) eps1e-8 wd0 clip1',
        generator_mode='eval with gradients, strict deterministic BF16 forward FP32 params/moments/integration',
        head_config=copy.deepcopy(H.HEAD_CONFIG),head_seed=20261020,head_lr=.001,head_betas=[.9,.95],head_weight_decay=.0001,
        head_loss='unweighted BCE mean over each C edges then mean over 8 C instances; no generator gradient from head',
        normalization_schema=H.NORMALIZATION_SCHEMA,
        head_threshold_logit=0.,head_policy='new C-only context head; source owner labels only during training; no DEV calibration',
        checkpoint_every=250,final_selection='precommitted local1500; no best selection, no automatic extension',
        evaluation_policy='existing48 frozen requests twice (new head / saved old15000 layout); four96-forward probes; no DEV training/calibration',
        comparison_limit='Recipe exploration: filter, xyz augmentation and normalized new head change jointly; historical15000/21000 have unequal update budgets',
        fixed_probe_local_steps=[0,500,1000,1500],fixed_probe_generator_forwards=384,
        model_config=copy.deepcopy(MODEL_CONFIG),training_code_hashes={str(p.resolve()):digest(p) for p in code})
    for name,p in (('manifest',manifest),('input_schedule',schedule),('evaluation_manifest',evaluation),
                   ('head_scaler',root/'data/head_scaler.json'),
                   ('probe_manifest',ROOT/'experiments/r2_bench_parts_convergence_v1/fixed_loss_probe/manifest.json')):
        reg[name]=str(p);reg[name+'_sha256']=digest(p)
    write(path,reg)
    return checked(root)[0]

def checked(root):
    from . import head as H
    root=resolve(root).resolve();reg=read(root/'training_registration.json')
    if root!=DEFAULT_ROOT.resolve() or reg['status']!='FROZEN' or reg['updates']!=UPDATES or reg['source_global_step']!=SOURCE_GLOBAL:
        raise ValueError('Only this fixed root, source and 1500-update budget are authorized')
    if (reg['batch'],reg['microbatch'],reg['hybrid_lambda'])!=(8,1,.25) or reg['head_config']!=H.HEAD_CONFIG:
        raise ValueError('Registered training definition changed')
    for p,expected in reg['training_code_hashes'].items():
        if digest(resolve(p))!=expected:raise ValueError('Frozen training code changed: '+p)
    for key in ('manifest','input_schedule','evaluation_manifest','head_scaler','probe_manifest'):
        if digest(resolve(reg[key]))!=reg[key+'_sha256']:raise ValueError('Frozen input changed: '+key)
    manifest=read(resolve(reg['manifest']));schedule=read(resolve(reg['input_schedule']))
    steps=schedule['steps']
    if len(steps)!=UPDATES or [int(s['step']) for s in steps]!=list(range(1,UPDATES+1)) or any(len(s['samples'])!=8 for s in steps):
        raise ValueError('Exactly 1500x8 frozen training examples required')
    if (len(manifest['rows']),len(manifest['train']),len(manifest['dev']))!=(316,195,39):
        raise ValueError('Full316/eligible195/DEV39 conditions required')
    for row in manifest['rows']:
        if digest(resolve(row['case_path']))!=row['case_sha256']:raise ValueError('Prepared training case changed: '+row['condition_id'])
    if len(read(resolve(reg['evaluation_manifest']))['requests'])!=48:raise ValueError('Exactly48 frozen evaluation requests required')
    return reg,manifest,steps

def stat(error,sample):
    valid=np.asarray(sample['valid_mask'],bool)
    fixed=np.asarray(sample['anchor_ids'])>=0
    free=valid[:,None]&~fixed
    seam=valid&(fixed.sum(1)==2);bulk=valid&(fixed.sum(1)==0)
    return dict(all_free_velocity_sse=float(error[free].sum(dtype=np.float64)),all_free_scalar_count=int(free.sum())*3,
                seam_tip_velocity_sse=float(error[seam,2].sum(dtype=np.float64)),seam_tip_scalar_count=int(seam.sum())*3,
                bulk_velocity_sse=float(error[bulk].sum(dtype=np.float64)),bulk_scalar_count=int(bulk.sum())*9)

def tensors(sample):
    n=int(sample['N']);cv=lambda a:torch.as_tensor(np.asarray(a),device='cuda')
    return dict(x=cv(sample['xt'].reshape(1,n,9)),target=cv(sample['u'].reshape(1,n,9)),
        t=cv(np.asarray([sample['t']],np.float32)),y=cv(np.asarray([n],np.int64)),
        valid=cv(np.asarray(sample['valid_mask'],bool)[None]),known=cv(np.asarray(sample['known_mask'],bool)[None]),
        fixed=cv((sample['anchor_ids']>=0)[None]))

def validate_head(head,opt,step,reg):
    if not head.training or len(opt.param_groups)!=1:raise ValueError('New head train/dropout mode required')
    group=opt.param_groups[0]
    if (group['lr'],tuple(group['betas']),group['weight_decay'])!=(reg['head_lr'],tuple(reg['head_betas']),reg['head_weight_decay']):
        raise ValueError('Head AdamW differs')
    if [id(p) for p in head.parameters()]!=[id(p) for p in group['params']]:raise ValueError('Head parameter order changed')
    for p in head.parameters():
        st=opt.state.get(p,{})
        if step==0 and not st:continue
        if set(st)!={'step','exp_avg','exp_avg_sq'} or float(st['step'])!=step:raise ValueError('Head cursor differs')
        if p.dtype!=torch.float32 or any(st[k].dtype!=torch.float32 or st[k].shape!=p.shape for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Head FP32 moment shape differs')

def require_cursor(ledger,step,head_step):
    expected=dict(train_forward=step*8,train_backward=step*8,optimizer_update=step,
                  head_forward=step*8,head_backward=step*8,head_optimizer_update=head_step)
    if ledger.pending or ledger.failed:raise RuntimeError('Failed or pending calls require explicit recovery; never silently replay')
    if any(ledger.attempted[k]!=v or ledger.successful[k]!=v for k,v in expected.items()):
        raise RuntimeError('Checkpoint and actual call cursor differ; no implicit rollback')
    sequence=[r['context']['global_step'] for r in ledger.rows if r['event']=='CALL_BEGIN' and r['kind']=='optimizer_update']
    if sequence!=list(range(SOURCE_GLOBAL+1,SOURCE_GLOBAL+step+1)):raise RuntimeError('Update sequence differs')

def fixed_probes(root,step,model,reg,cases,ledger):
    """Original96 frozen supervised inputs; no head calls, backward, RNG draws."""
    from . import data as D
    from r2.pilots.bench_parts_convergence_v1.train import _rng_snapshot,_assert_same_rng
    path=root/'training/fixed_loss'/f'step{step}.json'
    if path.exists():
        value=read(path)
        if value['generator_state_sha256']!=state_hash(model.state_dict()):
            raise ValueError('Frozen probe checkpoint identity differs')
        return value
    before=_rng_snapshot();start=time.perf_counter();rows=[]
    probe=read(resolve(reg['probe_manifest']))
    if len(probe['rows'])!=96:raise ValueError('Expected original96 fixed probes')
    for row in probe['rows']:
        if digest(resolve(row['noise_npz']))!=row['noise_sha256']:raise ValueError('Probe input changed')
        noise=D.arrays(row['noise_npz'])
        z=noise.get('z',noise.get('noise'));eps=noise.get('eps',noise.get('epsilon'))
        if z is None or eps is None:raise ValueError('Original probe noise missing')
        s=D.sample_for_probe(cases[row['condition_id']],z,eps,row['t'],row['permutation_seed'])
        b=tensors(s)
        context=dict(local_step=step,global_step=SOURCE_GLOBAL+step,probe_id=row['probe_id'])
        with torch.no_grad(),native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
            pred,_=ledger.call('probe_forward',lambda:seam_forward(model,b['x'],b['t'],b['y'],b['valid'],b['known'],b['fixed']).float(),context)
        error=(pred-b['target']).reshape(int(s['N']),3,3).square().cpu().numpy()
        if not np.isfinite(error).all():raise FloatingPointError('Nonfinite fixed probe')
        rows.append(dict(probe_id=row['probe_id'],condition_id=row['condition_id'],parent_id=row['parent_id'],
                         split=row['group'],part=row['part_type'],time_bin=str(row['t']),t=row['t'],**stat(error,s)))
        del b,pred
    _assert_same_rng(before)
    def aggregate(xs):
        out={}
        for key,label in (('all_free','ALL_free'),('bulk','bulk'),('seam_tip','seam_tip')):
            sse=sum(x[key+'_velocity_sse'] for x in xs);count=sum(x[key+'_scalar_count'] for x in xs)
            out[label]=dict(sse=sse,count=count,MSE=sse/count if count else None)
        return out
    summary={}
    for split in ('TRAIN','DEV'):
        xs=[r for r in rows if r['split']==split]
        summary[split.lower()]=aggregate(xs)
        summary[split.lower()]['parts']={p:aggregate([r for r in xs if r['part']==p]) for p in sorted({r['part'] for r in xs})}
    value=dict(schema='BENCH_CLEAN_FROZEN_ORIGINAL_PROBE_V1',local_step=step,global_step=SOURCE_GLOBAL+step,
               generator_state_sha256=state_hash(model.state_dict()),probe_manifest_sha256=reg['probe_manifest_sha256'],
               rows=rows,**summary,generator_forwards=96,head_forwards=0,backward_calls=0,
               RNG_unchanged=True,wall_seconds=time.perf_counter()-start)
    write(path,value)
    return value

def save_payload(path,payload,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError('Immutable checkpoint exists: '+str(path))
    tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    _IO.replace(tmp,path)
    receipt=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,
        **{k:payload[k] for k in ('schema','local_step','global_step','generator_state_sha256','optimizer_state_sha256','rng_sha256','head_state_sha256','head_optimizer_state_sha256') if k in payload})
    write(path.with_suffix('.json'),receipt);return receipt

def train(root,*,resume=False,target=UPDATES):
    from . import data as D, head as H
    root=resolve(root).resolve();install_io(root)
    reg,manifest,steps=checked(root);reg_sha=digest(root/'training_registration.json')
    go=read(root/'training_dispatch_go.json')
    if not go.get('authorized') or go.get('training_registration_sha256')!=reg_sha:raise ValueError('Parent GPU dispatch release missing')
    if not 0<target<=UPDATES:raise ValueError('Target exceeds authorized fixed budget')
    out=root/'training';out.mkdir(parents=True,exist_ok=True)
    status_path=out/'status.json';previous=read(status_path) if status_path.exists() else None
    if previous and previous['status']=='COMPLETE':
        if digest(resolve(previous['checkpoint']['path']))!=previous['checkpoint']['sha256']:raise ValueError('Final full checkpoint changed')
        return previous
    if previous and (not resume or previous['status'] not in ('INTERRUPTED_SAFE','SAFE_SEGMENT')):raise RuntimeError('Unresolved previous work; no automatic retry')
    if resume and previous is None:raise ValueError('No safe state to resume')
    step=int(previous['local_step']) if previous else 0;head_step=step
    if step>=target:raise ValueError('Target must advance the exact saved cursor')
    attempt='attempt_'+str(time.time_ns());adir=out/'attempts'/attempt;adir.mkdir(parents=True,exist_ok=False)
    limits={k:UPDATES*(8 if k.endswith('forward') or k.endswith('backward') else 1) for k in (
        'train_forward','train_backward','optimizer_update','head_forward','head_backward','head_optimizer_update')}
    limits['probe_forward']=384
    ledger=historical.CallLedger(out/'calls.jsonl',limits,ARM,attempt)
    require_cursor(ledger,step,head_step)
    state=dict(status='LOADING',arm=ARM,local_step=step,global_step=SOURCE_GLOBAL+step,target_local_step=target,
        registration_sha256=reg_sha,pid=os.getpid(),attempt=attempt)
    if previous:state['checkpoint']=previous['checkpoint']
    initial_resource=resource();write(adir/'resources_dispatch.json',initial_resource)
    if initial_resource['free_mib'] is None or initial_resource['free_mib']<2048:raise RuntimeError('RESOURCE_WAIT; no model loaded')
    observer=historical.GPUObserver('bench_clean_multipart_formal_training',interval=2.)
    start=time.perf_counter();stop={'signal':None};owned=False
    def handler(sig,unused):stop['signal']=sig
    old_handlers={sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with historical.exclusive_writer(out),observer:
            owned=True;write(status_path,state);torch.set_num_threads(2);runtime.configure_stable_runtime()
            source=resolve(previous['checkpoint']['path'] if previous else reg['source_checkpoint'])
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            if digest(source)!=expected:raise ValueError('Source checkpoint hash differs')
            p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
            if p['model_config']!=MODEL_CONFIG or p['global_step']!=SOURCE_GLOBAL+step or p.get('EMA') is not False:
                raise ValueError('Full completed live checkpoint identity differs')
            for key,fn,hk in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')):
                expected_hash=p[hk] if previous else reg['source_'+('generator' if key=='generator_state' else key)+'_sha256']
                if fn(p[key])!=expected_hash:raise ValueError('Checkpoint internal hash differs: '+key)
                p[hk]=expected_hash
            if previous:
                if p['schema']!=SCHEMA or p['registration_sha256']!=reg_sha or p['local_step']!=step or p.get('update_in_progress') is not False:
                    raise ValueError('Resume experiment differs')
            elif p['arm']!='COMMON' or p['step']!=12000 or p['stage_step']!=12000:
                raise ValueError('Only full COMMON12000 source is accepted')
            model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
            if previous and p['optimizer_parameter_names']!=base.optimizer_names(model,None):
                raise ValueError('AdamW parameter identity/order changed')
            model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
            opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer'])
            head=H.new_head(seed=reg['head_seed'],scaler=read(resolve(reg['head_scaler']))).train().to('cuda')
            if sum(x.numel() for x in head.parameters())!=18049:raise ValueError('Expected exactly18049 head parameters')
            hopt=torch.optim.AdamW(head.parameters(),lr=reg['head_lr'],betas=tuple(reg['head_betas']),weight_decay=reg['head_weight_decay'])
            if previous:
                if state_hash(p['head_state'])!=p['head_state_sha256'] or base.tree_hash(p['head_optimizer'])!=p['head_optimizer_state_sha256']:
                    raise ValueError('Head checkpoint hash differs')
                head.load_state_dict(p['head_state'],strict=True);hopt.load_state_dict(p['head_optimizer']);head_step=int(p['head_step'])
            restore_rng(p['rng'])
            if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:
                raise ValueError('Exact generator/AdamW/RNG restore failed')
            audit=dict(source_path=str(source),source_sha256=expected,generator_state_sha256=p['generator_state_sha256'],
                optimizer_state_sha256=p['optimizer_state_sha256'],rng_sha256=p['rng_sha256'],global_step=p['global_step'],
                generator_AdamW_RNG_exact=True,head_fresh=not resume,head_config=H.HEAD_CONFIG,head_parameters=sum(x.numel() for x in head.parameters()),
                head_initial_state_sha256=state_hash(head.state_dict()),head_no_generator_gradient=True)
            write(adir/'initialization.json',audit);del p
            base.validate_optimizer(model,None,opt,'COMMON',SOURCE_GLOBAL+step);validate_head(head,hopt,head_step,reg)
            cases={r['condition_id']:D.load_case(r) for r in manifest['rows']}
            def checkpoint_save(final=False):
                require_cursor(ledger,step,head_step)
                base.validate_optimizer(model,None,opt,'COMMON',SOURCE_GLOBAL+step);validate_head(head,hopt,head_step,reg)
                g=_cpu_copy(model.state_dict());_check_generator(g);o=_cpu_copy(opt.state_dict());r=_cpu_copy(rng_state())
                hs=_cpu_copy(head.state_dict());ho=_cpu_copy(hopt.state_dict())
                payload=dict(schema=SCHEMA,experiment=EXPERIMENT,arm=ARM,local_step=step,global_step=SOURCE_GLOBAL+step,
                    seam_total_step=SOURCE_SEAM_TOTAL+step,head_step=head_step,update_in_progress=False,EMA=False,
                    generator_state=g,generator_state_sha256=state_hash(g),optimizer=o,optimizer_state_sha256=base.tree_hash(o),
                    optimizer_parameter_names=base.optimizer_names(model,None),rng=r,rng_sha256=base.tree_hash(r),model_config=copy.deepcopy(MODEL_CONFIG),
                    head_state=hs,head_state_sha256=state_hash(hs),head_optimizer=ho,head_optimizer_state_sha256=base.tree_hash(ho),head_config=H.HEAD_CONFIG,
                    registration_sha256=reg_sha,source_checkpoint_sha256=reg['source_checkpoint_sha256'],execution=ledger.snapshot(),
                    stream_state=dict(batch_index=step,next_sample_index=8*step))
                dest=out/('checkpoint_step1500.pt' if final else 'rolling.pt')
                state['checkpoint']=save_payload(dest,payload,replace=not final)
                append_event(out/'checkpoint_history.jsonl',state['checkpoint'])
                if final:
                    export={k:payload[k] for k in ('experiment','arm','local_step','global_step','seam_total_step','generator_state','generator_state_sha256','model_config','EMA')}
                    export.update(schema='bench_clean_multipart_seam_generator_v1',state_sha256=payload['generator_state_sha256'],training_checkpoint_sha256=state['checkpoint']['sha256'])
                    state['inference_export']=save_payload(root/'inference/BENCH_CLEAN_MULTIPART_S1500_generator.pt',export)
                    he={k:payload[k] for k in ('experiment','arm','local_step','global_step','head_state','head_state_sha256','head_config')}
                    he.update(schema='bench_clean_multipart_seam_occupancy_v1',threshold_logit=0.,
                              normalization_schema=H.NORMALIZATION_SCHEMA,head_scaler_sha256=reg['head_scaler_sha256'],
                              training_checkpoint_sha256=state['checkpoint']['sha256'])
                    state['occupancy_export']=save_payload(root/'inference/BENCH_CLEAN_MULTIPART_S1500_occupancy.pt',he)
                write(status_path,state)
            state['status']='TRAINING';write(status_path,state)
            if step in reg['fixed_probe_local_steps']:
                fixed_probes(root,step,model,reg,cases,ledger)
            resume_step=step
            for entry in steps[step:target]:
                if stop['signal'] is not None:
                    checkpoint_save();state.update(status='INTERRUPTED_SAFE',signal=stop['signal']);break
                current=int(entry['step']);tick=time.perf_counter()
                if current==resume_step+1 or current%25==1:
                    rr=resource();append_event(out/'resource_checks.jsonl',dict(local_step_before=step,**rr))
                    if rr['free_mib'] is None or rr['free_mib']<2048:
                        checkpoint_save();state.update(status='INTERRUPTED_SAFE',reason='measured_vram_headroom_below_2048MiB_or_unknown');break
                samples=[D.sample_for_training(cases[row['condition_id']],row) for row in entry['samples']]
                den=sum(int(s['free_coordinate_count']) for s in samples);eden=sum(len(s['occupied_labels']) for s in samples)
                if any(len(s['occupied_labels'])==0 for s in samples):raise ValueError('Frozen eligible pool requires8 nonempty head samples')
                input_seconds=time.perf_counter()-tick;batch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                opt.zero_grad(set_to_none=True);hopt.zero_grad(set_to_none=True);fm=0.;bce_total=0.;rows=[];head_counts=dict(TP=0,FN=0,FP=0,TN=0)
                for mi,(s,row) in enumerate(zip(samples,entry['samples'])):
                    context=dict(local_step=current,global_step=SOURCE_GLOBAL+current,microbatch=mi,condition_id=row['condition_id'])
                    b=tensors(s)
                    with native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
                        pred,_=ledger.call('train_forward',lambda:seam_forward(model,b['x'],b['t'],b['y'],b['valid'],b['known'],b['fixed']).float(),context)
                        loss=corner_masked_fm_loss(pred,b['target'],b['valid'],b['fixed'],denominator=den)
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
                        ledger.call('train_backward',loss.backward,context)
                    err=(pred.detach()-b['target']).reshape(int(s['N']),3,3).square().cpu().numpy();st=stat(err,s)
                    if not np.isclose(st['all_free_velocity_sse']/den,float(loss.detach()),rtol=4e-6,atol=1e-7):raise ValueError('FM SSE/denominator accounting differs')
                    fm+=float(loss.detach())
                    features=torch.as_tensor(s['head_features'],device='cuda',dtype=torch.float32)
                    labels=torch.as_tensor(s['occupied_labels'],device='cuda',dtype=torch.float32)
                    with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,_=ledger.call('head_forward',lambda:head(features).float(),context)
                        bce=F.binary_cross_entropy_with_logits(logits.float(),labels,reduction='mean')/8
                    if not bool(torch.isfinite(bce)):raise FloatingPointError('Nonfinite C-only BCE')
                    ledger.call('head_backward',bce.backward,context);bce_total+=float(bce.detach())
                    prediction=logits.detach()>=0;truth=labels.bool()
                    for k,v in (('TP',prediction&truth),('FN',~prediction&truth),('FP',prediction&~truth),('TN',~prediction&~truth)):head_counts[k]+=int(v.sum())
                    tm=float(s['t']);tb=min(4,int(tm*5));part=row.get('part_type',row.get('part',s.get('part_type','UNKNOWN')))
                    result={**row,**context,'t':tm,'time_bin':f'{tb/5:.1f}-{(tb+1)/5:.1f}','part':part,'N':int(s['N']),'K':int(s['K']),
                        'head_edge_count':len(labels),'FM_contribution':float(loss.detach()),'BCE_contribution':float(bce.detach()),**st}
                    append_event(out/'exposure.jsonl',result);rows.append(st)
                    del pred,loss,b,features,labels,logits,bce
                if any(p.grad is None for p in model.parameters()):raise RuntimeError('Unused generator parameter')
                if any(p.grad is None for p in head.parameters()):raise RuntimeError('Unused head parameter')
                gnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                hnorm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
                ctx=dict(local_step=current,global_step=SOURCE_GLOBAL+current)
                ledger.call('optimizer_update',opt.step,ctx);ledger.call('head_optimizer_update',hopt.step,ctx)
                step=current;head_step+=1
                base.validate_optimizer(model,None,opt,'COMMON',SOURCE_GLOBAL+step);validate_head(head,hopt,head_step,reg)
                if any(not bool(torch.isfinite(p).all()) for p in list(model.parameters())+list(head.parameters())):raise FloatingPointError('Nonfinite updated parameter')
                sums={k:sum(r[k] for r in rows) for k in rows[0]}
                result=dict(local_step=step,global_step=SOURCE_GLOBAL+step,head_step=head_step,FM=fm,occupancy_BCE=bce_total,
                    free_coordinate_denominator=den,head_edge_denominator=eden,head_condition_denominator=8,**sums,**head_counts,
                    generator_gradient_norm=gnorm,head_gradient_norm=hnorm,input_seconds=input_seconds,update_seconds=time.perf_counter()-batch_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='one formal batch including generator/head backward and both AdamW updates; not process VRAM')
                append_event(out/'training_log.jsonl',result)
                state.update(local_step=step,global_step=SOURCE_GLOBAL+step,head_step=head_step,last=result,elapsed_seconds=time.perf_counter()-start)
                if step in reg['fixed_probe_local_steps']:
                    fixed_probes(root,step,model,reg,cases,ledger)
                if step%250==0:checkpoint_save(final=step==UPDATES)
                write(status_path,state)
                if step in (1,5,25) or step%100==0:
                    write(out/f'resources_step{step}.json',resource());print(json.dumps(result),flush=True)
                if step==1:write(out/'first_formal_backward.json',dict(status='PASS',no_extra_model_forward=True,**result))
            else:
                if step%250:checkpoint_save()
                state['status']='COMPLETE' if step==UPDATES else 'SAFE_SEGMENT'
            require_cursor(ledger,step,head_step);write(status_path,state)
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(adir/'failure.json',state);raise
    finally:
        for sig,old in old_handlers.items():signal.signal(sig,old)
        state.update(local_step=step,global_step=SOURCE_GLOBAL+step,head_step=head_step,elapsed_seconds=time.perf_counter()-start,calls=ledger.snapshot(),resources=observer.summary())
        write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        if owned:write(status_path,state)
        write(adir/'final_status.json',state)
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(DEFAULT_ROOT))
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--freeze-cpu',action='store_true');group.add_argument('--check-cpu',action='store_true');group.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--target',type=int,default=UPDATES);args=p.parse_args()
    root=resolve(args.root)
    if args.freeze_cpu:result=freeze(root)
    elif args.check_cpu:
        reg,manifest,steps=checked(root);result=dict(status='PASS',conditions=len(manifest['rows']),updates=len(steps),registration_sha256=digest(root/'training_registration.json'),model_constructions=0,model_forwards=0)
    else:result=train(root,resume=args.resume,target=args.target)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
