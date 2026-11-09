"""One GPU worker; resource admission and shared lock precede CUDA initialization."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
from native_t1 import runtime as _runtime
import argparse
import csv
import time
import traceback
import torch
from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor
from experiments.mf_h1_local0.chair_domain_control_v1.training import process_gpu_lock
from .common import *
from .ledger import Budget

def admission(snapshot,protocol):
    fields=snapshot['gpu']['stdout'].strip().split(',');total,used,free=map(float,fields[-4:-1])
    reserve=protocol['cost_gate']['reserve_mib'];required=1.2*6.8*1024
    result=dict(total_mib=total,used_mib=used,free_mib=free,reserve_mib=reserve,
        prior_training_process_estimate_mib=6.8*1024,required_mib=required,gpu_workers=1,
        reason='Explicit single GPU worker; resource estimate includes context/optimizer/backward, actual peaks monitored',
        status='PASS' if free-reserve>=required else 'WAIT_RESOURCE')
    if result['status']!='PASS':raise RuntimeError('Resource headroom unavailable: '+str(result))
    return result

SAFE_EXIT_CODE=75


def training_stage_status(result):
    status=result.get('status');step=int(result.get('moment_step',-1))
    if status=='DONE' and step==1000:return 'DONE'
    if status=='INTERRUPTED_SAFE' and 0<=step<1000:return 'INTERRUPTED_SAFE'
    raise RuntimeError('Training returned an invalid completion state: '+str((status,step)))


def verify_safe_resume(out,arm):
    """CPU-only proof of a complete checkpoint and covered actual-call ledger."""
    from .training import validate_recovery
    from .checkpoint import validate
    out=Path(out)
    previous=read_json(out/'stages'/f'train_{arm}.json')
    training=read_json(out/'training'/f'{arm}_status.json')
    if previous.get('status')!='INTERRUPTED_SAFE' or training.get('status')!='INTERRUPTED_SAFE':
        raise RuntimeError('Only an explicitly safe interrupted training stage may resume')
    step=int(training['moment_step'])
    if not 0<step<1000 or previous['result']['moment_step']!=step:
        raise RuntimeError('Safe resume requires a positive checkpoint-covered cursor below1000')
    pointer=read_json(out/'training'/f'{arm}_pointer.json');path=Path(pointer['path'])
    if path.resolve().parent!=(out/'checkpoints').resolve():raise ValueError('Recovery pointer escapes current checkpoint directory')
    if digest(path)!=pointer['checkpoint_sha256']:raise ValueError('Recovery file differs from pointer')
    side=read_json(path.with_suffix('.json'))
    if side['checkpoint_sha256']!=pointer['checkpoint_sha256'] or side['metadata']!=pointer['metadata']:
        raise ValueError('Recovery sidecar and committed pointer differ')
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    meta=validate(payload)
    if meta!=pointer['metadata'] or meta['arm']!=arm or meta['moment_step']!=step:
        raise ValueError('Recovery payload identity/cursor differs')
    if meta['protocol_sha256']!=json_hash(read_json(out/'protocol.json')):
        raise ValueError('Recovery protocol differs')
    del payload
    validate_recovery(ledger_rows(out/'training'/f'{arm}_events.jsonl'),step)
    begun={};returned={}
    kinds=('training_forward','training_backward','optimizer_update')
    for ledger in sorted((out/'ledger').glob('*.jsonl')):
        for row in ledger_rows(ledger):
            if row.get('kind') not in kinds or row.get('context',{}).get('arm')!=arm:continue
            cid=row['call_id'];event=row.get('event')
            if event=='CALL_BEGIN':
                if cid in begun:raise RuntimeError('Duplicate training call reservation')
                begun[cid]=row
            elif event=='CALL_RETURN':
                if cid in returned:raise RuntimeError('Duplicate training call return')
                returned[cid]=row
            else:raise RuntimeError('Failed or unknown training-call evidence prevents automatic resume')
    if set(begun)!=set(returned):raise RuntimeError('Unresolved training call prevents resume')
    actual={kind:set() for kind in kinds}
    for cid,row in begun.items():
        context=row['context'];kind=row['kind'];current=int(context['moment_step'])
        if not 1<=current<=step:raise RuntimeError('Actual call is not covered by committed checkpoint')
        key=(current,int(context['microbatch'])) if kind!='optimizer_update' else (current,)
        if key in actual[kind] or returned[cid]['kind']!=kind or returned[cid]['context']!=context:
            raise RuntimeError('Training call identity or unique exposure differs')
        actual[kind].add(key)
    for kind in kinds:
        expected={(current,micro) for current in range(1,step+1) for micro in range(8)} if kind!='optimizer_update' else {(current,) for current in range(1,step+1)}
        if actual[kind]!=expected:raise RuntimeError('Checkpoint does not have exactly covered actual '+kind)
    with (out/'training'/f'{arm}_training_log.csv').open(newline='',encoding='utf-8') as f:logs=list(csv.DictReader(f))
    with (out/'training'/f'{arm}_sample_exposure.csv').open(newline='',encoding='utf-8') as f:exposure=list(csv.DictReader(f))
    if [int(row['moment_step']) for row in logs]!=list(range(1,step+1)):
        raise RuntimeError('Training log cursor is not complete')
    if [int(row['sample_index']) for row in exposure]!=list(range(8*step)):
        raise RuntimeError('Exposure log cursor is not complete')
    return dict(status='SAFE_RESUME_VERIFIED',arm=arm,moment_step=step,
        checkpoint_path=str(path),checkpoint_sha256=pointer['checkpoint_sha256'],
        covered_forward_returns=8*step,covered_backward_returns=8*step,covered_optimizer_returns=step)


def run(out,stage,arm=None,resume=False):
    out=Path(out);key=stage if arm is None else stage+'_'+arm
    path=out/'stages'/f'{key}.json';protocol=read_json(out/'protocol.json')
    resume_proof=None
    if path.exists():
        previous=read_json(path)
        if previous['status']=='DONE':
            if stage=='train' and training_stage_status(previous['result'])!='DONE':
                raise RuntimeError('Completed stage contains incomplete training')
            return previous
        if stage=='train' and resume and previous['status']=='INTERRUPTED_SAFE':
            resume_proof=verify_safe_resume(out,arm)
        else:raise RuntimeError('Existing incomplete stage requires explicit verified safe resume or documented engineering recovery')
    elif resume:
        raise RuntimeError('No explicit safe-interruption stage exists to resume')
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);_runtime.configure_stable_runtime()
    before=resource_snapshot();plan=admission(before,protocol)
    atomic_json(out/'resources'/f'{key}_before.json',dict(resources=before,admission=plan))
    state=dict(status='RUNNING',stage=key,started_unix=time.time(),gpu_workers=1)
    if resume_proof is not None:
        history=out/'recovery'/key/f"resume_from_{resume_proof['moment_step']}.json"
        freeze_json(history,dict(previous=previous,proof=resume_proof,resumed_unix=time.time()))
        state['resume_proof']=resume_proof
    atomic_json(path,state)
    monitor=None;tick=time.perf_counter()
    try:
        with process_gpu_lock(out) as wait:
            locked=resource_snapshot();admission(locked,protocol);atomic_json(out/'resources'/f'{key}_locked.json',locked)
            (out/'training').mkdir(parents=True,exist_ok=True);monitor=GPUMonitor(out);monitor.start()
            if stage=='precheck':
                from .precheck import run as operation
                result=operation(out)
            else:
                from .checkpoint import load_start,load
                from .model_adapter import set_profile
                budget=Budget(out,key)
                if stage=='initial_replay':
                    from .evaluation import initial_replay
                    model,opt,meta,hashes,stream=load_start('MOMENT_XY','cuda',optimizer=False)
                    set_profile(model,False,collect_stats=False);result=initial_replay(out,model,budget)
                elif stage=='train':
                    from .training import train_arm
                    assert arm in ARMS and read_json(out/'gate_results.json')['status']=='PASS'
                    assert read_json(out/'initial_replay/result.json')['status']=='PASS'
                    from .registration import verify_identity
                    verify_identity(out,code=True)
                    if resume_proof is not None:
                        model,opt,meta,hashes,stream=load(resume_proof['checkpoint_path'],'cuda',trainable=True,optimizer=True)
                    else:
                        if (out/'training'/f'{arm}_pointer.json').exists() or list((out/'checkpoints').glob(f'{arm}_moment*_total*.pt')):
                            raise RuntimeError('Existing checkpoint requires explicit verified resume, never inferred latest')
                        model,opt,meta,hashes,stream=load_start(arm,'cuda',optimizer=True)
                    result=train_arm(out,arm,protocol,budget,model,opt,meta,stream)
                elif stage=='eval':
                    from .evaluation import run as evaluate
                    p=out/'checkpoints'/f'{arm}_moment1000_total6000.pt'
                    model,opt,meta,hashes,stream=load(p,'cuda',trainable=False,optimizer=False)
                    assert meta['moment_step']==1000 and meta['conditional_total_step']==6000
                    set_profile(model,False,collect_stats=False);result=evaluate(out,arm+'_END',model,budget,digest(p))
                else:raise ValueError(stage)
            monitor.stop();monitor=None
            completed_status=training_stage_status(result) if stage=='train' else 'DONE'
            state.update(status=completed_status,result=result,wall_seconds=time.perf_counter()-tick,lock_wait_seconds=wait,finished_unix=time.time())
            Budget(out,key+'_finish').snapshot();atomic_json(out/'resources'/f'{key}_after.json',resource_snapshot());atomic_json(path,state)
            print('STAGE_'+state['status']+' '+key,flush=True);return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-tick)
        atomic_json(path,state)
        try:Budget(out,key+'_failure').snapshot()
        except RuntimeError:pass
        raise
    finally:
        if monitor is not None:monitor.stop()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=('precheck','initial_replay','train','eval'),required=True);ap.add_argument('--arm',choices=ARMS)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();result=run(a.out,a.stage,a.arm,resume=a.resume)
    raise SystemExit(SAFE_EXIT_CODE if result['status']=='INTERRUPTED_SAFE' else 0)
